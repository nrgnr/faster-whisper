# Faster-Whisper Architecture Analysis

## 1. PROJECT OVERVIEW

### Problem Statement
This project provides a high-performance reimplementation of OpenAI's Whisper speech-to-text model using CTranslate2, an optimized inference engine for Transformer models. It solves the problem of slow transcription speed and high memory usage in the original Whisper implementation.

### High-Level Architecture
The system follows a **pipeline-based architecture** with clear separation of concerns:

```
Audio Input → Audio Decoding → Feature Extraction → Model Inference → Tokenization → Post-processing → Transcription Output
```

**Core Components:**
- **Audio Processing Layer**: Decodes audio files using PyAV (FFmpeg bundled)
- **Feature Extraction Layer**: Converts audio to Mel spectrograms
- **Model Inference Layer**: Uses CTranslate2 for optimized Whisper model execution
- **Tokenization Layer**: Handles text encoding/decoding and special tokens
- **VAD Layer**: Optional voice activity detection using Silero VAD
- **Post-processing Layer**: Word-level timestamps, segment splitting, timestamp restoration

### Intended Runtime Model

**Process Model:**
- Single-process Python application
- No multi-process architecture (unlike some ASR systems)
- Thread-based parallelism for multi-GPU scenarios (`num_workers` parameter)

**Threading Model:**
- Main thread: Orchestrates transcription workflow
- Worker threads: When `num_workers > 1`, enables concurrent model.generate() calls
- CPU threads: Controlled via `cpu_threads` (affects CTranslate2 internal threading)
- ONNX Runtime threads: VAD model uses single-threaded execution (inter_op_num_threads=1, intra_op_num_threads=1)

**Lifecycle:**
1. **Initialization**: Model loading, tokenizer setup, feature extractor initialization
2. **Transcription**: Iterative segment processing (30-second windows by default)
3. **Cleanup**: Automatic via Python GC (explicit GC call in audio decoding for PyAV memory)

**Memory Model:**
- Models loaded once and reused across transcriptions
- Generator-based segment yielding (memory-efficient for long audio)
- VAD model cached via `@functools.lru_cache`
- Encoder output can be moved to CPU for multi-GPU scenarios

### Component Interactions

```
WhisperModel (orchestrator)
    ├── FeatureExtractor: Converts audio → Mel spectrograms
    ├── CTranslate2 Whisper Model: Encodes features, generates tokens
    ├── Tokenizer: Encodes/decodes text, manages special tokens
    ├── VAD (optional): Filters silence before transcription
    └── Post-processing: Word timestamps, segment splitting, timestamp restoration

BatchedInferencePipeline (alternative orchestrator)
    └── Wraps WhisperModel for batch processing
        └── Uses same components but processes multiple chunks in parallel
```

---

## 2. FILE-BY-FILE BREAKDOWN

### Core Package: `faster_whisper/`

#### `__init__.py`
- **Purpose**: Public API exports
- **Type**: API/Interface
- **Exports**:
  - `WhisperModel`: Main transcription class
  - `BatchedInferencePipeline`: Batch processing wrapper
  - `decode_audio`: Audio decoding utility
  - `available_models`: Model listing
  - `download_model`: Model download utility
  - `format_timestamp`: Timestamp formatting
  - `__version__`: Version string

#### `version.py`
- **Purpose**: Version information
- **Type**: Configuration
- **Key Data**: `__version__ = "1.2.1"`

#### `audio.py`
- **Purpose**: Audio file decoding and preprocessing
- **Type**: Core Logic (Infrastructure)
- **Key Functions**:
  - `decode_audio()`: Main entry point - decodes audio files to float32 numpy arrays
  - `pad_or_trim()`: Pads/trims Mel features to 3000 frames
  - `_ignore_invalid_frames()`: Filters corrupted audio frames
  - `_group_frames()`: Groups audio frames using AudioFifo
  - `_resample_frames()`: Resamples audio to 16kHz mono
- **Dependencies**: PyAV (av), numpy
- **Notes**: 
  - Handles stereo splitting via `split_stereo` parameter
  - Explicit GC call to free PyAV resampler memory (issue #390 workaround)
  - No FFmpeg system dependency (bundled in PyAV)

#### `feature_extractor.py`
- **Purpose**: Converts audio waveforms to Mel spectrogram features
- **Type**: Core Logic
- **Key Class**: `FeatureExtractor`
- **Key Methods**:
  - `__init__()`: Initializes with sampling rate, hop length, chunk length
  - `__call__()`: Main processing - computes log-Mel spectrogram
  - `get_mel_filters()`: Static method - generates Mel filter bank weights
  - `stft()`: Static method - Short-Time Fourier Transform implementation
- **Parameters**:
  - Default: 80 mel bins, 16kHz sampling, 160 hop length, 30s chunks
  - Configurable via `preprocessor_config.json` in model directory
- **Output**: Float32 numpy array of shape `(n_mels, n_frames)`

#### `tokenizer.py`
- **Purpose**: Text tokenization and special token management
- **Type**: Core Logic
- **Key Class**: `Tokenizer`
- **Key Methods**:
  - `encode()`: Text → token IDs
  - `decode()`: Token IDs → text (filters timestamps)
  - `decode_with_timestamps()`: Includes timestamp tokens in output
  - `split_to_word_tokens()`: Splits tokens into words (handles space-based and Unicode-based languages)
  - `split_tokens_on_spaces()`: For languages with spaces
  - `split_tokens_on_unicode()`: For languages without spaces (zh, ja, th, lo, my, yue)
- **Properties** (cached):
  - `transcribe`, `translate`, `sot`, `sot_lm`, `sot_prev`, `eot`, `no_timestamps`, `no_speech`
- **Special Tokens**: Manages Whisper's special tokens (`<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, etc.)
- **Non-speech Tokens**: `non_speech_tokens` property returns tokens to suppress (punctuation, symbols, speaker tags)

#### `transcribe.py`
- **Purpose**: Main transcription orchestration and model inference
- **Type**: Core Logic (Primary API)
- **Key Classes**:
  - `WhisperModel`: Main transcription class
  - `BatchedInferencePipeline`: Batch processing wrapper
  - `Word`: Dataclass for word-level timestamps
  - `Segment`: Dataclass for transcription segments
  - `TranscriptionOptions`: Dataclass for transcription parameters
  - `TranscriptionInfo`: Dataclass for transcription metadata
- **Key Methods (WhisperModel)**:
  - `__init__()`: Model initialization, downloads model if needed
  - `transcribe()`: Main transcription entry point
  - `generate_segments()`: Iterative segment generation (30s windows)
  - `encode()`: Encodes Mel features to encoder output
  - `generate_with_fallback()`: Temperature fallback for failed segments
  - `get_prompt()`: Constructs decoder prompt with previous tokens
  - `add_word_timestamps()`: Extracts word-level timestamps using alignment
  - `find_alignment()`: Aligns tokens to audio frames using model.align()
  - `detect_language()`: Language detection using encoder output
  - `_split_segments_by_timestamps()`: Splits tokens by timestamp boundaries
- **Key Methods (BatchedInferencePipeline)**:
  - `transcribe()`: Batch transcription with VAD chunking
  - `forward()`: Processes batch of features
  - `generate_segment_batched()`: Batch generation
  - `_batched_segments_generator()`: Generator for batched segments
- **Helper Functions**:
  - `restore_speech_timestamps()`: Restores original timestamps after VAD filtering
  - `get_compression_ratio()`: Detects repetitive text (hallucination detection)
  - `get_suppressed_tokens()`: Builds token suppression list
  - `merge_punctuations()`: Merges punctuation with adjacent words
- **Dependencies**: ctranslate2, numpy, tokenizers, huggingface_hub

#### `utils.py`
- **Purpose**: Utility functions and model management
- **Type**: Infrastructure/Utility
- **Key Functions**:
  - `available_models()`: Returns list of available model names
  - `download_model()`: Downloads CTranslate2 models from Hugging Face Hub
  - `format_timestamp()`: Formats seconds to HH:MM:SS.mmm
  - `get_logger()`: Returns module logger
  - `get_assets_path()`: Returns path to assets directory
  - `get_end()`: Extracts last word end time from segments
- **Key Data**: `_MODELS` dictionary mapping model names to HF Hub IDs
- **Dependencies**: huggingface_hub, tqdm

#### `vad.py`
- **Purpose**: Voice Activity Detection using Silero VAD
- **Type**: Core Logic (Optional Feature)
- **Key Classes**:
  - `VadOptions`: Dataclass for VAD parameters
  - `SpeechTimestampsMap`: Helper for timestamp restoration
  - `SileroVADModel`: ONNX Runtime wrapper for VAD model
- **Key Functions**:
  - `get_speech_timestamps()`: Main VAD function - returns list of speech chunk dicts
  - `collect_chunks()`: Merges VAD chunks into max_duration segments
  - `get_vad_model()`: Cached VAD model loader
- **VAD Model**: 
  - ONNX model: `silero_vad_v6.onnx` (bundled in assets/)
  - Single-threaded CPU execution
  - Processes 512-sample windows
- **Dependencies**: onnxruntime, numpy

#### `assets/__init__.py`
- **Purpose**: Empty file (package marker)
- **Type**: Infrastructure

#### `assets/silero_vad_v6.onnx`
- **Purpose**: Pre-trained Silero VAD model (binary file)
- **Type**: Asset/Data

### Test Files: `tests/`

#### `conftest.py`
- **Purpose**: Pytest fixtures
- **Type**: Test Infrastructure
- **Fixtures**: `data_dir`, `jfk_path`, `physcisworks_path`

#### `test_transcribe.py`
- **Purpose**: Transcription functionality tests
- **Type**: Test
- **Coverage**: Basic transcription, VAD, multilingual, word timestamps, batched inference, clip timestamps

#### `test_tokenizer.py`
- **Purpose**: Tokenizer functionality tests
- **Type**: Test
- **Coverage**: Suppressed tokens, Unicode splitting

#### `test_utils.py`
- **Purpose**: Utility function tests
- **Type**: Test
- **Coverage**: Model listing, model download

### Configuration Files

#### `setup.py`
- **Purpose**: Package installation configuration
- **Type**: Configuration
- **Key Info**: 
  - Reads version from `version.py`
  - Reads requirements from `requirements.txt`
  - Includes conversion extras from `requirements.conversion.txt`
  - Dev extras: black, flake8, isort, pytest

#### `requirements.txt`
- **Purpose**: Runtime dependencies
- **Type**: Configuration
- **Dependencies**: ctranslate2, huggingface_hub, tokenizers, onnxruntime, av (PyAV), tqdm

#### `requirements.conversion.txt`
- **Purpose**: Model conversion dependencies
- **Type**: Configuration
- **Dependencies**: transformers[torch] (for converting models)

#### `setup.cfg`
- **Purpose**: Code style configuration
- **Type**: Configuration
- **Settings**: flake8 (max-line-length=100), isort (profile=black)

#### `MANIFEST.in`
- **Purpose**: Package data inclusion
- **Type**: Configuration
- **Includes**: VAD model, requirements files

### Documentation Files

#### `README.md`
- **Purpose**: User documentation
- **Type**: Documentation
- **Content**: Installation, usage examples, benchmarks, model conversion guide

#### `CONTRIBUTING.md`
- **Purpose**: Contributor guidelines
- **Type**: Documentation
- **Content**: Development setup, testing, code formatting

### CI/CD

#### `.github/workflows/ci.yml`
- **Purpose**: GitHub Actions CI pipeline
- **Type**: Infrastructure
- **Jobs**: Code format check, test execution, package build/publish

### Benchmark Files: `benchmark/`
- **Purpose**: Performance benchmarking scripts
- **Type**: Tooling
- **Files**: speed_benchmark.py, memory_benchmark.py, wer_benchmark.py, evaluate_yt_commons.py

### Docker Files: `docker/`
- **Purpose**: Docker deployment example
- **Type**: Infrastructure/Example
- **Files**: Dockerfile, infer.py (example script)

---

## 3. FUNCTION & CLASS INVENTORY

### `faster_whisper/audio.py`

**Functions:**
- `decode_audio()` - **PUBLIC**: Decodes audio file to numpy array
- `pad_or_trim()` - **PUBLIC**: Pads/trims array to specified length
- `_ignore_invalid_frames()` - **PRIVATE**: Filters invalid audio frames
- `_group_frames()` - **PRIVATE**: Groups frames using AudioFifo
- `_resample_frames()` - **PRIVATE**: Resamples frames

**Classes:** None

---

### `faster_whisper/feature_extractor.py`

**Functions:**
- `FeatureExtractor.get_mel_filters()` - **STATIC**: Generates Mel filter bank
- `FeatureExtractor.stft()` - **STATIC**: Short-Time Fourier Transform

**Classes:**
- `FeatureExtractor` - **PUBLIC**
  - `__init__()`: Initializes extractor
  - `__call__()`: Computes log-Mel spectrogram

---

### `faster_whisper/tokenizer.py`

**Functions:** None

**Classes:**
- `Tokenizer` - **PUBLIC**
  - `__init__()`: Initializes tokenizer with language/task
  - `encode()`: Text → token IDs
  - `decode()`: Token IDs → text (no timestamps)
  - `decode_with_timestamps()`: Token IDs → text (with timestamps)
  - `split_to_word_tokens()`: Splits tokens into words
  - `split_tokens_on_unicode()`: Unicode-based word splitting
  - `split_tokens_on_spaces()`: Space-based word splitting
  - **Properties (cached)**: `transcribe`, `translate`, `sot`, `sot_lm`, `sot_prev`, `eot`, `no_timestamps`, `no_speech`, `timestamp_begin`, `sot_sequence`, `non_speech_tokens`

**Constants:**
- `_TASKS`: ("transcribe", "translate")
- `_LANGUAGE_CODES`: Tuple of 99 language codes

---

### `faster_whisper/transcribe.py`

**Functions:**
- `restore_speech_timestamps()` - **PUBLIC**: Restores original timestamps after VAD
- `get_ctranslate2_storage()` - **PRIVATE**: Converts numpy to CTranslate2 StorageView
- `get_compression_ratio()` - **PRIVATE**: Computes gzip compression ratio (hallucination detection)
- `get_suppressed_tokens()` - **PRIVATE**: Builds token suppression list
- `merge_punctuations()` - **PRIVATE**: Merges punctuation with words

**Classes:**
- `Word` - **PUBLIC** (dataclass): Word-level timestamp data
- `Segment` - **PUBLIC** (dataclass): Transcription segment data
- `TranscriptionOptions` - **PUBLIC** (dataclass): Transcription parameters
- `TranscriptionInfo` - **PUBLIC** (dataclass): Transcription metadata
- `BatchedInferencePipeline` - **PUBLIC**
  - `__init__()`: Initializes pipeline with WhisperModel
  - `transcribe()`: Batch transcription entry point
  - `forward()`: Processes batch of features
  - `generate_segment_batched()`: Batch generation
  - `_batched_segments_generator()`: Generator for batched segments
- `WhisperModel` - **PUBLIC** (Main API)
  - `__init__()`: Model initialization
  - `transcribe()`: Main transcription entry point
  - `generate_segments()`: Iterative segment generation
  - `encode()`: Encodes features to encoder output
  - `generate_with_fallback()`: Generation with temperature fallback
  - `get_prompt()`: Constructs decoder prompt
  - `add_word_timestamps()`: Extracts word-level timestamps
  - `find_alignment()`: Aligns tokens to audio frames
  - `detect_language()`: Language detection
  - `_split_segments_by_timestamps()`: Splits tokens by timestamps
  - **Property**: `supported_languages`

---

### `faster_whisper/utils.py`

**Functions:**
- `available_models()` - **PUBLIC**: Returns list of model names
- `download_model()` - **PUBLIC**: Downloads model from HF Hub
- `format_timestamp()` - **PUBLIC**: Formats seconds to timestamp string
- `get_logger()` - **PUBLIC**: Returns module logger
- `get_assets_path()` - **PUBLIC**: Returns assets directory path
- `get_end()` - **PUBLIC**: Extracts last word end time

**Classes:**
- `disabled_tqdm` - **PRIVATE**: Tqdm wrapper that disables progress bar

**Constants:**
- `_MODELS`: Dictionary mapping model names to HF Hub IDs

---

### `faster_whisper/vad.py`

**Functions:**
- `get_speech_timestamps()` - **PUBLIC**: Main VAD function
- `collect_chunks()` - **PUBLIC**: Merges VAD chunks
- `get_vad_model()` - **PUBLIC** (cached): Loads VAD model

**Classes:**
- `VadOptions` - **PUBLIC** (dataclass): VAD parameters
- `SpeechTimestampsMap` - **PUBLIC**: Timestamp restoration helper
  - `__init__()`: Builds timestamp mapping
  - `get_original_time()`: Converts VAD time to original time
  - `get_chunk_index()`: Finds chunk index for time
- `SileroVADModel` - **PRIVATE**: ONNX Runtime wrapper
  - `__init__()`: Loads ONNX model
  - `__call__()`: Runs VAD inference

---

### Entry Points

**Public API (from `__init__.py`):**
- `WhisperModel` - Main transcription class
- `BatchedInferencePipeline` - Batch processing wrapper
- `decode_audio` - Audio decoding
- `available_models` - Model listing
- `download_model` - Model download
- `format_timestamp` - Timestamp formatting
- `__version__` - Version string

**Internal/Private Functions:**
- All functions prefixed with `_` are private
- Helper functions in `transcribe.py` are private unless exported

---

## 4. WORKFLOW & DATA FLOW

### Startup Sequence

1. **User imports and initializes:**
   ```python
   from faster_whisper import WhisperModel
   model = WhisperModel("large-v3", device="cuda", compute_type="float16")
   ```

2. **WhisperModel.__init__() execution:**
   - Determines model path (downloads from HF Hub if needed via `download_model()`)
   - Loads CTranslate2 Whisper model: `ctranslate2.models.Whisper(model_path, ...)`
   - Loads tokenizer: `tokenizers.Tokenizer.from_file()` or from HF Hub
   - Loads preprocessor config: `preprocessor_config.json` (if exists)
   - Initializes `FeatureExtractor` with config or defaults
   - Sets up timing constants: `frames_per_second`, `tokens_per_second`, `time_precision`
   - Sets `max_length = 448` (Whisper's max sequence length)

3. **Model ready for transcription**

### Model Initialization Details

**CTranslate2 Model Loading:**
- Device selection: "cpu", "cuda", or "auto"
- Compute type: "default", "float16", "int8", "int8_float16"
- Multi-GPU support: `device_index` can be list `[0, 1, 2, 3]`
- Thread configuration: `cpu_threads` (intra), `num_workers` (inter)

**Tokenizer Loading:**
- Primary: `tokenizer.json` from model directory
- Fallback: Downloads from `openai/whisper-tiny` (or `.en` variant)

**Feature Extractor:**
- Defaults: 80 mel bins, 16kHz, 160 hop, 30s chunks
- Override: `preprocessor_config.json` in model directory

### API Request Flow (Standard Transcription)

1. **User calls `model.transcribe(audio_path, ...)`**

2. **Audio Decoding:**
   - If string path: `decode_audio(audio_path)` → float32 numpy array (16kHz mono)
   - If numpy array: Used directly
   - Duration calculated: `duration = audio.shape[0] / sampling_rate`

3. **VAD Filtering (if enabled):**
   - `get_speech_timestamps(audio, vad_parameters)` → List of `{"start": int, "end": int}`
   - `collect_chunks(audio, speech_chunks)` → Audio chunks + metadata
   - Audio concatenated: `audio = np.concatenate(audio_chunks)`
   - `duration_after_vad` calculated

4. **Feature Extraction:**
   - `feature_extractor(audio, chunk_length=chunk_length)` → Mel spectrogram
   - Shape: `(n_mels, n_frames)` where `n_frames = audio_length / hop_length`

5. **Language Detection (if not provided):**
   - `detect_language(features=features[..., seek:])`
   - Encodes first segment: `encoder_output = encode(pad_or_trim(features[..., :nb_max_frames]))`
   - `model.detect_language(encoder_output)` → List of `(language_token, probability)`
   - Selects language with probability > threshold

6. **Tokenizer Initialization:**
   - `Tokenizer(hf_tokenizer, is_multilingual, task, language)`
   - Sets up language and task tokens

7. **Segment Generation Loop (`generate_segments()`):**
   - **For each 30-second window (or clip boundary):**
     a. Extract segment: `segment = features[:, seek : seek + segment_size]`
     b. Pad/trim: `segment = pad_or_trim(segment)` → `(n_mels, 3000)`
     c. Encode: `encoder_output = encode(segment)`
     d. Build prompt: `prompt = get_prompt(tokenizer, previous_tokens, ...)`
     e. Generate: `result = generate_with_fallback(encoder_output, prompt, ...)`
     f. Check quality: `no_speech_prob`, `avg_logprob`, `compression_ratio`
     g. Split by timestamps: `_split_segments_by_timestamps(...)` → List of subsegments
     h. Word timestamps (if enabled): `add_word_timestamps(...)`
     i. Yield segments: `yield Segment(...)`
     j. Update seek position for next window

8. **Timestamp Restoration (if VAD used):**
   - `restore_speech_timestamps(segments, speech_chunks, sampling_rate)`
   - Maps VAD-filtered timestamps back to original audio timestamps

9. **Return:**
   - Generator of `Segment` objects
   - `TranscriptionInfo` object with metadata

### Streaming Audio Lifecycle

**Note:** This codebase does NOT implement streaming audio input. It processes complete audio files.

**However, the output is streaming:**
- `transcribe()` returns a generator
- Segments are yielded as they are produced
- Memory-efficient for long audio files

### Segment Finalization Logic

1. **Token Generation:**
   - `model.generate()` returns `WhisperGenerationResult`
   - Contains: `sequences_ids`, `scores`, `no_speech_prob`

2. **Quality Checks:**
   - `compression_ratio = len(text_bytes) / len(zlib.compress(text_bytes))`
   - If `compression_ratio > threshold` → Too repetitive (hallucination)
   - If `avg_logprob < threshold` → Low confidence
   - If `no_speech_prob > threshold AND avg_logprob < threshold` → Silence
   - Temperature fallback: Retry with higher temperature if quality check fails

3. **Timestamp Splitting:**
   - `_split_segments_by_timestamps()`:
     - Finds consecutive timestamp tokens
     - Splits tokens at timestamp boundaries
     - Creates subsegments with `start`, `end`, `tokens`
     - Updates `seek` position for next window

4. **Word Timestamp Extraction (if enabled):**
   - `find_alignment()`: Uses `model.align()` to align tokens to audio frames
   - Extracts word boundaries from alignment
   - Applies median filtering and duration constraints
   - Merges punctuation with adjacent words
   - Handles segment boundary edge cases

5. **Segment Creation:**
   - `Segment(id, seek, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob, words, temperature)`

### Error Handling Paths

1. **Model Download Errors:**
   - `download_model()` raises `ValueError` for invalid model size
   - HuggingFace Hub errors propagate (network, auth, etc.)

2. **Audio Decoding Errors:**
   - `av.error.InvalidDataError`: Caught and skipped in `_ignore_invalid_frames()`
   - File not found: Propagates as FileNotFoundError

3. **Model Inference Errors:**
   - CTranslate2 errors propagate (device errors, memory errors, etc.)
   - No explicit error handling in `generate_with_fallback()` - relies on temperature fallback

4. **Quality Check Failures:**
   - Handled by temperature fallback mechanism
   - If all temperatures fail: Returns best result from `below_cr_threshold_results` or `all_results`

5. **VAD Errors:**
   - `onnxruntime` ImportError: Raises RuntimeError with helpful message
   - ONNX model file missing: FileNotFoundError

6. **Tokenization Errors:**
   - Invalid language code: `ValueError` in `Tokenizer.__init__()`
   - Invalid task: `ValueError` in `Tokenizer.__init__()`

### Shutdown / Cleanup Behavior

1. **No explicit cleanup required:**
   - Python GC handles memory
   - CTranslate2 models are C++ objects, cleaned up when Python object deleted

2. **Explicit GC in audio decoding:**
   - `gc.collect()` called after PyAV resampler deletion (workaround for issue #390)

3. **VAD Model Caching:**
   - `@functools.lru_cache` on `get_vad_model()`
   - Model persists for lifetime of Python process

4. **Generator Cleanup:**
   - Generators cleaned up when exhausted or garbage collected
   - Progress bars (`tqdm`) closed explicitly in `generate_segments()`

---

## 5. EXISTING CAPABILITIES VS GAPS

### Fully Implemented and Reusable

✅ **Audio Processing:**
- Audio file decoding (all formats supported by FFmpeg)
- Stereo channel splitting
- Automatic resampling to 16kHz mono
- Robust error handling for corrupted frames

✅ **Feature Extraction:**
- Mel spectrogram computation
- Configurable via model's `preprocessor_config.json`
- STFT implementation (pure NumPy, no librosa dependency)

✅ **Model Inference:**
- CTranslate2 integration (optimized Whisper)
- CPU and GPU support
- Multi-GPU support
- Quantization support (int8, float16)
- Batch processing via `BatchedInferencePipeline`

✅ **Tokenization:**
- Full Whisper tokenizer support
- Multilingual support (99 languages)
- Special token management
- Word splitting for space-based and Unicode-based languages
- Non-speech token suppression

✅ **Transcription:**
- Standard transcription workflow
- Language detection
- Multilingual per-segment detection
- Word-level timestamps
- Segment splitting by timestamps
- Quality checks (compression ratio, log probability, no-speech)
- Temperature fallback mechanism

✅ **VAD Integration:**
- Silero VAD v6 integration
- Configurable VAD parameters
- Timestamp restoration after VAD filtering
- Chunk collection with max duration

✅ **Utilities:**
- Model download from Hugging Face Hub
- Timestamp formatting
- Model listing

### Partially Implemented

⚠️ **Streaming Input:**
- **Status**: Not implemented
- **Current**: Processes complete audio files
- **Gap**: No real-time audio streaming API
- **Workaround**: External code can feed audio chunks sequentially

⚠️ **Batch Processing:**
- **Status**: Implemented but with limitations
- **Current**: `BatchedInferencePipeline` processes pre-chunked audio
- **Gap**: Requires VAD or manual chunking; no automatic batching of multiple files
- **Note**: VAD is enabled by default in batched mode

⚠️ **Error Recovery:**
- **Status**: Basic fallback (temperature)
- **Current**: Temperature fallback for quality issues
- **Gap**: No retry mechanism for model errors, no graceful degradation

⚠️ **Progress Reporting:**
- **Status**: Basic (tqdm)
- **Current**: Progress bar for segment processing
- **Gap**: No callback mechanism, no detailed progress (encoder/decoder steps)

### Missing Entirely

❌ **REST API:**
- No HTTP server
- No REST endpoints
- No request/response handling

❌ **WebSocket API:**
- No WebSocket server
- No real-time streaming protocol
- No bidirectional communication

❌ **Authentication/Authorization:**
- No user management
- No API keys
- No rate limiting

❌ **Concurrent Request Handling:**
- No request queue
- No load balancing
- No request prioritization

❌ **Model Management:**
- No dynamic model loading/unloading
- No model versioning API
- No A/B testing support

❌ **Monitoring/Logging:**
- Basic Python logging only
- No metrics collection
- No performance monitoring
- No request tracing

❌ **Configuration Management:**
- No config file support (beyond model's preprocessor_config.json)
- No environment variable configuration
- No runtime configuration changes

❌ **Caching:**
- No transcription result caching
- No encoder output caching (except in multi-GPU scenario)

❌ **Output Formats:**
- Only Python objects (Segment, Word)
- No SRT, VTT, JSON export
- No custom format plugins

### Intentional vs Accidental

**Intentional Design Choices:**
- ✅ Generator-based output (memory efficiency)
- ✅ Single-process model (simplicity)
- ✅ CTranslate2 dependency (performance)
- ✅ PyAV for audio (no FFmpeg system dependency)
- ✅ Pure NumPy STFT (no librosa dependency)
- ✅ Optional VAD (opt-in feature)
- ✅ Temperature fallback (quality assurance)

**Likely Accidental Gaps:**
- ⚠️ No streaming input API (may be intentional for simplicity)
- ⚠️ Limited error recovery (may be intentional - let caller handle)
- ⚠️ No export formats (may be intentional - keep API simple)
- ⚠️ Explicit GC in audio decoding (workaround for PyAV issue #390)

---

## 6. "DO NOT REINVENT" GUIDANCE

### Components to REUSE As-Is

✅ **Audio Decoding (`audio.py`):**
- `decode_audio()` is robust and handles all edge cases
- PyAV integration is well-tested
- Stereo splitting works correctly
- **DO NOT** replace with another audio library

✅ **Feature Extraction (`feature_extractor.py`):**
- Mel spectrogram computation matches Whisper's implementation
- STFT is correct and efficient
- Configurable via model config
- **DO NOT** replace with librosa or other libraries

✅ **Tokenizer (`tokenizer.py`):**
- Full Whisper tokenizer compatibility
- Handles all special tokens correctly
- Word splitting logic is language-aware
- **DO NOT** replace with custom tokenizer

✅ **VAD Integration (`vad.py`):**
- Silero VAD integration is correct
- Timestamp restoration logic is sound
- Chunk collection handles edge cases
- **DO NOT** replace VAD model or timestamp restoration

✅ **Model Inference (CTranslate2):**
- CTranslate2 is the core optimization
- Multi-GPU support works correctly
- Quantization is handled by CTranslate2
- **DO NOT** replace with PyTorch or other frameworks

✅ **Quality Checks:**
- Compression ratio detection is effective
- Log probability thresholds are reasonable
- Temperature fallback mechanism works
- **DO NOT** remove or significantly alter

### Components to EXTEND (Not Replace)

🔧 **WhisperModel Class:**
- Core transcription logic is solid
- **EXTEND** with new methods for additional features
- **DO NOT** refactor the main `transcribe()` workflow

🔧 **Segment Generation:**
- `generate_segments()` loop is correct
- **EXTEND** with additional post-processing hooks
- **DO NOT** change the core iteration logic

🔧 **Word Timestamp Extraction:**
- `find_alignment()` and `add_word_timestamps()` are correct
- **EXTEND** with additional alignment methods if needed
- **DO NOT** replace the core alignment algorithm

🔧 **BatchedInferencePipeline:**
- Batch processing logic is correct
- **EXTEND** with additional batching strategies
- **DO NOT** change the core batch forward pass

### Components to NOT Duplicate

❌ **Model Loading:**
- `download_model()` handles HF Hub correctly
- **DO NOT** create alternative model loading

❌ **Timestamp Formatting:**
- `format_timestamp()` is simple and correct
- **DO NOT** duplicate this logic

❌ **Logger Setup:**
- `get_logger()` returns standard Python logger
- **DO NOT** create custom logging infrastructure

❌ **VAD Model Loading:**
- `get_vad_model()` with LRU cache is efficient
- **DO NOT** create alternative VAD loading

### Safe Extension Points

✅ **New Transcription Options:**
- Add parameters to `TranscriptionOptions` dataclass
- Pass through `transcribe()` method
- Use in `generate_segments()` or `generate_with_fallback()`

✅ **New Post-Processing:**
- Add functions after segment generation
- Chain with `restore_speech_timestamps()` if needed
- Yield modified segments

✅ **New Output Formats:**
- Create export functions that consume `Segment` generators
- Do not modify core transcription output

✅ **New Quality Checks:**
- Add checks in `generate_with_fallback()`
- Extend `TranscriptionOptions` with new thresholds

---

## 7. OPEN QUESTIONS / ASSUMPTIONS

### Ambiguities in Codebase

❓ **Multi-GPU Encoder Output Handling:**
- Line 1394: `to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1`
- **Question**: Why move to CPU only for multi-GPU? Is this for load balancing?
- **Assumption**: Prevents GPU memory issues when distributing work across GPUs

❓ **Prompt Reset Logic:**
- Lines 1372-1383: Prompt reset based on temperature
- **Question**: Why reset at `prompt_reset_on_temperature` threshold?
- **Assumption**: Prevents error propagation when model is struggling (high temperature = low quality)

❓ **Hallucination Detection:**
- Lines 1294-1339: Complex hallucination silence skipping logic
- **Question**: Why such complex heuristics? Is this based on empirical testing?
- **Assumption**: Detects and skips model hallucinations (repetitive or anomalous text)

❓ **Word Duration Constraints:**
- Lines 1602-1616: Truncates long words at sentence boundaries
- **Question**: Why 0.7s median and 2x max? Are these tuned values?
- **Assumption**: Prevents unrealistic word durations from alignment errors

❓ **VAD Default Parameters:**
- `VadOptions` defaults: `min_silence_duration_ms=2000`, `speech_pad_ms=400`
- **Question**: Are these optimal for all use cases?
- **Assumption**: Conservative defaults that work well for most audio

❓ **Max Length Constant:**
- Line 722: `self.max_length = 448`
- **Question**: Is this always 448 for all Whisper models?
- **Assumption**: This is Whisper's fixed max sequence length

### Assumptions Made by Original Author

✅ **Audio Quality:**
- Assumes 16kHz mono audio (standard for Whisper)
- Assumes reasonable audio quality (not heavily corrupted)

✅ **Model Compatibility:**
- Assumes CTranslate2 models match OpenAI Whisper architecture
- Assumes tokenizer compatibility with OpenAI's tokenizer

✅ **Memory Management:**
- Assumes sufficient GPU/RAM for model size
- Assumes Python GC will clean up appropriately (except explicit GC in audio.py)

✅ **Thread Safety:**
- Assumes CTranslate2 handles thread safety for `num_workers > 1`
- Assumes no concurrent access to same model instance (not documented)

✅ **Error Handling:**
- Assumes caller will handle file I/O errors
- Assumes caller will handle model loading errors
- Quality issues handled internally via fallback

✅ **Performance:**
- Assumes batch processing is beneficial (BatchedInferencePipeline)
- Assumes VAD reduces processing time (filters silence)

### Areas Needing Clarification

🔍 **Concurrent Model Access:**
- **Question**: Is `WhisperModel` thread-safe?
- **Current**: `num_workers > 1` suggests multi-threading is supported
- **Gap**: No documentation on thread safety guarantees

🔍 **Model State:**
- **Question**: Can model be used concurrently from multiple threads?
- **Current**: Code suggests yes (num_workers parameter)
- **Gap**: No explicit locking or state management visible

🔍 **Memory Footprint:**
- **Question**: What is memory usage for different model sizes?
- **Current**: No memory profiling in codebase
- **Gap**: Users must determine empirically

🔍 **Batch Size Selection:**
- **Question**: How to choose optimal `batch_size` for BatchedInferencePipeline?
- **Current**: Default is 8, but no guidance on tuning
- **Gap**: No automatic batch size optimization

🔍 **VAD Performance:**
- **Question**: What is VAD overhead?
- **Current**: VAD runs on CPU, single-threaded
- **Gap**: No performance metrics

🔍 **Quantization Trade-offs:**
- **Question**: When to use int8 vs float16?
- **Current**: Both supported, but no guidance
- **Gap**: No accuracy/speed trade-off documentation

🔍 **Multi-GPU Load Balancing:**
- **Question**: How does CTranslate2 distribute work across GPUs?
- **Current**: `device_index=[0,1,2,3]` enables multi-GPU
- **Gap**: No documentation on load balancing strategy

### Code Comments and Documentation Gaps

📝 **Complex Algorithms:**
- Hallucination detection (lines 1294-1339): Needs detailed comments
- Word timestamp alignment (lines 1567-1696): Complex logic, sparse comments
- VAD chunk collection (lines 220-277): Logic is clear but could use examples

📝 **Magic Numbers:**
- `0.7` median duration (line 1602): Why this value?
- `2.0` max word duration (line 1250): Tuned value?
- `0.133` min word duration (line 1248): Why this threshold?
- `3.0` anomaly score threshold (line 1260): Empirical?

📝 **Configuration:**
- No documentation on `preprocessor_config.json` format
- No examples of custom VAD parameters
- No guidance on temperature selection

---

## SUMMARY

This codebase is a **well-architected, production-ready** implementation of Whisper transcription with clear separation of concerns. The core transcription logic is solid and should be extended rather than replaced. The main gaps are in **API layer** (REST/WebSocket), **monitoring**, and **configuration management** - areas that are typically handled by wrapper services rather than the core library.

The codebase follows Python best practices, uses appropriate abstractions (dataclasses, generators), and handles edge cases well. The integration with CTranslate2 is clean and the VAD integration is robust.

**Key Strengths:**
- Clean architecture with clear module boundaries
- Efficient memory usage (generators)
- Robust error handling for audio decoding
- Quality assurance mechanisms (fallback, checks)
- Extensible design (dataclasses, options)

**Key Gaps for API Service:**
- No HTTP/WebSocket server
- No authentication/authorization
- No request queuing/load balancing
- No monitoring/metrics
- No output format export

**Recommendation:** Build API service as a **wrapper** around this library, reusing all core components as-is.
