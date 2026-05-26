# local/whisper-cpp

## Uni-Xervo support

- Provider ID: `local/whisper-cpp`
- Feature flag: `provider-whisper-cpp` (opt-in; NOT in default features)
- Capabilities: `transcribe`

A whisper.cpp-backed transcription provider via the `whisper-rs` binding.
Loads a ggml `.bin` weights file from a HuggingFace repo, decodes the
input audio to 16 kHz mono f32 PCM, and runs
`whisper_rs::WhisperContext::full(...)` on a blocking thread via
`tokio::task::spawn_blocking` (whisper-rs is synchronous).

Introduced in 0.13.0.

## Build requirements

`whisper-rs` compiles whisper.cpp's C/C++ source via CMake. The build
host needs:

- `cmake` (any recent version; 3.10+ verified)
- A C/C++ compiler (`cc` / `clang` / MSVC)

These are present on standard Linux dev boxes, macOS with Xcode CLT,
and Windows with Visual Studio Build Tools. CI for `provider-whisper-cpp`
should install CMake explicitly if the runner image doesn't ship it.

This is also why the feature is **opt-in**: the default `cargo build`
keeps its toolchain requirement at "just Rust." Enable explicitly with
`--features provider-whisper-cpp`.

## Uni-Xervo provider options

- `model_path` (string, default `"ggml-base.bin"`) — file within the HF
  repo. Common choices: `ggml-tiny.en.bin` (~75 MB), `ggml-base.bin`
  (~150 MB), `ggml-small.bin` (~480 MB), `ggml-medium.bin` (~1.5 GB),
  `ggml-large-v3.bin` (~3 GB).
- `default_language` (string, optional) — ISO 639-1 code applied when
  `TranscribeOptions::language` is `None`. If both are `None`,
  whisper.cpp auto-detects.
- `cache_dir` (string, optional) — overrides `UNI_CACHE_DIR` and the
  default `.uni_cache/whisper-cpp/...` location.

## Audio decode (v1)

| Input | Behavior |
| --- | --- |
| `AudioInput::Pcm { sample_rate: 16000, channels: 1, samples }` | passed through |
| `AudioInput::Pcm { sample_rate: 16000, channels: 2, samples }` | collapsed to mono via simple averaging |
| `AudioInput::Pcm` at other sample rates | rejected — resample upstream |
| `AudioInput::Bytes { media_type: "audio/wav", ... }` | 16-bit PCM WAV mono/stereo decoded inline (no `symphonia` dep) |
| `AudioInput::Bytes` for other formats (MP3, FLAC, Opus, …) | rejected — decode upstream to `AudioInput::Pcm` |

Resampling and container-decode support beyond WAV are deferred to
follow-ups.

## Output behavior

- `TranscribeResult::language` reports the configured/default language,
  or `"auto"` when nothing was set. (whisper-rs 0.13 doesn't expose the
  detected language id on `WhisperState`; full auto-detect still happens
  inside whisper.cpp at runtime.)
- `TranscribeResult::segments` carries `start_ms` / `end_ms` /
  `text` (whitespace-trimmed) per segment.
- `TranscribeSegment::words` populates when
  `TranscribeOptions::word_timestamps = true` (`full_get_token_data`
  provides per-token `t0` / `t1` / `p`).
- `TranscribeSegment::speaker` is always `None` — whisper.cpp does not
  natively diarize. A pyannote-style segmenter is a separate concern.
- Sampling strategy is greedy (`SamplingStrategy::Greedy { best_of: 1 }`).
  Beam search / temperature sampling are mechanical follow-ups when a
  consumer needs them.

## Runtime contract

```rust
let model = runtime.transcriber("asr/whisper").await?;
let result = model.transcribe(audio, TranscribeOptions::default()).await?;
```

`transcribe_many(Vec<AudioInput>, TranscribeOptions)` defaults to the
trait-level concurrent fan-out (one `transcribe` call per input via
`futures::try_join_all`). This is correct for whisper.cpp since it
processes one stream at a time anyway.

## Example catalog entries

### Tiny English-only model (development / fast tests)

```json
{
  "alias": "asr/whisper-tiny",
  "task": "transcribe",
  "provider_id": "local/whisper-cpp",
  "model_id": "ggerganov/whisper.cpp",
  "options": {
    "model_path": "ggml-tiny.en.bin",
    "default_language": "en"
  }
}
```

### Base multilingual

```json
{
  "alias": "asr/whisper",
  "task": "transcribe",
  "provider_id": "local/whisper-cpp",
  "model_id": "ggerganov/whisper.cpp",
  "options": {
    "model_path": "ggml-base.bin"
  }
}
```

## External references

- whisper.cpp: <https://github.com/ggerganov/whisper.cpp>
- whisper-rs (binding): <https://github.com/tazz4843/whisper-rs>
- Pre-built ggml weights: <https://huggingface.co/ggerganov/whisper.cpp>
