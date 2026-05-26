//! Automatic speech recognition types and trait.

use crate::error::Result;
use crate::traits::multimodal::AudioInput;
use async_trait::async_trait;
use futures::future::try_join_all;
use std::any::Any;

/// Options for a [`TranscriptionModel::transcribe`] call.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// ISO 639-1 language code (e.g. `"en"`). `None` requests auto-detection.
    pub language: Option<String>,
    /// Whether to populate per-word timestamps on each segment.
    pub word_timestamps: bool,
    /// Whether to populate speaker labels via diarization.
    pub diarize: bool,
    /// Optional initial prompt for biasing decoding (e.g. domain-specific
    /// terminology, named-entity priming). Supported by whisper.cpp et al.
    pub initial_prompt: Option<String>,
}

/// Result of a transcription call.
#[derive(Debug, Clone)]
pub struct TranscribeResult {
    /// Detected (or supplied) ISO 639-1 language code.
    pub language: String,
    /// Ordered speech segments.
    pub segments: Vec<TranscribeSegment>,
}

/// One transcribed segment.
#[derive(Debug, Clone)]
pub struct TranscribeSegment {
    /// Start timestamp in milliseconds from the audio start.
    pub start_ms: u64,
    /// End timestamp in milliseconds from the audio start (exclusive).
    pub end_ms: u64,
    /// Recognized text.
    pub text: String,
    /// Speaker label, populated when [`TranscribeOptions::diarize`] is set
    /// and supported by the provider.
    pub speaker: Option<String>,
    /// Per-word timestamps, populated when
    /// [`TranscribeOptions::word_timestamps`] is set and supported.
    pub words: Vec<TranscribeWord>,
}

/// One word-level timestamp entry.
#[derive(Debug, Clone)]
pub struct TranscribeWord {
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// End timestamp in milliseconds (exclusive).
    pub end_ms: u64,
    /// Word surface form.
    pub text: String,
    /// Provider-reported confidence in `[0.0, 1.0]`, if available.
    pub confidence: Option<f32>,
}

/// A model that transcribes speech audio into text with timing information.
///
/// Targets whisper.cpp / whisper-rs, OpenAI Whisper API, AssemblyAI, and
/// similar one-stream-at-a-time ASR engines. Batch transcription is supported
/// via [`TranscriptionModel::transcribe_many`]; the default implementation
/// fans out concurrently to [`TranscriptionModel::transcribe`].
#[async_trait]
pub trait TranscriptionModel: Send + Sync + Any {
    /// Transcribe a single audio stream.
    ///
    /// This is the canonical primitive every implementation must provide.
    ///
    /// # Errors
    /// Returns an error if the provider cannot decode the input or fails
    /// internally.
    async fn transcribe(
        &self,
        audio: AudioInput,
        options: TranscribeOptions,
    ) -> Result<TranscribeResult>;

    /// Transcribe a batch of audio inputs.
    ///
    /// The default implementation fans out concurrently via
    /// [`try_join_all`](futures::future::try_join_all), which is what
    /// whisper.cpp and remote single-stream APIs want. Providers that can
    /// genuinely batch internally (GPU-batched ASR, server pools) should
    /// override this to take advantage of that.
    ///
    /// Results are returned in the same order as `audios`. The shared
    /// `options` apply to every input.
    ///
    /// # Errors
    /// Returns the first error encountered; in-flight transcriptions for
    /// other inputs are not awaited (per `try_join_all` semantics).
    async fn transcribe_many(
        &self,
        audios: Vec<AudioInput>,
        options: TranscribeOptions,
    ) -> Result<Vec<TranscribeResult>> {
        try_join_all(
            audios
                .into_iter()
                .map(|audio| self.transcribe(audio, options.clone())),
        )
        .await
    }

    /// ISO 639-1 language codes the model can handle. May be empty for
    /// providers that report language only at runtime via auto-detection.
    fn supported_languages(&self) -> &[String];

    /// The underlying model identifier.
    fn model_id(&self) -> &str;

    /// Optional warmup hook. The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn transcription_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn TranscriptionModel>) {}
    }
}
