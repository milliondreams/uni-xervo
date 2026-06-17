//! Multimodal embedding types and traits (image / audio / mixed-modality).
//!
//! These types accompany the trait definitions for embedders that consume
//! non-text inputs. Each trait sits alongside
//! [`EmbeddingModel`](crate::traits::EmbeddingModel) and returns an
//! [`EmbedResult`] so remote providers can attach per-call usage counts.

use crate::error::Result;
use crate::traits::{EmbedResult, ImageInput};
use async_trait::async_trait;
use std::any::Any;

/// A model that produces dense vector embeddings from images.
///
/// One vector is returned per input image, each of dimension
/// [`dimensions`](ImageEmbeddingModel::dimensions). Implementations
/// typically batch internally for GPU efficiency; callers should pass as
/// large a batch as fits their latency budget.
#[async_trait]
pub trait ImageEmbeddingModel: Send + Sync + Any {
    /// Embed a batch of images.
    ///
    /// # Errors
    /// Returns an error if the provider fails to load any input, runs out
    /// of memory, or the upstream API rejects the request.
    async fn embed(&self, images: Vec<ImageInput>) -> Result<EmbedResult>;

    /// The dimensionality of vectors produced by this model.
    fn dimensions(&self) -> u32;

    /// The underlying model identifier (HuggingFace repo ID or remote API model name).
    fn model_id(&self) -> &str;

    /// Optional warmup hook (e.g. load weights into memory on first access).
    /// The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// A model that produces dense vector embeddings from audio.
///
/// One vector is returned per input audio, each of dimension
/// [`dimensions`](AudioEmbeddingModel::dimensions). Use cases include
/// CLAP-style audio similarity, music tagging, and audio-text retrieval.
#[async_trait]
pub trait AudioEmbeddingModel: Send + Sync + Any {
    /// Embed a batch of audio inputs.
    ///
    /// # Errors
    /// Returns an error if the provider cannot decode an input or runs out
    /// of memory.
    async fn embed(&self, audios: Vec<AudioInput>) -> Result<EmbedResult>;

    /// The dimensionality of vectors produced by this model.
    fn dimensions(&self) -> u32;

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

/// A model that produces a single dense vector from heterogeneous content
/// (text + image + audio together).
///
/// Supports models like Cohere Embed v4, Jina Embeddings v4, and Gemini
/// Embedding 2 that map mixed-modality inputs into a single shared embedding
/// space.
#[async_trait]
pub trait MultimodalEmbeddingModel: Send + Sync + Any {
    /// Embed a batch of mixed-modality inputs.
    ///
    /// Each [`MultimodalInput`] produces one vector spanning all its blocks.
    ///
    /// # Errors
    /// Returns an error if the provider rejects an unsupported modality
    /// combination or runs out of memory.
    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<EmbedResult>;

    /// The dimensionality of vectors produced by this model.
    fn dimensions(&self) -> u32;

    /// The underlying model identifier.
    fn model_id(&self) -> &str;

    /// Modalities accepted by this model (e.g. `&[Text, Image]` for Cohere
    /// Embed v4 vs `&[Text, Image, Audio, Video]` for Gemini Embedding 2).
    fn supported_modalities(&self) -> &[Modality];

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

    // Dyn-safety smoke checks. These will fail to compile if any of these
    // traits accidentally becomes non-object-safe.
    #[test]
    fn image_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn ImageEmbeddingModel>) {}
    }

    #[test]
    fn audio_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn AudioEmbeddingModel>) {}
    }

    #[test]
    fn multimodal_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn MultimodalEmbeddingModel>) {}
    }
}

/// An audio input to a transcription or audio embedding model.
///
/// Two variants reflect the realistic input shapes:
/// raw container bytes for HTTP handlers, and pre-decoded PCM for in-process
/// audio capture. Local-file ingest is the caller's responsibility — uni-xervo
/// providers do not perform file I/O, matching the [`ImageInput`] design.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Raw container bytes (e.g. WAV, MP3, FLAC). The provider decides how to
    /// decode based on `media_type`.
    Bytes {
        /// Audio payload bytes in container format.
        data: Vec<u8>,
        /// MIME type indicating the container format (e.g. `"audio/wav"`).
        media_type: String,
    },
    /// Pre-decoded PCM samples. The provider skips the decode step.
    Pcm {
        /// Samples per second (Hz).
        sample_rate: u32,
        /// Number of interleaved channels (1 = mono, 2 = stereo, …).
        channels: u16,
        /// Sample data, length `sample_rate * channels * duration_seconds`.
        samples: Vec<f32>,
    },
}

/// One block of input to a [`MultimodalEmbeddingModel`].
///
/// Distinct from [`ContentBlock`](crate::traits::ContentBlock) — that type
/// is for conversational generation contexts and only carries text + image.
/// This type extends to audio for use cases like Cohere Embed v4 and Gemini
/// Embedding 2 that produce a single embedding spanning multiple modalities.
#[derive(Debug, Clone)]
pub enum MultimodalBlock {
    /// Plain text content.
    Text(String),
    /// An image input (URL or raw bytes — see [`ImageInput`]).
    Image(ImageInput),
    /// An audio input.
    Audio(AudioInput),
}

/// A heterogeneous input to a [`MultimodalEmbeddingModel`].
///
/// Each input produces one embedding spanning all of its blocks. Multiple
/// inputs are batched together by passing a `Vec<MultimodalInput>`.
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    /// Ordered sequence of content blocks that share a single embedding.
    pub blocks: Vec<MultimodalBlock>,
}

/// Modality labels reported by [`MultimodalEmbeddingModel::supported_modalities`].
///
/// The exact set varies per model: Cohere Embed v4 supports `Text + Image`,
/// Gemini Embedding 2 supports `Text + Image + Audio + Video`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Plain text.
    Text,
    /// Still images.
    Image,
    /// Audio (speech, music, ambient).
    Audio,
    /// Video (frame sequences with optional audio).
    Video,
}
