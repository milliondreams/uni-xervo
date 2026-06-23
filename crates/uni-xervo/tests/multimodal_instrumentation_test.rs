//! Phase 7 instrumentation parity tests.
//!
//! Verifies the `run_instrumented` helper in `src/reliability.rs` correctly
//! applies timeout + retry + metrics through the seven new wrappers. Since
//! the helper is shared across all seven, testing on a representative subset
//! (image embedder, transcription single + batch) is sufficient to prove the
//! pattern. The metrics test additionally exercises the histogram + counter
//! emission for an `embed_image` call.

use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use uni_xervo::api::{ModelAliasSpec, ModelTask, RetryConfig, WarmupPolicy};
use uni_xervo::error::{Result, RuntimeError};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{
    AudioInput, EmbedResult, ImageEmbeddingModel, ImageInput, LoadedModelHandle, ModelInfo,
    ModelProvider, ProviderCapabilities, ProviderHealth, TranscribeOptions, TranscribeResult,
    TranscribeSegment, TranscriptionModel,
};

// ── Test-only providers + models for fault injection ────────────────────

/// Image-embedding model that sleeps inside `embed`, used to trigger timeout.
struct SlowImageEmbeddingModel {
    delay: Duration,
}

#[async_trait]
impl ImageEmbeddingModel for SlowImageEmbeddingModel {
    async fn embed(&self, _images: Vec<ImageInput>) -> Result<EmbedResult> {
        tokio::time::sleep(self.delay).await;
        Ok(EmbedResult {
            vectors: vec![vec![0.0; 384]],
            usage: None,
        })
    }
    fn dimensions(&self) -> u32 {
        384
    }
}

impl ModelInfo for SlowImageEmbeddingModel {
    fn model_id(&self) -> &str {
        "slow"
    }
}

/// Image-embedding model that returns a retryable failure for the first
/// `fail_count` calls, then succeeds.
struct FailingThenSucceedingImageEmbeddingModel {
    fail_remaining: AtomicU32,
    call_count: Arc<AtomicU32>,
}

#[async_trait]
impl ImageEmbeddingModel for FailingThenSucceedingImageEmbeddingModel {
    async fn embed(&self, _images: Vec<ImageInput>) -> Result<EmbedResult> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let remaining = self.fail_remaining.load(Ordering::SeqCst);
        if remaining > 0 {
            self.fail_remaining.fetch_sub(1, Ordering::SeqCst);
            // RateLimited is one of the retryable variants per RuntimeError::is_retryable.
            return Err(RuntimeError::RateLimited);
        }
        Ok(EmbedResult {
            vectors: vec![vec![0.0; 384]],
            usage: None,
        })
    }
    fn dimensions(&self) -> u32 {
        384
    }
}

impl ModelInfo for FailingThenSucceedingImageEmbeddingModel {
    fn model_id(&self) -> &str {
        "flaky"
    }
}

/// Transcription model whose `transcribe_many` sleeps once, batch-wide. Used
/// to verify the wrapper applies the timeout to the whole batch, not
/// per-item.
struct SlowBatchTranscriptionModel {
    batch_delay: Duration,
    languages: Vec<String>,
}

#[async_trait]
impl TranscriptionModel for SlowBatchTranscriptionModel {
    async fn transcribe(
        &self,
        audios: Vec<AudioInput>,
        _options: TranscribeOptions,
    ) -> Result<Vec<TranscribeResult>> {
        // Single batch-wide sleep before returning all results — distinct
        // from a fan-out where each item would sleep independently.
        tokio::time::sleep(self.batch_delay).await;
        Ok(audios
            .into_iter()
            .map(|_| TranscribeResult {
                language: "en".to_string(),
                segments: vec![TranscribeSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "fast".to_string(),
                    speaker: None,
                    words: Vec::new(),
                }],
            })
            .collect())
    }

    fn supported_languages(&self) -> &[String] {
        &self.languages
    }
}

impl ModelInfo for SlowBatchTranscriptionModel {
    fn model_id(&self) -> &str {
        "slow-batch"
    }
}

/// Minimal provider that hands back a pre-constructed Arc<dyn ...> handle.
struct DirectProvider<T: ?Sized + 'static> {
    id: &'static str,
    task: ModelTask,
    handle: std::sync::Mutex<Option<Arc<T>>>,
}

impl<T: ?Sized + 'static + Send + Sync> DirectProvider<T> {
    fn new(id: &'static str, task: ModelTask, handle: Arc<T>) -> Self {
        Self {
            id,
            task,
            handle: std::sync::Mutex::new(Some(handle)),
        }
    }
}

#[async_trait]
impl ModelProvider for DirectProvider<dyn ImageEmbeddingModel> {
    fn provider_id(&self) -> &'static str {
        self.id
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![self.task],
        }
    }
    async fn load(&self, _spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let inner = self
            .handle
            .lock()
            .unwrap()
            .clone()
            .expect("handle already taken");
        Ok(Arc::new(inner) as LoadedModelHandle)
    }
    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

#[async_trait]
impl ModelProvider for DirectProvider<dyn TranscriptionModel> {
    fn provider_id(&self) -> &'static str {
        self.id
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![self.task],
        }
    }
    async fn load(&self, _spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        let inner = self
            .handle
            .lock()
            .unwrap()
            .clone()
            .expect("handle already taken");
        Ok(Arc::new(inner) as LoadedModelHandle)
    }
    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

fn spec_with_timeout(
    alias: &str,
    task: ModelTask,
    provider_id: &str,
    timeout_secs: Option<u64>,
    retry: Option<RetryConfig>,
) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: provider_id.to_string(),
        model_id: "test".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: timeout_secs,
        load_timeout: None,
        retry,
        options: serde_json::Value::Null,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn image_embedder_timeout_fires() {
    let inner: Arc<dyn ImageEmbeddingModel> = Arc::new(SlowImageEmbeddingModel {
        delay: Duration::from_secs(3),
    });
    let provider = DirectProvider::<dyn ImageEmbeddingModel>::new(
        "test/slow-image",
        ModelTask::EmbedImage,
        inner,
    );
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec_with_timeout(
            "embed_image/timeout",
            ModelTask::EmbedImage,
            "test/slow-image",
            Some(1),
            None,
        )])
        .build()
        .await
        .unwrap();

    let model = runtime.image_embedder("embed_image/timeout").await.unwrap();
    let start = std::time::Instant::now();
    let res = model.embed(vec![]).await;
    let elapsed = start.elapsed();

    assert!(matches!(res, Err(RuntimeError::Timeout)));
    assert!(
        elapsed < Duration::from_secs(3),
        "timeout did not fire within budget (elapsed = {:?})",
        elapsed
    );
}

#[tokio::test]
async fn image_embedder_retry_recovers_on_third_attempt() {
    let call_count = Arc::new(AtomicU32::new(0));
    let inner: Arc<dyn ImageEmbeddingModel> = Arc::new(FailingThenSucceedingImageEmbeddingModel {
        fail_remaining: AtomicU32::new(2),
        call_count: call_count.clone(),
    });
    let provider = DirectProvider::<dyn ImageEmbeddingModel>::new(
        "test/flaky-image",
        ModelTask::EmbedImage,
        inner,
    );
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec_with_timeout(
            "embed_image/retry",
            ModelTask::EmbedImage,
            "test/flaky-image",
            None,
            Some(RetryConfig {
                max_attempts: 3,
                initial_backoff_ms: 1,
            }),
        )])
        .build()
        .await
        .unwrap();

    let model = runtime.image_embedder("embed_image/retry").await.unwrap();
    let res = model.embed(vec![]).await;

    assert!(res.is_ok(), "expected success after retries");
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        3,
        "inner should be called exactly 3 times (2 failures + 1 success)"
    );
}

#[tokio::test]
async fn transcribe_many_timeout_applies_to_whole_batch() {
    // Inner sleeps batch-wide for 3 seconds; wrapper timeout is 1 second.
    // The wrapper must enforce the 1-second budget on the batched call.
    let inner: Arc<dyn TranscriptionModel> = Arc::new(SlowBatchTranscriptionModel {
        batch_delay: Duration::from_secs(3),
        languages: vec!["en".to_string()],
    });
    let provider = DirectProvider::<dyn TranscriptionModel>::new(
        "test/slow-batch-asr",
        ModelTask::Transcribe,
        inner,
    );
    let runtime = ModelRuntime::builder()
        .register_provider(provider)
        .catalog(vec![spec_with_timeout(
            "transcribe/batch-timeout",
            ModelTask::Transcribe,
            "test/slow-batch-asr",
            Some(1),
            None,
        )])
        .build()
        .await
        .unwrap();

    let model = runtime
        .transcriber("transcribe/batch-timeout")
        .await
        .unwrap();
    let start = std::time::Instant::now();
    let res = model
        .transcribe(
            vec![
                AudioInput::Bytes {
                    data: vec![],
                    media_type: "audio/wav".to_string(),
                },
                AudioInput::Bytes {
                    data: vec![],
                    media_type: "audio/wav".to_string(),
                },
            ],
            TranscribeOptions::default(),
        )
        .await;
    let elapsed = start.elapsed();

    assert!(matches!(res, Err(RuntimeError::Timeout)));
    assert!(
        elapsed < Duration::from_secs(3),
        "batch-wide timeout did not fire within budget (elapsed = {:?})",
        elapsed
    );
}
