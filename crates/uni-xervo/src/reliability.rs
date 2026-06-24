//! Reliability primitives: circuit breaker, instrumented model wrappers with
//! timeout and retry support, and metrics emission.

use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, GenerationOptions, GenerationResult, GeneratorModel, Message, RawTensorModel,
    RerankerModel, ScoredDoc, TensorBatch, TensorSpec,
};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Internal circuit breaker state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed,
    Open,
    HalfOpen,
}

/// Tunable parameters for the circuit breaker.
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before the breaker opens.
    pub failure_threshold: u32,
    /// Seconds to wait in the open state before allowing a probe call.
    pub open_wait_seconds: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_wait_seconds: 10,
        }
    }
}

struct Inner {
    state: State,
    failures: u32,
    last_failure: Option<Instant>,
    config: CircuitBreakerConfig,
    half_open_probe_in_flight: bool,
}

/// Thread-safe circuit breaker that tracks failures and short-circuits calls
/// when a provider is unhealthy.
///
/// State transitions: **Closed** -> (failures >= threshold) -> **Open** ->
/// (wait period elapsed) -> **HalfOpen** -> (probe succeeds) -> **Closed**
/// (or probe fails -> back to **Open**).
#[derive(Clone)]
pub struct CircuitBreakerWrapper {
    inner: Arc<Mutex<Inner>>,
}

impl CircuitBreakerWrapper {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                state: State::Closed,
                failures: 0,
                last_failure: None,
                config,
                half_open_probe_in_flight: false,
            })),
        }
    }

    /// Execute `f` through the circuit breaker.
    ///
    /// Returns [`RuntimeError::Unavailable`] immediately when the breaker is
    /// open.  In the half-open state only a single probe call is allowed;
    /// concurrent callers receive `Unavailable` until the probe completes.
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let is_probe_call;

        // 1. Check state
        {
            let mut inner = self.inner.lock().unwrap();
            match inner.state {
                State::Open => {
                    if let Some(last) = inner.last_failure {
                        if last.elapsed() >= Duration::from_secs(inner.config.open_wait_seconds) {
                            inner.state = State::HalfOpen;
                        } else {
                            return Err(RuntimeError::Unavailable);
                        }
                    }
                }
                State::HalfOpen => {
                    if inner.half_open_probe_in_flight {
                        return Err(RuntimeError::Unavailable);
                    }
                }
                State::Closed => {}
            }
            is_probe_call = inner.state == State::HalfOpen;
            if is_probe_call {
                inner.half_open_probe_in_flight = true;
            }
        }

        // 2. Execute
        let result = f().await;

        // 3. Update state
        let mut inner = self.inner.lock().unwrap();
        match result {
            Ok(val) => {
                if is_probe_call {
                    inner.state = State::Closed;
                    inner.failures = 0;
                    inner.half_open_probe_in_flight = false;
                } else if inner.state == State::Closed {
                    inner.failures = 0;
                }
                Ok(val)
            }
            Err(e) => {
                if is_probe_call {
                    inner.half_open_probe_in_flight = false;
                }
                inner.failures += 1;
                inner.last_failure = Some(Instant::now());

                if is_probe_call
                    || (inner.state == State::Closed
                        && inner.failures >= inner.config.failure_threshold)
                {
                    inner.state = State::Open;
                }
                Err(e)
            }
        }
    }
}

/// Wrapper around an [`EmbeddingModel`] that adds per-call timeout enforcement,
/// exponential-backoff retries for transient errors, and metrics emission
/// (`model_inference.duration_seconds`, `model_inference.total`).
pub struct InstrumentedEmbeddingModel {
    pub inner: Arc<dyn EmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl EmbeddingModel for InstrumentedEmbeddingModel {
    async fn embed(&self, texts: &[&str]) -> Result<crate::traits::EmbedResult> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.embed(texts);

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying embedding call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "embed",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "embed",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`GeneratorModel`] that adds timeout, retry, and metrics.
///
/// See [`InstrumentedEmbeddingModel`] for details on the instrumentation behavior.
pub struct InstrumentedGeneratorModel {
    pub inner: Arc<dyn GeneratorModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl GeneratorModel for InstrumentedGeneratorModel {
    async fn generate(
        &self,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.generate(messages, options.clone());

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying generation call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "generate",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "generate",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedGeneratorModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around an [`RawTensorModel`] that adds timeout, retry, and metrics.
pub struct InstrumentedRawTensorModel {
    pub inner: Arc<dyn RawTensorModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl RawTensorModel for InstrumentedRawTensorModel {
    async fn run(&self, inputs: &TensorBatch) -> Result<TensorBatch> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.run(inputs);

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying ONNX run call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(rewrite_onnx_error_alias(e, &self.alias)),
            }
        };

        record_onnx_metrics(
            &self.alias,
            &self.provider_id,
            start.elapsed(),
            if res.is_ok() { "success" } else { "failure" },
        );
        res
    }

    async fn run_batch(&self, inputs: &[TensorBatch]) -> Result<Vec<TensorBatch>> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.run_batch(inputs);

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying ONNX run_batch call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(rewrite_onnx_error_alias(e, &self.alias)),
            }
        };

        record_onnx_metrics(
            &self.alias,
            &self.provider_id,
            start.elapsed(),
            if res.is_ok() { "success" } else { "failure" },
        );
        res
    }

    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size()
    }

    fn input_signature(&self) -> &[TensorSpec] {
        self.inner.input_signature()
    }

    fn output_signature(&self) -> &[TensorSpec] {
        self.inner.output_signature()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedRawTensorModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

fn rewrite_onnx_error_alias(error: RuntimeError, alias: &str) -> RuntimeError {
    match error {
        RuntimeError::OnnxModelNotFound { path, .. } => RuntimeError::OnnxModelNotFound {
            alias: alias.to_string(),
            path,
        },
        RuntimeError::OnnxArtifactSelectionFailure { cause, .. } => {
            RuntimeError::OnnxArtifactSelectionFailure {
                alias: alias.to_string(),
                cause,
            }
        }
        RuntimeError::OnnxDownloadFailure { cause, .. } => RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause,
        },
        RuntimeError::OnnxLoadFailure { path, cause, .. } => RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path,
            cause,
        },
        RuntimeError::OnnxSignatureIntrospectionFailure { cause, .. } => {
            RuntimeError::OnnxSignatureIntrospectionFailure {
                alias: alias.to_string(),
                cause,
            }
        }
        RuntimeError::OnnxInputMissing { required_input, .. } => RuntimeError::OnnxInputMissing {
            alias: alias.to_string(),
            required_input,
        },
        RuntimeError::OnnxInputTypeMismatch {
            input_name,
            expected,
            got,
            ..
        } => RuntimeError::OnnxInputTypeMismatch {
            alias: alias.to_string(),
            input_name,
            expected,
            got,
        },
        RuntimeError::OnnxInputShapeMismatch {
            input_name,
            expected,
            got,
            ..
        } => RuntimeError::OnnxInputShapeMismatch {
            alias: alias.to_string(),
            input_name,
            expected,
            got,
        },
        RuntimeError::OnnxInvocationFailure { cause, .. } => RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause,
        },
        RuntimeError::OnnxBatchStackingFailure { cause, .. } => {
            RuntimeError::OnnxBatchStackingFailure {
                alias: alias.to_string(),
                cause,
            }
        }
        other => other,
    }
}

fn record_onnx_metrics(alias: &str, provider_id: &str, duration: Duration, status: &str) {
    metrics::histogram!(
        "model_inference.duration_seconds",
        "alias" => alias.to_string(),
        "task" => "raw",
        "provider" => provider_id.to_string()
    )
    .record(duration.as_secs_f64());

    metrics::counter!(
        "model_inference.total",
        "alias" => alias.to_string(),
        "task" => "raw",
        "provider" => provider_id.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

/// Wrapper around a [`RerankerModel`] that adds timeout, retry, and metrics.
///
/// See [`InstrumentedEmbeddingModel`] for details on the instrumentation behavior.
pub struct InstrumentedRerankerModel {
    pub inner: Arc<dyn RerankerModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl RerankerModel for InstrumentedRerankerModel {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        let start = Instant::now();
        let mut attempts = 0;
        let max_attempts = self.retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

        let res = loop {
            attempts += 1;
            let fut = self.inner.rerank(query, docs);

            let res = if let Some(timeout) = self.timeout {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(r) => r,
                    Err(_) => Err(RuntimeError::Timeout),
                }
            } else {
                fut.await
            };

            match res {
                Ok(val) => break Ok(val),
                Err(e) if e.is_retryable() && attempts < max_attempts => {
                    let backoff = self.retry.as_ref().unwrap().get_backoff(attempts);
                    tracing::warn!(
                        alias = %self.alias,
                        attempt = attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %e,
                        "Retrying rerank call"
                    );
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => break Err(e),
            }
        };

        let duration = start.elapsed();
        let status = if res.is_ok() { "success" } else { "failure" };

        metrics::histogram!(
            "model_inference.duration_seconds",
            "alias" => self.alias.clone(),
            "task" => "rerank",
            "provider" => self.provider_id.clone()
        )
        .record(duration.as_secs_f64());

        metrics::counter!(
            "model_inference.total",
            "alias" => self.alias.clone(),
            "task" => "rerank",
            "provider" => self.provider_id.clone(),
            "status" => status
        )
        .increment(1);

        res
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedRerankerModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

// ---------------------------------------------------------------------------
// Multimodal instrumentation wrappers.
//
// Each wraps a future-producing call (the trait's business method) with the
// same timeout-then-retry-then-metrics pattern used by
// `InstrumentedEmbeddingModel` and `InstrumentedGeneratorModel`. The pattern
// is centralized into the helper `run_instrumented` below so the seven new
// wrappers stay short. Existing wrappers keep their inlined form (could be
// migrated to this helper in a follow-up).
// ---------------------------------------------------------------------------

/// Run an async operation with the standard timeout / retry / metrics
/// envelope used by every instrumented model wrapper.
///
/// `make_fut` is a closure invoked once per attempt — it must produce a fresh
/// future each call so retries operate on a fresh inner call rather than
/// re-polling a completed future. `task` is the wire-format task name used
/// for the metric labels (`embed_image`, `nlp`, etc.).
async fn run_instrumented<F, Fut, T>(
    alias: &str,
    provider_id: &str,
    task: &'static str,
    timeout: Option<Duration>,
    retry: Option<&crate::api::RetryConfig>,
    mut make_fut: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let start = Instant::now();
    let mut attempts = 0u32;
    let max_attempts = retry.as_ref().map(|r| r.max_attempts).unwrap_or(1);

    let res = loop {
        attempts += 1;
        let fut = make_fut();
        let attempt_res = if let Some(t) = timeout {
            match tokio::time::timeout(t, fut).await {
                Ok(r) => r,
                Err(_) => Err(RuntimeError::Timeout),
            }
        } else {
            fut.await
        };

        match attempt_res {
            Ok(val) => break Ok(val),
            Err(e) if e.is_retryable() && attempts < max_attempts => {
                let backoff = retry.as_ref().unwrap().get_backoff(attempts);
                tracing::warn!(
                    alias = %alias,
                    attempt = attempts,
                    backoff_ms = backoff.as_millis(),
                    error = %e,
                    "Retrying {task} call",
                );
                tokio::time::sleep(backoff).await;
                continue;
            }
            Err(e) => break Err(e),
        }
    };

    let duration = start.elapsed();
    let status = if res.is_ok() { "success" } else { "failure" };
    metrics::histogram!(
        "model_inference.duration_seconds",
        "alias" => alias.to_string(),
        "task" => task,
        "provider" => provider_id.to_string()
    )
    .record(duration.as_secs_f64());
    metrics::counter!(
        "model_inference.total",
        "alias" => alias.to_string(),
        "task" => task,
        "provider" => provider_id.to_string(),
        "status" => status
    )
    .increment(1);

    res
}

/// Wrapper around an [`ImageEmbeddingModel`](crate::traits::ImageEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedImageEmbeddingModel {
    pub inner: Arc<dyn crate::traits::ImageEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::ImageEmbeddingModel for InstrumentedImageEmbeddingModel {
    async fn embed(
        &self,
        images: Vec<crate::traits::ImageInput>,
    ) -> Result<crate::traits::EmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_image",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(images.clone()),
        )
        .await
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedImageEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around an [`AudioEmbeddingModel`](crate::traits::AudioEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedAudioEmbeddingModel {
    pub inner: Arc<dyn crate::traits::AudioEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::AudioEmbeddingModel for InstrumentedAudioEmbeddingModel {
    async fn embed(
        &self,
        audios: Vec<crate::traits::AudioInput>,
    ) -> Result<crate::traits::EmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_audio",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(audios.clone()),
        )
        .await
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedAudioEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`MultimodalEmbeddingModel`](crate::traits::MultimodalEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedMultimodalEmbeddingModel {
    pub inner: Arc<dyn crate::traits::MultimodalEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::MultimodalEmbeddingModel for InstrumentedMultimodalEmbeddingModel {
    async fn embed(
        &self,
        inputs: Vec<crate::traits::MultimodalInput>,
    ) -> Result<crate::traits::EmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_multimodal",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(inputs.clone()),
        )
        .await
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    fn supported_modalities(&self) -> &[crate::traits::Modality] {
        self.inner.supported_modalities()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedMultimodalEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`SparseEmbeddingModel`](crate::traits::SparseEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedSparseEmbeddingModel {
    pub inner: Arc<dyn crate::traits::SparseEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::SparseEmbeddingModel for InstrumentedSparseEmbeddingModel {
    async fn embed(&self, texts: &[&str]) -> Result<crate::traits::SparseEmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_sparse",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(texts),
        )
        .await
    }

    fn vocab_size(&self) -> u32 {
        self.inner.vocab_size()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedSparseEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`MultiVectorEmbeddingModel`](crate::traits::MultiVectorEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedMultiVectorEmbeddingModel {
    pub inner: Arc<dyn crate::traits::MultiVectorEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::MultiVectorEmbeddingModel for InstrumentedMultiVectorEmbeddingModel {
    async fn embed(&self, texts: &[&str]) -> Result<crate::traits::MultiVectorEmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_multi_vector",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(texts),
        )
        .await
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedMultiVectorEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`HybridEmbeddingModel`](crate::traits::HybridEmbeddingModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedHybridEmbeddingModel {
    pub inner: Arc<dyn crate::traits::HybridEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::HybridEmbeddingModel for InstrumentedHybridEmbeddingModel {
    async fn embed(
        &self,
        texts: &[&str],
        heads: crate::traits::HeadSet,
    ) -> Result<crate::traits::HybridEmbedResult> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "embed_hybrid",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.embed(texts, heads),
        )
        .await
    }

    fn available_heads(&self) -> crate::traits::HeadSet {
        self.inner.available_heads()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedHybridEmbeddingModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around an [`NlpModel`](crate::traits::NlpModel) adding timeout / retry / metrics.
pub struct InstrumentedNlpModel {
    pub inner: Arc<dyn crate::traits::NlpModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::NlpModel for InstrumentedNlpModel {
    async fn analyze(
        &self,
        requests: Vec<crate::traits::NlpRequest<'_>>,
    ) -> Result<Vec<crate::traits::NlpResult>> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "nlp",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.analyze(requests.clone()),
        )
        .await
    }

    fn supported_tasks(&self) -> crate::traits::NlpTasks {
        self.inner.supported_tasks()
    }

    fn label_maps(&self) -> Option<&crate::traits::NlpLabelMaps> {
        self.inner.label_maps()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedNlpModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`DocumentExtractionModel`](crate::traits::DocumentExtractionModel)
/// adding timeout / retry / metrics.
pub struct InstrumentedDocumentExtractionModel {
    pub inner: Arc<dyn crate::traits::DocumentExtractionModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::DocumentExtractionModel for InstrumentedDocumentExtractionModel {
    async fn extract(
        &self,
        pages: Vec<crate::traits::ImageInput>,
        options: crate::traits::DocExtractOptions,
    ) -> Result<Vec<crate::traits::DocExtractResult>> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "document_extract",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.extract(pages.clone(), options.clone()),
        )
        .await
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedDocumentExtractionModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around a [`TranscriptionModel`](crate::traits::TranscriptionModel)
/// adding timeout / retry / metrics to the batch `transcribe` call.
///
/// The timeout applies batch-wide — operators tuning `timeout` for batched
/// ingest need to budget for the full batch, not a per-item slice.
pub struct InstrumentedTranscriptionModel {
    pub inner: Arc<dyn crate::traits::TranscriptionModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::TranscriptionModel for InstrumentedTranscriptionModel {
    async fn transcribe(
        &self,
        audios: Vec<crate::traits::AudioInput>,
        options: crate::traits::TranscribeOptions,
    ) -> Result<Vec<crate::traits::TranscribeResult>> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "transcribe",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.transcribe(audios.clone(), options.clone()),
        )
        .await
    }

    fn supported_languages(&self) -> &[String] {
        self.inner.supported_languages()
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedTranscriptionModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

/// Wrapper around an [`OcrModel`](crate::traits::OcrModel) adding timeout / retry / metrics.
pub struct InstrumentedOcrModel {
    pub inner: Arc<dyn crate::traits::OcrModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<crate::api::RetryConfig>,
}

#[async_trait]
impl crate::traits::OcrModel for InstrumentedOcrModel {
    async fn recognize(
        &self,
        images: Vec<crate::traits::ImageInput>,
    ) -> Result<Vec<crate::traits::OcrResult>> {
        run_instrumented(
            &self.alias,
            &self.provider_id,
            "ocr",
            self.timeout,
            self.retry.as_ref(),
            || self.inner.recognize(images.clone()),
        )
        .await
    }

    async fn warmup(&self) -> Result<()> {
        self.inner.warmup().await
    }
}

impl crate::traits::ModelInfo for InstrumentedOcrModel {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn active_execution_providers(&self) -> Vec<String> {
        self.inner.active_execution_providers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ModelInfo, NlpModel};
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_circuit_breaker_transitions() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            open_wait_seconds: 1,
        };
        let cb = CircuitBreakerWrapper::new(config);
        let counter = Arc::new(AtomicU32::new(0));

        // 1. Success calls - state remains Closed
        let res = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(res.is_ok());

        // 2. Failures - state transitions to Open
        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err()); // Fail 1

        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err()); // Fail 2 -> Open

        // 3. Open state - calls rejected immediately
        let res = cb
            .call(|| async {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .await;
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().to_string(), "Unavailable");
        assert_eq!(counter.load(Ordering::SeqCst), 0); // Should not have run

        // 4. Wait for HalfOpen
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // 5. HalfOpen - allow one call
        // If it fails, go back to Open
        let res = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;
        assert!(res.is_err());

        // Should be Open again
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().to_string(), "Unavailable");

        // 6. Wait again for HalfOpen
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // 7. Success - transition to Closed
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_ok());

        // Should be closed now, next call works
        let res = cb.call(|| async { Ok(()) }).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_half_open_allows_single_probe() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_wait_seconds: 1,
        };
        let cb = CircuitBreakerWrapper::new(config);

        // Open breaker.
        let _ = cb
            .call(|| async { Err::<(), _>(RuntimeError::InferenceError("fail".into())) })
            .await;

        tokio::time::sleep(Duration::from_millis(1100)).await;

        let started = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let finished = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let cb_probe = cb.clone();
        let started_probe = started.clone();
        let finished_probe = finished.clone();
        let probe = tokio::spawn(async move {
            cb_probe
                .call(|| async move {
                    started_probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    finished_probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok::<_, RuntimeError>(())
                })
                .await
        });

        // Allow the first probe to enter.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // A concurrent call during half-open probe should fail fast.
        let second = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(matches!(second, Err(RuntimeError::Unavailable)));

        let probe_result = probe.await.unwrap();
        assert!(probe_result.is_ok());
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(finished.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Closed again.
        let res = cb.call(|| async { Ok::<_, RuntimeError>(()) }).await;
        assert!(res.is_ok());
    }

    /// A minimal `NlpModel` whose `label_maps` returns a marker vocabulary, used
    /// to verify the instrumentation wrapper forwards introspection methods.
    struct LabelMapNlpModel {
        labels: crate::traits::NlpLabelMaps,
    }

    #[async_trait]
    impl crate::traits::NlpModel for LabelMapNlpModel {
        async fn analyze(
            &self,
            _requests: Vec<crate::traits::NlpRequest<'_>>,
        ) -> Result<Vec<crate::traits::NlpResult>> {
            Ok(Vec::new())
        }

        fn supported_tasks(&self) -> crate::traits::NlpTasks {
            crate::traits::NlpTasks::CLS
        }

        fn label_maps(&self) -> Option<&crate::traits::NlpLabelMaps> {
            Some(&self.labels)
        }
    }

    impl crate::traits::ModelInfo for LabelMapNlpModel {
        fn model_id(&self) -> &str {
            "mock/labelmaps"
        }
    }

    /// The instrumentation wrapper forwards `label_maps()` to the inner model
    /// rather than falling back to the trait default `None` (R5 regression: the
    /// runtime wraps every model, so a non-forwarded method is invisible to
    /// callers).
    #[tokio::test]
    async fn instrumented_nlp_forwards_label_maps() {
        let labels = crate::traits::NlpLabelMaps {
            cls: vec!["statement".to_string(), "question".to_string()],
            ..Default::default()
        };
        let inner: Arc<dyn crate::traits::NlpModel> = Arc::new(LabelMapNlpModel { labels });
        let wrapped = InstrumentedNlpModel {
            inner,
            alias: "nlp/x".to_string(),
            provider_id: "test".to_string(),
            timeout: None,
            retry: None,
        };

        let maps = wrapped.label_maps().expect("wrapper forwards label_maps");
        assert_eq!(maps.cls, ["statement", "question"]);
        assert_eq!(wrapped.model_id(), "mock/labelmaps");
        assert_eq!(wrapped.supported_tasks(), crate::traits::NlpTasks::CLS);
    }

    /// A model without label maps reports `None` through the wrapper, confirming
    /// the default path is preserved (remote/mock providers stay `None`).
    #[tokio::test]
    async fn instrumented_nlp_label_maps_none_passes_through() {
        let inner: Arc<dyn crate::traits::NlpModel> = Arc::new(crate::mock::MockNlpModel::new());
        let wrapped = InstrumentedNlpModel {
            inner,
            alias: "nlp/y".to_string(),
            provider_id: "test".to_string(),
            timeout: None,
            retry: None,
        };
        assert!(wrapped.label_maps().is_none());
    }
}
