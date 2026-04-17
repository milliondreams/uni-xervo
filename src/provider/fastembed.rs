use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
};
use anyhow::anyhow;
use async_trait::async_trait;
use fastembed::{ExecutionProviderDispatch, InitOptions, TextEmbedding};
use ort::ep::{CPU, CUDA};
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::oneshot;

/// Local embedding provider using [FastEmbed](https://github.com/Anush008/fastembed-rs)
/// (ONNX Runtime).
///
/// Supports a wide range of embedding models. Inference is offloaded to a
/// dedicated thread with an enlarged stack to accommodate ONNX Runtime's
/// requirements.
pub struct LocalFastEmbedProvider;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FastEmbedExecutionProvider {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone)]
struct LocalFastEmbedOptions {
    execution_providers: Option<Vec<FastEmbedExecutionProvider>>,
}

impl LocalFastEmbedProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalFastEmbedProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for LocalFastEmbedProvider {
    fn provider_id(&self) -> &'static str {
        "local/fastembed"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Embed],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        if spec.task != ModelTask::Embed {
            return Err(RuntimeError::CapabilityMismatch(format!(
                "FastEmbed provider does not support task {:?}",
                spec.task
            )));
        }

        let model_name = spec.model_id.clone();
        let cache_dir = crate::cache::resolve_cache_dir("fastembed", &model_name, &spec.options);
        let provider_options = LocalFastEmbedOptions::from_value(&spec.options)?;

        // Offload initialization to a blocking thread because it can refer to onnxruntime which might be heavy
        // fastembed init might block.
        let service = tokio::task::spawn_blocking(move || {
            FastEmbedService::new(&model_name, &cache_dir, &provider_options)
        })
        .await
        .map_err(|e| RuntimeError::Load(format!("Join error: {}", e)))?
        .map_err(|e| RuntimeError::Load(e.to_string()))?;

        let handle: Arc<dyn EmbeddingModel> = Arc::new(service);
        Ok(Arc::new(handle) as LoadedModelHandle)
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// Stack size for embedding threads.
const EMBEDDING_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Wrapper around a [`TextEmbedding`] instance that implements
/// [`EmbeddingModel`].
///
/// Each inference call spawns a short-lived worker thread with a larger stack
/// to satisfy ONNX Runtime's stack requirements.
pub struct FastEmbedService {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: u32,
}

impl FastEmbedService {
    fn new(
        model_name: &str,
        cache_dir: &Path,
        provider_options: &LocalFastEmbedOptions,
    ) -> anyhow::Result<Self> {
        let model_enum = match model_name {
            "AllMiniLML6V2" | "all-MiniLM-L6-v2" => fastembed::EmbeddingModel::AllMiniLML6V2,
            "AllMiniLML6V2Q" => fastembed::EmbeddingModel::AllMiniLML6V2Q,
            "AllMiniLML12V2" => fastembed::EmbeddingModel::AllMiniLML12V2,
            "AllMiniLML12V2Q" => fastembed::EmbeddingModel::AllMiniLML12V2Q,
            "AllMpnetBaseV2" | "all-mpnet-base-v2" => fastembed::EmbeddingModel::AllMpnetBaseV2,
            "BGEBaseENV15" | "bge-base-en-v1.5" => fastembed::EmbeddingModel::BGEBaseENV15,
            "BGEBaseENV15Q" => fastembed::EmbeddingModel::BGEBaseENV15Q,
            "BGELargeENV15" | "bge-large-en-v1.5" => fastembed::EmbeddingModel::BGELargeENV15,
            "BGELargeENV15Q" => fastembed::EmbeddingModel::BGELargeENV15Q,
            "BGESmallENV15" | "bge-small-en-v1.5" => fastembed::EmbeddingModel::BGESmallENV15,
            "BGESmallENV15Q" => fastembed::EmbeddingModel::BGESmallENV15Q,
            "NomicEmbedTextV1" => fastembed::EmbeddingModel::NomicEmbedTextV1,
            "NomicEmbedTextV15" | "nomic-embed-text-v1.5" => {
                fastembed::EmbeddingModel::NomicEmbedTextV15
            }
            "NomicEmbedTextV15Q" => fastembed::EmbeddingModel::NomicEmbedTextV15Q,
            "ParaphraseMLMiniLML12V2" => fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMiniLML12V2Q" => fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            "ParaphraseMLMpnetBaseV2" => fastembed::EmbeddingModel::ParaphraseMLMpnetBaseV2,
            "BGESmallZHV15" => fastembed::EmbeddingModel::BGESmallZHV15,
            "BGELargeZHV15" => fastembed::EmbeddingModel::BGELargeZHV15,
            "BGEM3" => fastembed::EmbeddingModel::BGEM3,
            "ModernBertEmbedLarge" => fastembed::EmbeddingModel::ModernBertEmbedLarge,
            "MultilingualE5Small" | "multilingual-e5-small" => {
                fastembed::EmbeddingModel::MultilingualE5Small
            }
            "MultilingualE5Base" | "multilingual-e5-base" => {
                fastembed::EmbeddingModel::MultilingualE5Base
            }
            "MultilingualE5Large" | "multilingual-e5-large" => {
                fastembed::EmbeddingModel::MultilingualE5Large
            }
            "MxbaiEmbedLargeV1" | "mxbai-embed-large-v1" => {
                fastembed::EmbeddingModel::MxbaiEmbedLargeV1
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported FastEmbed model: {}. Please check fastembed docs for supported models.",
                    model_name
                ));
            }
        };

        let mut options = InitOptions::new(model_enum.clone());
        options = options.with_cache_dir(cache_dir.to_path_buf());
        options = options.with_execution_providers(build_execution_providers(
            provider_options.execution_providers.as_deref(),
        )?);

        let model = TextEmbedding::try_new(options)
            .map_err(|e| anyhow!("Failed to initialize FastEmbed model: {}", e))?;

        // Determine dimensions
        let dimensions = match model_enum {
            fastembed::EmbeddingModel::AllMiniLML6V2
            | fastembed::EmbeddingModel::AllMiniLML6V2Q
            | fastembed::EmbeddingModel::AllMiniLML12V2
            | fastembed::EmbeddingModel::AllMiniLML12V2Q
            | fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2
            | fastembed::EmbeddingModel::ParaphraseMLMiniLML12V2Q
            | fastembed::EmbeddingModel::BGESmallENV15
            | fastembed::EmbeddingModel::BGESmallENV15Q
            | fastembed::EmbeddingModel::MultilingualE5Small => 384,

            fastembed::EmbeddingModel::BGESmallZHV15 => 512,

            fastembed::EmbeddingModel::AllMpnetBaseV2
            | fastembed::EmbeddingModel::ParaphraseMLMpnetBaseV2
            | fastembed::EmbeddingModel::BGEBaseENV15
            | fastembed::EmbeddingModel::BGEBaseENV15Q
            | fastembed::EmbeddingModel::NomicEmbedTextV1
            | fastembed::EmbeddingModel::NomicEmbedTextV15
            | fastembed::EmbeddingModel::NomicEmbedTextV15Q
            | fastembed::EmbeddingModel::MultilingualE5Base => 768,

            fastembed::EmbeddingModel::BGELargeENV15
            | fastembed::EmbeddingModel::BGELargeENV15Q
            | fastembed::EmbeddingModel::BGELargeZHV15
            | fastembed::EmbeddingModel::BGEM3
            | fastembed::EmbeddingModel::ModernBertEmbedLarge
            | fastembed::EmbeddingModel::MultilingualE5Large
            | fastembed::EmbeddingModel::MxbaiEmbedLargeV1 => 1024,

            _ => {
                // Fallback for new models or quantized variants not explicitly listed
                // We could log a warning here or return a default.
                // Assuming 768 is a safe-ish bet for unknown models or 1024 for "Large" ones.
                // Better approach: Since we can't easily probe without loading, we might just
                // assume a default and let the user override via config if needed.
                // But for now, to satisfy exhaustiveness:
                1024
            }
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: model_name.to_string(),
            dimensions,
        })
    }
}

impl LocalFastEmbedOptions {
    fn from_value(value: &Value) -> Result<Self> {
        let map = value.as_object();
        let execution_providers = map
            .and_then(|m| m.get("execution_providers"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(FastEmbedExecutionProvider::from_str)
                    .collect::<Vec<_>>()
            });

        Ok(Self {
            execution_providers,
        })
    }
}

impl FastEmbedExecutionProvider {
    fn from_str(value: &str) -> Self {
        match value {
            "cuda" => Self::Cuda,
            _ => Self::Cpu,
        }
    }
}

fn build_execution_providers(
    configured: Option<&[FastEmbedExecutionProvider]>,
) -> anyhow::Result<Vec<ExecutionProviderDispatch>> {
    let providers = configured
        .map(|value| value.to_vec())
        .unwrap_or_else(default_execution_providers);
    let cpu_present = providers.contains(&FastEmbedExecutionProvider::Cpu);
    let last_index = providers.len().saturating_sub(1);

    providers
        .into_iter()
        .enumerate()
        .map(|(index, provider)| {
            let strict = configured.is_some() && !cpu_present && index == last_index;
            execution_provider_dispatch(provider, strict)
        })
        .collect()
}

fn default_execution_providers() -> Vec<FastEmbedExecutionProvider> {
    #[cfg(feature = "gpu-cuda")]
    {
        vec![
            FastEmbedExecutionProvider::Cuda,
            FastEmbedExecutionProvider::Cpu,
        ]
    }
    #[cfg(not(feature = "gpu-cuda"))]
    {
        vec![FastEmbedExecutionProvider::Cpu]
    }
}

fn execution_provider_dispatch(
    provider: FastEmbedExecutionProvider,
    strict: bool,
) -> anyhow::Result<ExecutionProviderDispatch> {
    let dispatch = match provider {
        FastEmbedExecutionProvider::Cpu => CPU::default().build(),
        FastEmbedExecutionProvider::Cuda => {
            if !cfg!(feature = "gpu-cuda") {
                return Err(anyhow!(
                    "FastEmbed requested CUDA execution, but gpu-cuda is not enabled"
                ));
            }
            CUDA::default().build()
        }
    };

    Ok(if strict {
        dispatch.error_on_failure()
    } else {
        dispatch.fail_silently()
    })
}

#[async_trait]
impl EmbeddingModel for FastEmbedService {
    async fn embed(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let texts_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model = self.model.clone();

        let (tx, rx) = oneshot::channel();

        // Spawn a dedicated thread with larger stack for ONNX Runtime
        thread::Builder::new()
            .name("fastembed-worker".to_string())
            .stack_size(EMBEDDING_THREAD_STACK_SIZE)
            .spawn(move || {
                let result = model
                    .lock()
                    .map_err(|_| anyhow!("Failed to lock embedding model"))
                    .and_then(|mut guard| {
                        guard
                            .embed(texts_vec, None)
                            .map_err(|e| anyhow!("FastEmbed error: {}", e))
                    });
                let _ = tx.send(result);
            })
            .map_err(|e| {
                RuntimeError::InferenceError(format!("Failed to spawn embedding thread: {}", e))
            })?;

        let result = rx
            .await
            .map_err(|_| RuntimeError::InferenceError("Embedding thread panicked".to_string()))?;

        result.map_err(|e| RuntimeError::InferenceError(e.to_string()))
    }

    fn dimensions(&self) -> u32 {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::{FastEmbedExecutionProvider, default_execution_providers};

    #[test]
    fn default_execution_providers_include_cpu() {
        let providers = default_execution_providers();
        assert!(providers.contains(&FastEmbedExecutionProvider::Cpu));
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn default_execution_providers_prefer_cuda_when_enabled() {
        let providers = default_execution_providers();
        assert_eq!(
            providers,
            vec![
                FastEmbedExecutionProvider::Cuda,
                FastEmbedExecutionProvider::Cpu,
            ]
        );
    }
}
