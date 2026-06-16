//! Real-inference tests that load specific model families through
//! `LocalMistralRsProvider` and generate a small completion.
//!
//! Models with test bodies: Gemma 3n E2B, Gemma 4 E2B, Gemma 4 E4B.
//! Constants for Qwen 3.5-VL and GLM exist as future placeholders and
//! aren't exercised yet.
//!
//! Each model has up to three device/loader variants:
//!
//! - **CPU + ISQ**: forced to CPU with `Q4K` in-situ quantization. Loads
//!   bf16 weights then quantizes in memory; slow load, fits anywhere.
//! - **CPU + UQFF**: loads pre-quantized UQFF weights from
//!   `mistralrs-community/*-UQFF` repos. Skips the bf16 step → roughly
//!   2× faster load than CPU + ISQ for the same model.
//! - **GPU**: auto-selected device (CUDA when `gpu-cuda` is enabled,
//!   Metal when `gpu-metal` is enabled) with auto-device-mapper
//!   overrides sized for 8 GB VRAM. Compiled out on CPU-only builds.
//!   Larger Gemma 4 variants need ≥12 GB (E2B) or ≥24 GB (E4B) of VRAM
//!   in practice — the residual.safetensors plus the planner's
//!   bf16-sized layer reservation overflow 8 GB even with UQFF.
//!
//! All tests are gated by both:
//!
//! - `#[cfg(feature = "provider-mistralrs")]` — only built when the
//!   provider is enabled. GPU tests additionally require `gpu-cuda` or
//!   `gpu-metal`.
//! - `EXPENSIVE_TESTS=1` env var (via `require_expensive_tests!`) — each
//!   test downloads multi-GB weights from HuggingFace and runs real
//!   inference, so they are skipped unless explicitly opted in.
//!
//! Run with:
//!
//! ```sh
//! # CPU (ISQ) only
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features "provider-mistralrs" \
//!     --test mistralrs_models_expensive_test \
//!     --run-ignored all
//!
//! # CPU + GPU
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features "provider-mistralrs,gpu-cuda" \
//!     --test mistralrs_models_expensive_test \
//!     --run-ignored all
//! ```
//!
//! NOTE: model identifiers below target the smallest publicly available
//! variant of each family at time of writing. They may need updating when
//! new minor revisions ship — these are HF repo strings, not URLs, and
//! the tests are `#[ignore]`'d so a stale ID will not break CI.
//
// Rust guideline compliant

#![cfg(feature = "provider-mistralrs")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{GenerationOptions, Message};

// ---------------------------------------------------------------------------
// Test gating
// ---------------------------------------------------------------------------

fn should_run_expensive_tests() -> bool {
    // Truthy: "1" / "true" / "yes" (case-insensitive). Falsy: anything
    // else, including "0" / "false" / "no" / empty / unset. Matches the
    // CI convention so users can copy `EXPENSIVE_TESTS=1` from the docs
    // and see the tests run, while a leftover `EXPENSIVE_TESTS=0` in
    // the shell doesn't accidentally enable them.
    match env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// Model identifiers
//
// Smallest publicly available variant of each requested family. Update as
// new minor revisions ship.
// ---------------------------------------------------------------------------

const GEMMA_3N_MODEL_ID: &str = "google/gemma-3n-E2B-it";
// Gemma 4 follows the same E2B/E4B naming as 3n. The E2B variant is
// multimodal (Any-to-Any), so it must use the vision pipeline.
const GEMMA_4_MODEL_ID: &str = "google/gemma-4-E2B-it";
// Pre-quantized UQFF mirror of Gemma 4 E2B published by mistralrs-community.
// Used by both CPU and GPU tests to exercise the UQFF loading path.
const GEMMA_4_UQFF_MODEL_ID: &str = "mistralrs-community/gemma-4-E2B-it-UQFF";
// Gemma 4 E4B: ~8B effective params, multimodal (Any-to-Any). Bigger than
// E2B; bf16 footprint exceeds 16 GB and even q4k-UQFF + residual is
// ~11 GB at load. CPU is the realistic target on commodity hardware.
const GEMMA_4_E4B_MODEL_ID: &str = "google/gemma-4-E4B-it";
const GEMMA_4_E4B_UQFF_MODEL_ID: &str = "mistralrs-community/gemma-4-E4B-it-UQFF";
const QWEN_VL_MODEL_ID: &str = "Qwen/Qwen3-VL-2B-Instruct";
const GLM_MODEL_ID: &str = "THUDM/glm-edge-1.5b-chat";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a generator alias spec for a CPU run loading pre-quantized UQFF.
///
/// Exercises the UQFF code path without GPU. `force_cpu: true` keeps the
/// test reproducible on machines without CUDA/Metal.
fn cpu_uqff_spec(alias: &str, model_id: &str, uqff_file: &str, pipeline: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "force_cpu": true,
            "dtype": "f32",
            "pipeline": pipeline,
            "uqff_files": [uqff_file],
        }),
    }
}

/// Build a generator alias spec for a CPU + ISQ Q4K run.
fn cpu_isq_spec(alias: &str, model_id: &str, pipeline: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "force_cpu": true,
            "isq": "Q4K",
            "dtype": "f32",
            "pipeline": pipeline,
        }),
    }
}

/// Build a generator alias spec for a GPU run using UQFF (mistralrs's
/// pre-quantized format).
///
/// Use this helper for models that don't fit in VRAM at bf16 load time.
/// UQFF weights are loaded directly at quantized precision so the load
/// step never touches the unquantized footprint. `uqff_file` is the
/// first shard filename in the UQFF HF repo (e.g. "q4k-0.uqff");
/// remaining shards are auto-discovered.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
fn gpu_uqff_spec(alias: &str, model_id: &str, uqff_file: &str, pipeline: &str) -> ModelAliasSpec {
    let mut options = serde_json::json!({
        "pipeline": pipeline,
        "uqff_files": [uqff_file],
        // Aggressive minima for the auto-device-mapper. The planner sizes
        // KV-cache and image-buffer reservation from these values, eating
        // into the budget for weight placement. Smallest practical values
        // give the residual + quantized weights the most room on tiny GPUs.
        "max_seq_len": 256,
    });
    if pipeline == "vision" {
        let map = options.as_object_mut().unwrap();
        map.insert("max_image_shape".into(), serde_json::json!([112, 112]));
        map.insert("max_num_images".into(), serde_json::json!(1));
    }
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

/// Build a generator alias spec for a GPU run.
///
/// We apply ISQ Q4K on GPU as well as CPU because the test matrix is
/// expected to run on commodity GPUs (e.g. 8 GB laptop cards). bf16 of
/// even a small ~5B-param model exceeds that budget once the KV cache
/// is included; Q4K shrinks weights ~8× and lets the same VRAM hold a
/// usable context window.
///
/// The auto-device-mapper sizes layers from the *unquantized* dtype, so
/// we shrink `max_seq_len` and `max_image_shape` from their 4096 / 1024×1024
/// defaults to match what an 8 GB card can actually hold. Without these
/// overrides the planner concludes "Device cuda[0] can fit 0 layers" and
/// silently falls back to CPU (then crashes on the first bf16 op).
#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
fn gpu_spec(alias: &str, model_id: &str, pipeline: &str) -> ModelAliasSpec {
    let mut options = serde_json::json!({
        "dtype": "bf16",
        "isq": "Q4K",
        "pipeline": pipeline,
        "max_seq_len": 1024,
    });
    if pipeline == "vision" {
        let map = options.as_object_mut().unwrap();
        map.insert("max_image_shape".into(), serde_json::json!([224, 224]));
        map.insert("max_num_images".into(), serde_json::json!(1));
    }
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

/// Build runtime, load the generator, run a tiny completion, and assert
/// non-empty output. Shared body for every per-model test.
async fn run_generation_smoke_test(spec: ModelAliasSpec) {
    let alias = spec.alias.clone();
    let model_id = spec.model_id.clone();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap_or_else(|e| panic!("runtime build failed for {model_id}: {e:?}"));

    let model = runtime
        .generator(&alias)
        .await
        .unwrap_or_else(|e| panic!("loading {alias} ({model_id}) failed: {e:?}"));

    let messages = vec![Message::user("Say 'hello' and nothing else.")];
    let options = GenerationOptions {
        max_tokens: Some(16),
        temperature: Some(0.1),
        ..Default::default()
    };

    let result = model
        .generate(&messages, options)
        .await
        .unwrap_or_else(|e| panic!("generate call failed for {model_id}: {e:?}"));

    assert!(
        !result.text.is_empty(),
        "generated text must be non-empty for {model_id}"
    );
    assert!(
        result.usage.is_some(),
        "usage must be reported for {model_id}"
    );
    let usage = result.usage.unwrap();
    assert!(
        usage.total_tokens > 0,
        "total_tokens must be > 0 for {model_id}"
    );

    println!(
        "✓ {model_id} generated {} tokens: {}",
        usage.total_tokens, result.text
    );
}

// ---------------------------------------------------------------------------
// CPU (ISQ Q4K) tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_gemma_3n_cpu_isq() {
    require_expensive_tests!();
    // Gemma 3n is a multimodal model class
    // (`Gemma3nForConditionalGeneration`); mistralrs's TextModelBuilder
    // rejects it. Load through the vision pipeline and send a text-only
    // prompt — `MistralRsVisionService::generate` accepts messages
    // without image blocks.
    run_generation_smoke_test(cpu_isq_spec(
        "generate/gemma-3n-cpu-isq",
        GEMMA_3N_MODEL_ID,
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_gemma_4_cpu_isq() {
    require_expensive_tests!();
    // Gemma 4 E2B is multimodal (Any-to-Any); use the vision pipeline.
    run_generation_smoke_test(cpu_isq_spec(
        "generate/gemma-4-cpu-isq",
        GEMMA_4_MODEL_ID,
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_gemma_4_cpu_uqff() {
    require_expensive_tests!();
    // Validates the UQFF loading path end-to-end. UQFF is the fast-load
    // alternative to ISQ — quantized weights are read directly from disk
    // instead of being computed from bf16 in memory. Useful when bf16 of
    // the full model wouldn't fit in the load-time footprint.
    run_generation_smoke_test(cpu_uqff_spec(
        "generate/gemma-4-cpu-uqff",
        GEMMA_4_UQFF_MODEL_ID,
        "q4k-0.uqff",
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_gemma_4_e4b_cpu_isq() {
    require_expensive_tests!();
    // Gemma 4 E4B (~8B params) via ISQ Q4K. Slower than UQFF because
    // weights load in bf16 then quantize; included for parity with the
    // E2B test pair.
    run_generation_smoke_test(cpu_isq_spec(
        "generate/gemma-4-e4b-cpu-isq",
        GEMMA_4_E4B_MODEL_ID,
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_gemma_4_e4b_cpu_uqff() {
    require_expensive_tests!();
    // Gemma 4 E4B via UQFF — the fast and small-RAM path for the larger
    // sibling of E2B. Total load footprint is ~11 GB (8.55 GB residual +
    // 2.27 GB q4k), which fits comfortably in CPU RAM but exceeds 8 GB
    // GPUs (and 16 GB once mistralrs's bf16-sized layer reservation is
    // added on top).
    run_generation_smoke_test(cpu_uqff_spec(
        "generate/gemma-4-e4b-cpu-uqff",
        GEMMA_4_E4B_UQFF_MODEL_ID,
        "q4k-0.uqff",
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_qwen_vl_cpu_isq() {
    require_expensive_tests!();
    // Vision pipeline; we send a text-only prompt. The vision service in
    // mistralrs.rs accepts messages without image blocks (see the
    // `images.is_empty()` branch in MistralRsVisionService::generate),
    // which is enough to confirm the multimodal builder loads end-to-end.
    run_generation_smoke_test(cpu_isq_spec(
        "generate/qwen-vl-cpu-isq",
        QWEN_VL_MODEL_ID,
        "vision",
    ))
    .await;
}

#[tokio::test]
#[ignore]
async fn test_glm_cpu_isq() {
    require_expensive_tests!();
    run_generation_smoke_test(cpu_isq_spec("generate/glm-cpu-isq", GLM_MODEL_ID, "text")).await;
}

// ---------------------------------------------------------------------------
// GPU tests (CUDA or Metal)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "gpu-cuda", feature = "gpu-metal"))]
mod gpu {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_gemma_3n_gpu() {
        require_expensive_tests!();
        // See note on the CPU sibling: Gemma 3n requires the vision pipeline.
        run_generation_smoke_test(gpu_spec(
            "generate/gemma-3n-gpu",
            GEMMA_3N_MODEL_ID,
            "vision",
        ))
        .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_gemma_4_gpu() {
        require_expensive_tests!();
        // **Hardware requirement: ~12 GB+ VRAM.** Gemma 4 E2B doesn't fit
        // on 8 GB even via UQFF: the unquantizable residual.safetensors
        // (vision/audio/video encoders + embeddings) is ~7 GB, and the
        // auto-device-mapper sizes layer reservation from bf16 estimates
        // (~10 GB) regardless of UQFF — so it places all 35 layers on
        // cuda[0] and the residual + reservation collide with available
        // VRAM. CPU + UQFF (`test_gemma_4_cpu_uqff`) is the small-VRAM
        // fallback. To make this fit on 8 GB we'd need manual layer
        // mapping (`DeviceMapMetadata::from_num_device_layers`), which
        // isn't currently re-exported by the `mistralrs` facade.
        run_generation_smoke_test(gpu_uqff_spec(
            "generate/gemma-4-gpu",
            GEMMA_4_UQFF_MODEL_ID,
            "q4k-0.uqff",
            "vision",
        ))
        .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_gemma_4_e4b_gpu() {
        require_expensive_tests!();
        // **Hardware requirement: ~24 GB+ VRAM.** Gemma 4 E4B residual
        // is ~8.55 GB and q4k UQFF adds another ~2.27 GB, so even
        // ignoring the planner's bf16-sized layer reservation the actual
        // GPU footprint at load is ~11 GB. With the bf16 layer
        // reservation added on top this needs ~24 GB.
        run_generation_smoke_test(gpu_uqff_spec(
            "generate/gemma-4-e4b-gpu",
            GEMMA_4_E4B_UQFF_MODEL_ID,
            "q4k-0.uqff",
            "vision",
        ))
        .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_qwen_vl_gpu() {
        require_expensive_tests!();
        run_generation_smoke_test(gpu_spec("generate/qwen-vl-gpu", QWEN_VL_MODEL_ID, "vision"))
            .await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_glm_gpu() {
        require_expensive_tests!();
        run_generation_smoke_test(gpu_spec("generate/glm-gpu", GLM_MODEL_ID, "text")).await;
    }
}
