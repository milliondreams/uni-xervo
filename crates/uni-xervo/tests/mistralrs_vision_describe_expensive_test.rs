//! End-to-end (real model) test of the vision generation path: image + prompt
//! -> textual description, via SmolVLM on the mistral.rs vision pipeline.
//!
//! Exercises image captioning / description through `GeneratorModel` with a
//! `ContentBlock::Image`, using `HuggingFaceTB/SmolVLM-256M-Instruct` (~0.3B,
//! Apache-2.0) — the smallest VLM `local/mistralrs` can load (Idefics3
//! architecture, auto-detected from the model config).
//!
//! # ⚠ Status: KNOWN-FAILING (upstream mistral.rs bug)
//!
//! The model **loads** correctly, but **inference panics inside mistral.rs**
//! (verified against `mistralrs-core` 0.8.1):
//!
//! ```text
//! vision_models/idefics3/mod.rs:230: called `Option::unwrap()` on a `None` value
//! ```
//!
//! The Idefics3 encoder-cache path sizes `per_image` by the post-filter image
//! *tile* count (`pixel_values.dim(0)`) but only fills it per *image hash*; when
//! image splitting yields more tiles than hashes, the tail stays `None` and the
//! `.unwrap()` panics. This is upstream and still present on `master` (it is not
//! caused by this repo's code, config, or model files; the smallest reproducer
//! is any split image). Tracked under mistral.rs issues #1068 / #1025 / #935.
//!
//! Kept as a deliberate living reproduction: it is `#[ignore]`'d **and**
//! `EXPENSIVE_TESTS`-gated, so it never runs in CI. When upstream fixes the
//! Idefics3 forward pass, this should pass unchanged — do not weaken the
//! assertions to make it "pass" in the meantime.
//!
//! Gated by both:
//! - `#![cfg(feature = "provider-mistralrs")]`
//! - `EXPENSIVE_TESTS=1` (downloads ~500 MB of weights from HuggingFace).
//!
//! Hardware: ~0.3B params; runs on CPU (forced here for reproducibility, with
//! `dtype: "f32"` since bf16 is a GPU dtype). Output is generative and
//! non-deterministic, so the assertion is a smoke check (non-empty caption),
//! not an exact-text match.
//!
//! Run with:
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test --features provider-mistralrs \
//!     --test mistralrs_vision_describe_expensive_test -- --ignored --nocapture
//! ```

#![cfg(feature = "provider-mistralrs")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{ContentBlock, GenerationOptions, ImageInput, Message, MessageRole};

fn should_run_expensive_tests() -> bool {
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

const MODEL_ID: &str = "HuggingFaceTB/SmolVLM-256M-Instruct";

fn smolvlm_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "vision/smolvlm-256m".to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        // `pipeline: "vision"` is required (the architecture is auto-detected).
        // CPU-forced for reproducibility; f32 because bf16 is a GPU dtype.
        options: serde_json::json!({
            "pipeline": "vision",
            "force_cpu": true,
            "dtype": "f32"
        }),
    }
}

fn image_bytes(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/vision/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

/// SmolVLM-256M loads and returns a non-empty description of an image.
#[tokio::test]
#[ignore]
async fn smolvlm_256m_describes_image() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![smolvlm_spec()])
        .build()
        .await
        .expect("build runtime");
    let vision = runtime
        .generator("vision/smolvlm-256m")
        .await
        .expect("load vision generator");

    let message = Message {
        role: MessageRole::User,
        content: vec![
            ContentBlock::Image(ImageInput::Bytes {
                data: image_bytes("earthrise.jpg"),
                media_type: "image/jpeg".to_string(),
            }),
            ContentBlock::Text("Describe this image.".to_string()),
        ],
    };
    let options = GenerationOptions {
        max_tokens: Some(64),
        temperature: Some(0.2),
        ..Default::default()
    };
    let result = vision
        .generate(&[message], options)
        .await
        .expect("generate");

    eprintln!("caption: {}", result.text);
    assert!(
        !result.text.trim().is_empty(),
        "expected a non-empty description"
    );
    let usage = result.usage.expect("usage reported");
    assert!(usage.total_tokens > 0, "expected total_tokens > 0");
}
