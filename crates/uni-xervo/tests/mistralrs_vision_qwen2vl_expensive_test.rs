//! End-to-end (real model) test of vision generation via Qwen2-VL-2B on the
//! mistral.rs vision pipeline (image + prompt -> caption).
//!
//! Qwen2-VL uses a *different* loader (`qwen2vl`) than SmolVLM's Idefics3 path,
//! so this is a second, independent witness of the vision generation surface.
//!
//! # ⚠ Status: KNOWN-FAILING (upstream mistral.rs bug)
//!
//! The model **loads** correctly, but **inference fails inside mistral.rs**
//! (verified against `mistralrs-core` 0.8.1, CPU, after ~30 min of compute):
//!
//! ```text
//! model error: index-select invalid index 350 with dim size 350
//! ```
//!
//! An out-of-bounds index in the Qwen2-VL vision forward — a *different* upstream
//! defect from the Idefics3 one in `mistralrs_vision_describe_expensive_test.rs`.
//! Together they show vision *generation* inference is broken on this pinned
//! mistral.rs across two loaders (cf. issue #935 "couldn't run any vision model",
//! and Qwen2-VL issue #1108). Not caused by this repo.
//!
//! Kept as a deliberate living reproduction: `#[ignore]`'d **and**
//! `EXPENSIVE_TESTS`-gated, so it never runs in CI. ~4.5 GB download; CPU-forced
//! with ISQ Q4K (mirrors the repo's existing CPU generation spec). Do not weaken
//! the assertions to make it "pass" — it should pass unchanged once upstream is
//! fixed.

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

fn image_bytes(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/vision/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

#[tokio::test]
#[ignore]
async fn qwen2vl_2b_describes_image() {
    require_expensive_tests!();

    let spec = ModelAliasSpec {
        alias: "vision/qwen2vl-2b".to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: "Qwen/Qwen2-VL-2B-Instruct".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "pipeline": "vision",
            "force_cpu": true,
            "isq": "Q4K",
            "dtype": "f32"
        }),
    };

    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![spec])
        .build()
        .await
        .expect("build runtime");
    let vision = runtime
        .generator("vision/qwen2vl-2b")
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
