//! Vision example: describe an image with a small vision-language model.
//!
//! Sends an image plus a text prompt to a VLM through the mistral.rs vision
//! pipeline and prints the generated description. Image description is not a
//! dedicated task — it is ordinary text generation (`runtime.generator`) with an
//! image attached to the message, so the same path handles captioning, visual
//! Q&A, and "extract X from this image".
//!
//! Model: `HuggingFaceTB/SmolVLM-256M-Instruct` (~0.3B, Apache-2.0) — the
//! smallest VLM the `local/mistralrs` provider can load (Idefics3 architecture,
//! auto-detected from the model's config). ~500 MB download on first run; small
//! enough to run on CPU. On a CPU-only machine, add `"force_cpu": true` and
//! `"dtype": "f32"` to the options below (bf16 is a GPU dtype).
//!
//! ⚠ This example demonstrates the correct API, but local vision *generation*
//! currently fails at inference due to an upstream mistral.rs bug (verified on
//! `mistralrs-core` 0.8.1): the model loads, then the Idefics3 forward pass
//! panics (`vision_models/idefics3/mod.rs` — `per_image` unwrap). Qwen2-VL fails
//! similarly via a different path. See mistral.rs issues #1068 / #1025 / #935.
//! The code below is correct and will work once upstream inference is fixed.
//!
//! Run with:
//! ```sh
//! cargo run --example vision_describe --features provider-mistralrs
//! ```

#[cfg(feature = "provider-mistralrs")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::traits::{ContentBlock, GenerationOptions, ImageInput, Message, MessageRole};

#[cfg(feature = "provider-mistralrs")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the vision model. `pipeline: "vision"` is required; the
    //    Idefics3/SmolVLM architecture is auto-detected from the model config.
    let spec = ModelAliasSpec {
        alias: "vision/smolvlm-256m".to_string(),
        task: ModelTask::Generate,
        provider_id: "local/mistralrs".to_string(),
        model_id: "HuggingFaceTB/SmolVLM-256M-Instruct".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "pipeline": "vision" }),
    };

    // 2. Build the runtime with the local mistral.rs provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample photo (replace with your own as needed).
    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/earthrise.jpg");
    let message = Message {
        role: MessageRole::User,
        content: vec![
            ContentBlock::Image(ImageInput::Bytes {
                data: std::fs::read(image_path)?,
                media_type: "image/jpeg".to_string(),
            }),
            ContentBlock::Text("Describe this image.".to_string()),
        ],
    };

    // 4. Generate. The description is returned in `result.text`.
    let vision = runtime.generator("vision/smolvlm-256m").await?;
    let result = vision
        .generate(&[message], GenerationOptions::default())
        .await?;
    println!("{}", result.text);

    Ok(())
}

#[cfg(not(feature = "provider-mistralrs"))]
fn main() {
    eprintln!(
        "This example requires the `provider-mistralrs` feature.\n\
         Run with: cargo run --example vision_describe --features provider-mistralrs"
    );
}
