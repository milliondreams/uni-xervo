//! Image-embedding example: turn an image into a dense vector with ONNX.
//!
//! Builds a [`ModelRuntime`] with a ViT-style image embedder (SigLIP-2 vision
//! tower) and prints the embedding dimension for the bundled sample image.
//! Image embeddings power visual search and multimodal retrieval.
//!
//! Model note: `onnx-community/siglip2-so400m-patch16-384-ONNX` is a real,
//! existing repo, but its top-level `onnx/model.onnx` is the full vision+text
//! model (which also needs `input_ids`). The image embedder feeds only
//! `pixel_values`, so this example points at the vision tower
//! (`onnx/vision_model.onnx`). The exact `output_name`/`pool` that yield a clean
//! per-image vector depend on the export's output tensors and may need tuning —
//! verify against the model card before relying on the numbers. The download is
//! large (the vision tower is ~1.7 GB).
//!
//! Run with:
//! ```sh
//! cargo run --example image_embed --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(feature = "provider-onnx")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-onnx")]
use uni_xervo::traits::ImageInput;

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the image embedder (SigLIP-2 So400m, 384px input, dim 1152).
    let spec = ModelAliasSpec {
        alias: "embed/siglip".to_string(),
        task: ModelTask::EmbedImage,
        provider_id: "local/onnx".to_string(),
        model_id: "onnx-community/siglip2-so400m-patch16-384-ONNX".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "onnx_path": "onnx/vision_model.onnx",
            "image_size": 384,
            "dimensions": 1152,
            "normalization": "siglip",
            "pool": "none",
            "normalize": true
        }),
    };

    // 2. Build the runtime with the local ONNX provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample image (replace with your own as needed).
    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/text_lines.png");
    let input = ImageInput::Bytes {
        data: std::fs::read(image_path)?,
        media_type: "image/png".to_string(),
    };

    // 4. Embed. One vector is returned per input image.
    let embedder = runtime.image_embedder("embed/siglip").await?;
    let result = embedder.embed(vec![input]).await?;

    println!("embedded {} image(s)", result.vectors.len());
    println!("vector dimension: {}", result.vectors[0].len());

    Ok(())
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --example image_embed --features provider-onnx"
    );
}
