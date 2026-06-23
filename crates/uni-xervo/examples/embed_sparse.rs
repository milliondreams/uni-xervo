//! Learned-sparse embedding example: turn text into term-weight vectors with ONNX.
//!
//! Builds a [`ModelRuntime`] with a SPLADE++ sparse embedder and prints the
//! number of non-zero terms produced for a sample query. Sparse embeddings
//! power hybrid (dense + lexical) first-stage retrieval and are
//! inverted-index compatible.
//!
//! Model note: `prithivida/Splade_PP_en_v1` ships an in-graph MLM head at
//! `onnx/model.onnx`, so xervo only applies `log(1 + ReLU(x))` + masked
//! max-pool — no separate projection weights. The first run downloads ~500 MB.
//!
//! Run with:
//! ```sh
//! cargo run --example embed_sparse --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(feature = "provider-onnx")]
use uni_xervo::runtime::ModelRuntime;

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the sparse embedder. The preset supplies the file layout,
    //    sparse method (mlm), and vocabulary size; options can override.
    let spec = ModelAliasSpec {
        alias: "embed_sparse/splade".to_string(),
        task: ModelTask::EmbedSparse,
        provider_id: "local/onnx".to_string(),
        model_id: "prithivida/Splade_PP_en_v1".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        // Keep only the 64 highest-weight terms per vector to bound index size.
        options: serde_json::json!({ "top_k": 64 }),
    };

    // 2. Build the runtime with the local ONNX provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Embed a couple of inputs. One sparse vector is returned per input.
    let embedder = runtime.sparse_embedder("embed_sparse/splade").await?;
    let result = embedder
        .embed(vec![
            "how do I make sourdough bread at home",
            "the capital of France is Paris",
        ])
        .await?;

    println!("vocab size: {}", embedder.vocab_size());
    for (i, vector) in result.vectors.iter().enumerate() {
        // Show the three highest-weight terms for a feel of the output.
        let mut top = vector.clone();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        top.truncate(3);
        println!(
            "input {i}: {} non-zero terms; top (term_id, weight): {:?}",
            vector.len(),
            top
        );
    }

    Ok(())
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --example embed_sparse --features provider-onnx"
    );
}
