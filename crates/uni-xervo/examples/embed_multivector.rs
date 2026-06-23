//! Multi-vector (ColBERT) embedding example: per-token vectors + MaxSim reranking.
//!
//! Builds a [`ModelRuntime`] with the `answerai-colbert-small` model, embeds a
//! query and two candidate documents into per-token vectors, then ranks the
//! documents with late-interaction MaxSim ([`uni_xervo::score::max_sim`]) — the
//! producer side of ColBERT-style retrieval, usable without any index.
//!
//! Model note: `answerdotai/answerai-colbert-small-v1` ships an in-graph
//! 384->96 projection (`vespa_colbert.onnx`), so xervo just strips padding and
//! L2-normalizes each token vector. The first run downloads ~130 MB.
//!
//! Run with:
//! ```sh
//! cargo run --example embed_multivector --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(feature = "provider-onnx")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-onnx")]
use uni_xervo::score::max_sim;

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the multi-vector embedder (preset supplies layout + dim 96).
    let spec = ModelAliasSpec {
        alias: "embed_mv/colbert".to_string(),
        task: ModelTask::EmbedMultiVector,
        provider_id: "local/onnx".to_string(),
        model_id: "answerdotai/answerai-colbert-small-v1".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    };

    // 2. Build the runtime with the local ONNX provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Embed a query and two candidate documents.
    let embedder = runtime.multi_vector_embedder("embed_mv/colbert").await?;
    let result = embedder
        .embed(&[
            "how do tides work",
            "tides are caused by the gravitational pull of the moon and sun",
            "sourdough bread relies on a fermented flour-and-water starter",
        ])
        .await?;

    let query = &result.vectors[0];
    let relevant = &result.vectors[1];
    let irrelevant = &result.vectors[2];

    println!("query tokens: {}", query.len());
    println!("dimensions: {}", embedder.dimensions());

    // 4. Rank documents by late-interaction MaxSim.
    let score_relevant = max_sim(query, relevant);
    let score_irrelevant = max_sim(query, irrelevant);
    println!("MaxSim(query, relevant doc)   = {score_relevant:.3}");
    println!("MaxSim(query, irrelevant doc) = {score_irrelevant:.3}");
    println!(
        "relevant ranked higher: {}",
        score_relevant > score_irrelevant
    );

    Ok(())
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --example embed_multivector --features provider-onnx"
    );
}
