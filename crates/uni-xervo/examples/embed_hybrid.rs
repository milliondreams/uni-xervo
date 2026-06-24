//! Hybrid embedding example: dense + sparse + ColBERT from one forward pass.
//!
//! Builds a [`ModelRuntime`] with BGE-M3 (`aapot/bge-m3-onnx`) as a single
//! [`HybridEmbeddingModel`](uni_xervo::traits::HybridEmbeddingModel) and embeds
//! a batch once, materializing all three heads via [`HeadSet::ALL`]. This is the
//! whole point of BGE-M3: three retrieval signals — dense (cosine), learned
//! sparse (lexical), and ColBERT (late-interaction MaxSim) — from one weight
//! load and one pass, instead of three separate per-task sessions.
//!
//! Model note: `aapot/bge-m3-onnx` is a multi-output export
//! (`dense_vecs` / `sparse_vecs` / `colbert_vecs`). The first run downloads
//! ~2.1 GB of XLM-RoBERTa weights.
//!
//! Run with:
//! ```sh
//! cargo run --example embed_hybrid --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(feature = "provider-onnx")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-onnx")]
use uni_xervo::score::{max_sim, sparse_dot};
#[cfg(feature = "provider-onnx")]
use uni_xervo::traits::HeadSet;

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the hybrid embedder (the `BGEM3Hybrid` preset declares all
    //    three heads of the multi-output graph).
    let spec = ModelAliasSpec {
        alias: "embed_hybrid/bgem3".to_string(),
        task: ModelTask::EmbedHybrid,
        provider_id: "local/onnx".to_string(),
        model_id: "BGEM3Hybrid".to_string(),
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

    // 3. One forward pass, all three heads.
    let embedder = runtime.hybrid_embedder("embed_hybrid/bgem3").await?;
    println!("available heads: {:?}", embedder.available_heads());

    let query = "how do ocean tides work";
    let relevant = "tides are caused by the gravitational pull of the moon and sun";
    let irrelevant = "sourdough bread relies on a fermented flour-and-water starter";

    let result = embedder
        .embed(&[query, relevant, irrelevant], HeadSet::ALL)
        .await?;

    let dense = result.dense.expect("dense head requested and available");
    let sparse = result.sparse.expect("sparse head requested and available");
    let colbert = result
        .multi_vector
        .expect("ColBERT head requested and available");

    // 4. Score the same candidates three ways from the single pass.
    println!("\n-- dense (cosine) --");
    println!("relevant   = {:.3}", cosine(&dense[0], &dense[1]));
    println!("irrelevant = {:.3}", cosine(&dense[0], &dense[2]));

    println!("\n-- sparse (lexical dot) --");
    println!("relevant   = {:.3}", sparse_dot(&sparse[0], &sparse[1]));
    println!("irrelevant = {:.3}", sparse_dot(&sparse[0], &sparse[2]));

    println!("\n-- ColBERT (MaxSim) --");
    println!("relevant   = {:.3}", max_sim(&colbert[0], &colbert[1]));
    println!("irrelevant = {:.3}", max_sim(&colbert[0], &colbert[2]));

    Ok(())
}

/// Cosine similarity for already-L2-normalized dense vectors (just a dot product).
#[cfg(feature = "provider-onnx")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --example embed_hybrid --features provider-onnx"
    );
}
