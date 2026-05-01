//! Real-inference tests that load specific embedding and reranking model
//! families through `LocalOnnxProvider` and exercise the public
//! `EmbeddingModel` / `RerankerModel` API end-to-end.
//!
//! Coverage today: the BGE family (BGE-small-en-v1.5, BGE-large-en-v1.5,
//! BGE-reranker-base) — encoder-style models that fit the current
//! `provider-onnx` contract (CLS-pooled embedders, single-logit
//! cross-encoder rerankers).
//!
//! Tests for decoder-style embedders (Qwen3-Embedding family — needs
//! last-token pooling) and generative rerankers (Qwen3-Reranker family —
//! needs the new `style: "generative"` rerank path) live alongside the
//! BGE tests below once those provider capabilities land.
//!
//! All tests are gated by both:
//!
//! - `#![cfg(feature = "provider-onnx")]` — only built when the provider is
//!   enabled.
//! - `EXPENSIVE_TESTS=1` env var (via `require_expensive_tests!`) — each
//!   test downloads multi-MB weights from HuggingFace and runs real
//!   inference, so they are skipped unless explicitly opted in.
//!
//! Run with:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 cargo nextest run \
//!     --features provider-onnx \
//!     --test onnx_models_expensive_test \
//!     --run-ignored all
//! ```
//!
//! NOTE: model identifiers below target stable, public HuggingFace repos.
//! They may need updating when new minor revisions ship — these are HF
//! repo strings, not URLs, and the tests are `#[ignore]`'d so a stale ID
//! will not break CI.

#![cfg(feature = "provider-onnx")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

// ---------------------------------------------------------------------------
// Test gating
// ---------------------------------------------------------------------------

fn should_run_expensive_tests() -> bool {
    env::var("EXPENSIVE_TESTS").is_ok()
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
// ---------------------------------------------------------------------------

const BGE_SMALL_PRESET_ALIAS: &str = "BGESmallENV15";
const BGE_SMALL_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
const BGE_LARGE_PRESET_ALIAS: &str = "BGELargeENV15";
const BGE_RERANKER_BASE_MODEL_ID: &str = "BAAI/bge-reranker-base";
const QWEN3_EMBED_PRESET_ALIAS: &str = "Qwen3Embedding06B";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn embed_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Embed,
        provider_id: "local/onnx".to_string(),
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

fn rerank_spec(alias: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

/// Tolerance for the L2-norm check on a normalized embedding. ORT inference
/// produces small floating-point drift versus a perfect unit vector; 1e-2
/// is loose enough for both fp32 and quantized weights without being so
/// loose that an un-normalized output would slip through.
const L2_NORM_TOL: f32 = 1e-2;

fn assert_unit_norm(vec: &[f32], label: &str) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < L2_NORM_TOL,
        "{label}: expected L2-normalized embedding (norm ≈ 1.0), got norm={norm}"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine over equal-length vectors");
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

// ---------------------------------------------------------------------------
// Embedding tests
// ---------------------------------------------------------------------------

/// Loads BGE-small-en-v1.5 via the built-in preset alias `BGESmallENV15`.
/// Verifies the preset path: the spec carries no options, so all defaults
/// (CLS pooling, 384-dim, L2-normalize, 512 max_seq_len) come from the
/// preset table.
#[tokio::test]
#[ignore]
async fn test_bge_small_embedding_via_preset() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-small-preset",
            BGE_SMALL_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-small-preset")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 384);

    let texts = vec!["Hello world", "Rust is amazing", "Machine learning"];
    let embeddings = model.embed(texts).await.expect("embed call failed");

    assert_eq!(embeddings.len(), 3);
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "row {i}: BGE-small is 384-dim");
        assert_unit_norm(emb, &format!("row {i}"));
    }
}

/// Loads BGE-small-en-v1.5 using the raw HF repo id `BAAI/bge-small-en-v1.5`
/// — exercises the same preset table via a different alias path. Confirms
/// that users can refer to the model by its canonical HF id and still
/// inherit the preset defaults without specifying options.
#[tokio::test]
#[ignore]
async fn test_bge_small_embedding_via_model_id() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-small-by-id",
            BGE_SMALL_MODEL_ID,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-small-by-id")
        .await
        .expect("resolve embedding model");

    let embeddings = model
        .embed(vec!["one", "two"])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 384);
    assert_unit_norm(&embeddings[0], "row 0");
    assert_unit_norm(&embeddings[1], "row 1");
}

/// Loads BGE-large-en-v1.5 via preset alias. Verifies the larger 1024-dim
/// variant downloads, loads, and produces correctly-shaped output. This is
/// the heaviest embedder in the BGE family currently presetted (~1.3GB
/// weights), so it doubles as a download-path stress test.
#[tokio::test]
#[ignore]
async fn test_bge_large_embedding() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-large",
            BGE_LARGE_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-large")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 1024);

    let embeddings = model
        .embed(vec!["a sample sentence to embed"])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 1024);
    assert_unit_norm(&embeddings[0], "single-input");
}

/// Verifies that BGE-small produces semantically-meaningful embeddings:
/// related sentences should have higher cosine similarity than unrelated
/// pairs. This is a sanity check, not a quality benchmark — the gap on
/// these obvious examples is large enough that any seriously broken
/// model (wrong pooling, missing normalization, swapped tensors) fails.
#[tokio::test]
#[ignore]
async fn test_bge_embedding_semantic_similarity() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/bge-sim",
            BGE_SMALL_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/bge-sim")
        .await
        .expect("resolve embedding model");

    // Two related dog sentences and one unrelated cooking sentence.
    let embeddings = model
        .embed(vec![
            "The dog chased the ball across the yard.",
            "A puppy ran after a tennis ball in the park.",
            "Boil pasta in salted water for nine minutes.",
        ])
        .await
        .expect("embed call failed");

    let related_sim = cosine(&embeddings[0], &embeddings[1]);
    let unrelated_sim = cosine(&embeddings[0], &embeddings[2]);

    assert!(
        related_sim > unrelated_sim + 0.10,
        "expected related-pair cosine ({related_sim:.3}) to exceed \
         unrelated-pair cosine ({unrelated_sim:.3}) by a clear margin"
    );
}

/// Loads Qwen3-Embedding-0.6B via the `Qwen3Embedding06B` preset alias.
/// Exercises the decoder-style embedder path: external-data download
/// (`onnx/model.onnx_data` sidecar), last-token pooling, 1024-dim output.
/// This is the heaviest embedder in the test suite (~600 MB ONNX +
/// external data), so it doubles as a stress test for the
/// `additional_files` download loop.
#[tokio::test]
#[ignore]
async fn test_qwen3_embedding_06b() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![embed_spec(
            "embed/qwen3-06b",
            QWEN3_EMBED_PRESET_ALIAS,
            serde_json::Value::Null,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let model = runtime
        .embedding("embed/qwen3-06b")
        .await
        .expect("resolve embedding model");

    assert_eq!(model.dimensions(), 1024);

    let embeddings = model
        .embed(vec![
            "The dog chased the ball across the yard.",
            "A puppy ran after a tennis ball in the park.",
            "Boil pasta in salted water for nine minutes.",
        ])
        .await
        .expect("embed call failed");

    assert_eq!(embeddings.len(), 3);
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 1024, "row {i}: Qwen3-Embedding is 1024-dim");
        assert_unit_norm(emb, &format!("row {i}"));
    }

    // Last-token pooling on a decoder-only model should still capture
    // semantic similarity. Same sanity check as the BGE variant.
    let related_sim = cosine(&embeddings[0], &embeddings[1]);
    let unrelated_sim = cosine(&embeddings[0], &embeddings[2]);
    assert!(
        related_sim > unrelated_sim + 0.05,
        "expected related-pair cosine ({related_sim:.3}) to exceed \
         unrelated-pair cosine ({unrelated_sim:.3}) by a clear margin"
    );
}

// ---------------------------------------------------------------------------
// Reranker tests
// ---------------------------------------------------------------------------

/// Loads BGE-reranker-base and verifies it ranks an obviously-relevant
/// document above three obviously-irrelevant ones. Exercises the full
/// cross-encoder path: text-pair tokenization, batched ONNX inference,
/// single-logit output extraction, descending-score sort.
#[tokio::test]
#[ignore]
async fn test_bge_reranker_base_orders_relevant_first() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![rerank_spec(
            "rerank/bge-base",
            BGE_RERANKER_BASE_MODEL_ID,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let reranker = runtime
        .reranker("rerank/bge-base")
        .await
        .expect("resolve reranker");

    let relevant = "Giant pandas live in the bamboo forests of central China.";
    let docs = vec![
        "The Eiffel Tower is in Paris, France.",
        relevant,
        "Quantum entanglement was first described in the 1930s.",
        "The Pacific Ocean is the largest body of water on Earth.",
    ];

    let scored = reranker
        .rerank("Where do giant pandas live?", &docs)
        .await
        .expect("rerank call failed");

    assert_eq!(scored.len(), docs.len());

    // Scores must be monotonically descending — the provider sorts by score.
    for w in scored.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores not in descending order: {} then {}",
            w[0].score,
            w[1].score
        );
    }

    let top = &scored[0];
    assert_eq!(
        docs[top.index], relevant,
        "expected the panda doc to rank first, got idx={} ({:?})",
        top.index, docs[top.index]
    );
}

/// Sanity check that ranking is stable when the input doc order is
/// permuted. Catches batch-padding / index-tracking bugs (e.g. forgetting
/// to translate an internal sorted index back to the caller's input
/// position) by re-running with a different doc order and confirming the
/// same document still wins.
#[tokio::test]
#[ignore]
async fn test_bge_reranker_score_ordering_stable_across_permutation() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![rerank_spec(
            "rerank/bge-stable",
            BGE_RERANKER_BASE_MODEL_ID,
        )])
        .build()
        .await
        .expect("runtime build failed");

    let reranker = runtime
        .reranker("rerank/bge-stable")
        .await
        .expect("resolve reranker");

    let query = "What is the capital of France?";
    let relevant = "Paris is the capital and largest city of France.";

    let order_a = vec![
        "Tokyo is the capital of Japan.",
        relevant,
        "The Amazon is the largest rainforest on Earth.",
        "Mount Everest is the tallest mountain in the world.",
    ];
    let order_b = vec![
        "Mount Everest is the tallest mountain in the world.",
        "The Amazon is the largest rainforest on Earth.",
        "Tokyo is the capital of Japan.",
        relevant,
    ];

    let a = reranker.rerank(query, &order_a).await.expect("rerank A");
    let b = reranker.rerank(query, &order_b).await.expect("rerank B");

    assert_eq!(order_a[a[0].index], relevant, "order A: relevant doc first");
    assert_eq!(order_b[b[0].index], relevant, "order B: relevant doc first");
}
