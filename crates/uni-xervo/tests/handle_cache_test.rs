//! Tests for per-alias typed handle caching.
//!
//! The handle cache sits above the model registry and ensures that repeated
//! calls to `embedding()`, `reranker()`, `generator()`, or `raw_tensor_model()`
//! for the same alias return a cached `Arc` without redoing spec lookup,
//! key hashing, or wrapper allocation.

use uni_xervo::api::{ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
mod common;
use common::mock_support::{MockProvider, make_spec};
use uni_xervo::runtime::ModelRuntime;

// ── Happy path: repeated calls return the same Arc ──────────────────────

#[tokio::test]
async fn handle_cache_returns_same_arc_on_repeated_embedding_calls() {
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let h1 = runtime.embedding("embed/test").await.unwrap();
    let h2 = runtime.embedding("embed/test").await.unwrap();

    assert!(
        std::sync::Arc::ptr_eq(&h1, &h2),
        "repeated embedding() calls must return the same Arc"
    );
}

#[tokio::test]
async fn handle_cache_returns_same_arc_on_repeated_reranker_calls() {
    let spec = make_spec(
        "rerank/test",
        ModelTask::Rerank,
        "mock/rerank",
        "test-model",
    );
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::rerank_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let h1 = runtime.reranker("rerank/test").await.unwrap();
    let h2 = runtime.reranker("rerank/test").await.unwrap();

    assert!(
        std::sync::Arc::ptr_eq(&h1, &h2),
        "repeated reranker() calls must return the same Arc"
    );
}

#[tokio::test]
async fn handle_cache_returns_same_arc_on_repeated_generator_calls() {
    let spec = make_spec(
        "generate/test",
        ModelTask::Generate,
        "mock/generate",
        "test-model",
    );
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::generate_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let h1 = runtime.generator("generate/test").await.unwrap();
    let h2 = runtime.generator("generate/test").await.unwrap();

    assert!(
        std::sync::Arc::ptr_eq(&h1, &h2),
        "repeated generator() calls must return the same Arc"
    );
}

#[tokio::test]
async fn handle_cache_returns_same_arc_on_repeated_raw_tensor_model_calls() {
    let spec = make_spec("nlp/test", ModelTask::Raw, "mock/raw", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::raw_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let h1 = runtime.raw_tensor_model("nlp/test").await.unwrap();
    let h2 = runtime.raw_tensor_model("nlp/test").await.unwrap();

    assert!(
        std::sync::Arc::ptr_eq(&h1, &h2),
        "repeated raw_tensor_model() calls must return the same Arc"
    );
}

#[tokio::test]
async fn handle_cache_distinct_aliases_get_distinct_handles() {
    let spec1 = make_spec("embed/a", ModelTask::Embed, "mock/embed", "model-a");
    let spec2 = make_spec("embed/b", ModelTask::Embed, "mock/embed", "model-b");

    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec1, spec2])
        .build()
        .await
        .unwrap();

    let ha = runtime.embedding("embed/a").await.unwrap();
    let hb = runtime.embedding("embed/b").await.unwrap();

    assert!(
        !std::sync::Arc::ptr_eq(&ha, &hb),
        "different aliases must produce different handles"
    );

    // Each alias should still be individually cached
    let ha2 = runtime.embedding("embed/a").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&ha, &ha2));
}

// ── Happy path: cached handles are functional ───────────────────────────

#[tokio::test]
async fn cached_embedding_handle_is_functional() {
    let spec = make_spec("embed/func", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    // First call populates cache
    let _ = runtime.embedding("embed/func").await.unwrap();
    // Second call returns cached handle — verify it actually works
    let cached = runtime.embedding("embed/func").await.unwrap();
    let result = cached.embed(&["hello world"]).await.unwrap();
    assert_eq!(result.vectors.len(), 1);
    assert_eq!(result.vectors[0].len(), 384);
}

#[tokio::test]
async fn cached_raw_tensor_model_handle_is_functional() {
    use uni_xervo::traits::TensorBatch;

    let spec = make_spec("nlp/func", ModelTask::Raw, "mock/raw", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::raw_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let _ = runtime.raw_tensor_model("nlp/func").await.unwrap();
    let cached = runtime.raw_tensor_model("nlp/func").await.unwrap();

    let inputs = TensorBatch::new();
    let output = cached.run(&inputs).await.unwrap();
    assert!(output.tensors.is_empty()); // mock echoes input
}

// ── Happy path: mixed types cached independently in one runtime ─────────

#[tokio::test]
async fn mixed_types_cached_independently() {
    let embed_spec = make_spec("embed/mix", ModelTask::Embed, "mock/embed", "test-model");
    let rerank_spec = make_spec("rerank/mix", ModelTask::Rerank, "mock/rerank", "test-model");
    let gen_spec = make_spec(
        "generate/mix",
        ModelTask::Generate,
        "mock/generate",
        "test-model",
    );
    let raw_spec = make_spec("nlp/mix", ModelTask::Raw, "mock/raw", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .register_provider(MockProvider::rerank_only())
        .register_provider(MockProvider::generate_only())
        .register_provider(MockProvider::raw_only())
        .catalog(vec![embed_spec, rerank_spec, gen_spec, raw_spec])
        .build()
        .await
        .unwrap();

    // First access populates each type's cache
    let e1 = runtime.embedding("embed/mix").await.unwrap();
    let r1 = runtime.reranker("rerank/mix").await.unwrap();
    let g1 = runtime.generator("generate/mix").await.unwrap();
    let o1 = runtime.raw_tensor_model("nlp/mix").await.unwrap();

    // Second access returns cached handles
    let e2 = runtime.embedding("embed/mix").await.unwrap();
    let r2 = runtime.reranker("rerank/mix").await.unwrap();
    let g2 = runtime.generator("generate/mix").await.unwrap();
    let o2 = runtime.raw_tensor_model("nlp/mix").await.unwrap();

    assert!(std::sync::Arc::ptr_eq(&e1, &e2));
    assert!(std::sync::Arc::ptr_eq(&r1, &r2));
    assert!(std::sync::Arc::ptr_eq(&g1, &g2));
    assert!(std::sync::Arc::ptr_eq(&o1, &o2));
}

// ── Happy path: prefetch primes registry, handle cache works on top ─────

#[tokio::test]
async fn prefetch_all_then_handle_cache_works() {
    let spec = make_spec("embed/pre", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    // Prefetch loads into registry (not handle cache)
    runtime.prefetch_all().await.unwrap();

    // First typed access should still populate handle cache from the registry
    let h1 = runtime.embedding("embed/pre").await.unwrap();
    let h2 = runtime.embedding("embed/pre").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn prefetch_specific_then_handle_cache_works() {
    let spec = make_spec("embed/pre2", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    runtime.prefetch(&["embed/pre2"]).await.unwrap();

    let h1 = runtime.embedding("embed/pre2").await.unwrap();
    let h2 = runtime.embedding("embed/pre2").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

// ── Happy path: eager warmup + handle cache ─────────────────────────────

#[tokio::test]
async fn eager_warmup_then_handle_cache_works() {
    let mut spec = make_spec("embed/eager", ModelTask::Embed, "mock/embed", "test-model");
    spec.warmup = WarmupPolicy::Eager;

    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    // Model was loaded during build; handle cache populates on first typed access
    let h1 = runtime.embedding("embed/eager").await.unwrap();
    let h2 = runtime.embedding("embed/eager").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

// ── Happy path: dynamic register() works with handle cache ──────────────

#[tokio::test]
async fn dynamically_registered_alias_works_with_handle_cache() {
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    // Register a new alias at runtime
    let spec = make_spec(
        "embed/dynamic",
        ModelTask::Embed,
        "mock/embed",
        "test-model",
    );
    runtime.register(spec).await.unwrap();

    let h1 = runtime.embedding("embed/dynamic").await.unwrap();
    let h2 = runtime.embedding("embed/dynamic").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

// ── Failure: unknown alias returns error, no cache pollution ────────────

#[tokio::test]
async fn unknown_alias_returns_error_not_cached() {
    let spec = make_spec("embed/exists", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let err = runtime
        .embedding("embed/nope")
        .await
        .err()
        .expect("expected error");
    assert!(
        matches!(err, RuntimeError::AliasNotFound { .. }),
        "expected AliasNotFound, got: {err}"
    );

    // Calling again should produce the same error (not a stale cached result)
    let err2 = runtime
        .embedding("embed/nope")
        .await
        .err()
        .expect("expected error");
    assert!(matches!(err2, RuntimeError::AliasNotFound { .. }));
}

#[tokio::test]
async fn unknown_alias_errors_for_all_handle_types() {
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![])
        .build()
        .await
        .unwrap();

    assert!(matches!(
        runtime.embedding("embed/nope").await,
        Err(RuntimeError::AliasNotFound { .. })
    ));
    assert!(matches!(
        runtime.reranker("rerank/nope").await,
        Err(RuntimeError::AliasNotFound { .. })
    ));
    assert!(matches!(
        runtime.generator("generate/nope").await,
        Err(RuntimeError::AliasNotFound { .. })
    ));
    assert!(matches!(
        runtime.raw_tensor_model("nlp/nope").await,
        Err(RuntimeError::AliasNotFound { .. })
    ));
}

// ── Failure: type mismatch returns error, no cache pollution ────────────

#[tokio::test]
async fn type_mismatch_embedding_on_raw_alias_returns_error() {
    let spec = make_spec("nlp/raw", ModelTask::Raw, "mock/raw", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::raw_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    // Asking for embedding() on a Raw alias should fail with CapabilityMismatch
    let err = runtime
        .embedding("nlp/raw")
        .await
        .err()
        .expect("expected error");
    assert!(
        matches!(err, RuntimeError::CapabilityMismatch(_)),
        "expected CapabilityMismatch, got: {err}"
    );

    // Should still fail on retry (error is not cached as success)
    let err2 = runtime
        .embedding("nlp/raw")
        .await
        .err()
        .expect("expected error");
    assert!(matches!(err2, RuntimeError::CapabilityMismatch(_)));

    // But the correct type should still work
    let h = runtime.raw_tensor_model("nlp/raw").await.unwrap();
    let h2 = runtime.raw_tensor_model("nlp/raw").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h, &h2));
}

#[tokio::test]
async fn type_mismatch_raw_tensor_model_on_embed_alias_returns_error() {
    let spec = make_spec("embed/test", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let err = runtime
        .raw_tensor_model("embed/test")
        .await
        .err()
        .expect("expected error");
    assert!(
        matches!(err, RuntimeError::ProviderCapabilityMissing { .. }),
        "expected ProviderCapabilityMissing, got: {err}"
    );

    // Correct type should work
    let h = runtime.embedding("embed/test").await.unwrap();
    let h2 = runtime.embedding("embed/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h, &h2));
}

// ── Failure: load failure doesn't pollute handle cache ──────────────────

#[tokio::test]
async fn load_failure_does_not_pollute_handle_cache() {
    let mut spec = make_spec("embed/fail", ModelTask::Embed, "mock/failing", "test-model");
    spec.warmup = WarmupPolicy::Lazy;

    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::failing())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let err = runtime
        .embedding("embed/fail")
        .await
        .err()
        .expect("expected error");
    assert!(
        matches!(err, RuntimeError::Load(_)),
        "expected Load error, got: {err}"
    );

    // Calling again should retry the load (not return a cached error)
    let err2 = runtime
        .embedding("embed/fail")
        .await
        .err()
        .expect("expected error");
    assert!(matches!(err2, RuntimeError::Load(_)));
}

// ── Concurrency ─────────────────────────────────────────────────────────

#[tokio::test]
async fn handle_cache_concurrent_first_access_all_succeed() {
    let spec = make_spec("embed/race", ModelTask::Embed, "mock/embed", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .catalog(vec![spec])
        .build()
        .await
        .unwrap();

    let mut tasks = Vec::new();
    for _ in 0..20 {
        let rt = runtime.clone();
        tasks.push(tokio::spawn(async move {
            rt.embedding("embed/race").await.unwrap()
        }));
    }

    let mut handles = Vec::with_capacity(tasks.len());
    for task in tasks {
        handles.push(task.await.unwrap());
    }

    // Singleflight: every concurrent first-caller must observe the same Arc
    // — wrapper allocation should happen exactly once per alias.
    let first = &handles[0];
    for (i, h) in handles.iter().enumerate().skip(1) {
        assert!(
            std::sync::Arc::ptr_eq(first, h),
            "task {i} returned a different Arc — wrapper allocation is not singleflight"
        );
    }

    // And subsequent calls keep returning that same Arc.
    let cached = runtime.embedding("embed/race").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(first, &cached));
}

#[tokio::test]
async fn handle_cache_concurrent_mixed_types() {
    let embed_spec = make_spec("embed/crace", ModelTask::Embed, "mock/embed", "test-model");
    let raw_spec = make_spec("nlp/crace", ModelTask::Raw, "mock/raw", "test-model");

    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::embed_only())
        .register_provider(MockProvider::raw_only())
        .catalog(vec![embed_spec, raw_spec])
        .build()
        .await
        .unwrap();

    enum Racer {
        Embed(std::sync::Arc<dyn uni_xervo::traits::EmbeddingModel>),
        Onnx(std::sync::Arc<dyn uni_xervo::traits::RawTensorModel>),
    }

    let mut tasks = Vec::new();
    for i in 0..20 {
        let rt = runtime.clone();
        if i % 2 == 0 {
            tasks.push(tokio::spawn(async move {
                Racer::Embed(rt.embedding("embed/crace").await.unwrap())
            }));
        } else {
            tasks.push(tokio::spawn(async move {
                Racer::Onnx(rt.raw_tensor_model("nlp/crace").await.unwrap())
            }));
        }
    }

    let mut embeds = Vec::new();
    let mut onnxs = Vec::new();
    for task in tasks {
        match task.await.unwrap() {
            Racer::Embed(h) => embeds.push(h),
            Racer::Onnx(h) => onnxs.push(h),
        }
    }

    // Singleflight per type: every racing task sees the same Arc within its type.
    let first_embed = &embeds[0];
    for (i, h) in embeds.iter().enumerate().skip(1) {
        assert!(
            std::sync::Arc::ptr_eq(first_embed, h),
            "embedding task {i} returned a different Arc"
        );
    }
    let first_onnx = &onnxs[0];
    for (i, h) in onnxs.iter().enumerate().skip(1) {
        assert!(
            std::sync::Arc::ptr_eq(first_onnx, h),
            "raw_tensor_model task {i} returned a different Arc"
        );
    }

    // Both caches have converged
    let e_again = runtime.embedding("embed/crace").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(first_embed, &e_again));

    let o_again = runtime.raw_tensor_model("nlp/crace").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(first_onnx, &o_again));
}
