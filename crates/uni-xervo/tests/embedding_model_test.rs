//! Tests for embedding model operations

mod common;
use common::mock_support::{MockEmbeddingModel, runtime_with_embed};
use uni_xervo::traits::EmbeddingModel;
use uni_xervo::traits::ModelInfo;

#[tokio::test]
async fn test_single_text_embed() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string());
    let result = model.embed(&["hello world"]).await;

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.vectors.len(), 1);
    assert_eq!(embeddings.vectors[0].len(), 384);
    assert_eq!(embeddings.vectors[0][0], 0.1);
}

#[tokio::test]
async fn test_multiple_texts_embed() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string());
    let result = model.embed(&["hello", "world", "test"]).await;

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.vectors.len(), 3);
    for embedding in &embeddings.vectors {
        assert_eq!(embedding.len(), 384);
    }
}

#[tokio::test]
async fn test_empty_batch_embed() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string());
    let result = model.embed(&[]).await;

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.vectors.len(), 0);
}

#[tokio::test]
async fn test_large_batch_embed() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string());
    let texts: Vec<&str> = (0..100).map(|_| "test").collect();
    let result = model.embed(&texts).await;

    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.vectors.len(), 100);
}

#[tokio::test]
async fn test_dimensions_match() {
    let model = MockEmbeddingModel::new(768, "test-model".to_string());
    assert_eq!(model.dimensions(), 768);

    let result = model.embed(&["test"]).await.unwrap();
    assert_eq!(result.vectors[0].len(), 768);
}

#[tokio::test]
async fn test_model_id_correct() {
    let model = MockEmbeddingModel::new(384, "custom-model-id".to_string());
    assert_eq!(model.model_id(), "custom-model-id");
}

#[tokio::test]
async fn test_failure_propagation() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string()).with_failure(true);
    let result = model.embed(&["test"]).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Mock embedding failure")
    );
}

#[tokio::test]
async fn test_call_counting() {
    let model = MockEmbeddingModel::new(384, "test-model".to_string());
    assert_eq!(model.call_count(), 0);

    let _ = model.embed(&["test1"]).await;
    assert_eq!(model.call_count(), 1);

    let _ = model.embed(&["test2"]).await;
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn test_end_to_end_via_runtime() {
    let runtime = runtime_with_embed().await.unwrap();
    let model = runtime.embedding("embed/test").await.unwrap();

    let result = model.embed(&["hello", "world"]).await;
    assert!(result.is_ok());

    let embeddings = result.unwrap();
    assert_eq!(embeddings.vectors.len(), 2);
    assert_eq!(embeddings.vectors[0].len(), 384);
}
