//! Encode a JSONL corpus + queries into ColBERT multi-vectors and dump them as a
//! compact binary, for the uni-db Phase-2 recall benchmark (rustic-ai/uni-db#96).
//!
//! This is the PRODUCER side: it runs a real late-interaction model through the
//! local-ONNX provider to turn real text into per-token vectors, so the uni-db
//! benchmark can measure native-index recall on *real* embeddings rather than
//! synthetic ones.
//!
//! Inputs (JSONL, one `{ "id": <int>, "text": <string> }` per line):
//!   $BENCH_DIR/corpus.jsonl, $BENCH_DIR/queries.jsonl
//! Outputs (binary: u32 magic 'MVEC', u32 dim, u32 n, then per item u32 n_tokens
//! followed by n_tokens*dim little-endian f32):
//!   $BENCH_DIR/docs.bin, $BENCH_DIR/queries.bin
//! (`BENCH_DIR`, not `OUT_DIR` — cargo reserves `OUT_DIR` for build scripts.)
//!
//! Run with (first run downloads the model from HF):
//! ```sh
//! BENCH_DIR=/tmp/colbert_bench cargo run --release \
//!   --example encode_corpus_multivec --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
mod imp {
    use std::fs::File;
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use std::sync::Arc;

    use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
    use uni_xervo::provider::LocalOnnxProvider;
    use uni_xervo::runtime::ModelRuntime;
    use uni_xervo::traits::MultiVectorEmbeddingModel;

    /// Reads the `text` field of every non-blank line in a `{id, text}` JSONL file.
    ///
    /// Order is preserved and **malformed lines are fatal**: the `MVEC` output is
    /// positional (item N corresponds to input line N), so silently dropping a
    /// line would shift every later vector and corrupt the downstream benchmark.
    /// Blank lines are skipped; any other parse failure panics with the line number.
    fn read_jsonl_texts(path: &str) -> Vec<String> {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut texts = Vec::new();
        for (idx, line) in BufReader::new(file).lines().enumerate() {
            let lineno = idx + 1;
            let line = line.unwrap_or_else(|e| panic!("{path}:{lineno}: read error: {e}"));
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("{path}:{lineno}: invalid JSON: {e}"));
            let text = value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("{path}:{lineno}: missing string `text` field"));
            texts.push(text.to_string());
        }
        texts
    }

    /// Embeds all texts in batches, returning ragged per-token vectors per text.
    async fn embed_all(
        embedder: &Arc<dyn MultiVectorEmbeddingModel>,
        texts: &[String],
    ) -> Result<Vec<Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(16) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let res = embedder.embed(&refs).await?;
            out.extend(res.vectors);
            eprintln!("  embedded {}/{}", out.len(), texts.len());
        }
        Ok(out)
    }

    /// Writes ragged multi-vectors as `MVEC` binary (see module docs).
    fn write_bin(path: &str, items: &[Vec<Vec<f32>>], dim: u32) {
        let file = File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
        let mut w = BufWriter::new(file);
        w.write_all(&0x4D56_4543_u32.to_le_bytes()).unwrap(); // 'MVEC'
        w.write_all(&dim.to_le_bytes()).unwrap();
        w.write_all(&(items.len() as u32).to_le_bytes()).unwrap();
        let mut total_tokens = 0usize;
        for item in items {
            w.write_all(&(item.len() as u32).to_le_bytes()).unwrap();
            total_tokens += item.len();
            for tok in item {
                assert_eq!(tok.len() as u32, dim, "token dim mismatch");
                for &x in tok {
                    w.write_all(&x.to_le_bytes()).unwrap();
                }
            }
        }
        w.flush().unwrap();
        eprintln!(
            "wrote {path}: {} items, {total_tokens} token vectors (dim {dim})",
            items.len()
        );
    }

    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
        // NB: not `OUT_DIR` — cargo reserves that env var for build scripts.
        let out_dir =
            std::env::var("BENCH_DIR").unwrap_or_else(|_| "/tmp/colbert_bench".to_string());
        let model_id = std::env::var("MODEL")
            .unwrap_or_else(|_| "answerdotai/answerai-colbert-small-v1".to_string());
        // Cap doc length to keep CPU encoding + storage tractable (scifact
        // abstracts are long); 192 tokens still covers most of an abstract.
        let max_seq_len: usize = std::env::var("MAX_SEQ_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(192);

        let spec = ModelAliasSpec {
            alias: "embed_mv/colbert".to_string(),
            task: ModelTask::EmbedMultiVector,
            provider_id: "local/onnx".to_string(),
            model_id,
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: true,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({ "max_seq_len": max_seq_len }),
        };
        let runtime = ModelRuntime::builder()
            .register_provider(LocalOnnxProvider::new())
            .catalog(vec![spec])
            .build()
            .await?;
        let embedder = runtime.multi_vector_embedder("embed_mv/colbert").await?;
        let dim = embedder.dimensions();
        eprintln!("model loaded, dim={dim}");

        let docs = read_jsonl_texts(&format!("{out_dir}/corpus.jsonl"));
        let queries = read_jsonl_texts(&format!("{out_dir}/queries.jsonl"));
        if docs.is_empty() || queries.is_empty() {
            return Err(format!(
                "no inputs: {} docs, {} queries under {out_dir} \
                 (expected corpus.jsonl and queries.jsonl — is BENCH_DIR correct?)",
                docs.len(),
                queries.len()
            )
            .into());
        }
        eprintln!(
            "encoding {} docs, {} queries ...",
            docs.len(),
            queries.len()
        );

        let doc_vecs = embed_all(&embedder, &docs).await?;
        let q_vecs = embed_all(&embedder, &queries).await?;

        write_bin(&format!("{out_dir}/docs.bin"), &doc_vecs, dim);
        write_bin(&format!("{out_dir}/queries.bin"), &q_vecs, dim);
        eprintln!("DONE");
        Ok(())
    }
}

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run().await
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --release --example encode_corpus_multivec --features provider-onnx"
    );
}
