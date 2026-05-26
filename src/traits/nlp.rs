//! Structured natural-language analysis types and traits.
//!
//! The trait is designed around multi-head cascade models (e.g.
//! `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`) that produce
//! token-level POS / NER / DEP tags, sentence segmentation, optional SRL
//! frames, and optional dialog-act classification in a single forward pass.

use crate::error::Result;
use async_trait::async_trait;
use bitflags::bitflags;
use std::any::Any;

bitflags! {
    /// Selects which NLP heads a caller wants populated.
    ///
    /// Implementations that support a head only do its post-processing
    /// (label decoding, dependency tree reconstruction, SRL frame
    /// assembly) when its flag is set — this saves measurable wall time
    /// on hot ingest paths where, for example, only NER is needed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NlpTasks: u32 {
        /// Part-of-speech tagging.
        const POS = 1 << 0;
        /// Named entity recognition.
        const NER = 1 << 1;
        /// Dependency parsing.
        const DEP = 1 << 2;
        /// Semantic role labeling.
        const SRL = 1 << 3;
        /// Sentence-level classification (e.g. dialog acts).
        const CLS = 1 << 4;
        /// Convenience: all currently-defined heads.
        const ALL = Self::POS.bits() | Self::NER.bits() | Self::DEP.bits()
                  | Self::SRL.bits() | Self::CLS.bits();
    }
}

/// One NLP analysis request: a text plus the heads to populate.
#[derive(Debug, Clone)]
pub struct NlpRequest<'a> {
    /// Input text. Borrowed so batch callers don't have to clone.
    pub text: &'a str,
    /// Which heads to populate. Unset flags result in empty fields on the
    /// corresponding [`NlpResult`] entries.
    pub tasks: NlpTasks,
}

/// Structured output of an NLP analysis call.
///
/// Field population depends on the [`NlpTasks`] flags in the request and on
/// the model's [`NlpModel::supported_tasks`]. Unrequested or unsupported
/// fields are empty (`Vec::new()`).
#[derive(Debug, Clone)]
pub struct NlpResult {
    /// Token sequence with optional POS / NER / DEP annotations.
    pub tokens: Vec<NlpToken>,
    /// Sentence boundaries (always populated when any head is requested).
    pub sentences: Vec<NlpSentence>,
    /// SRL frames. Empty unless [`NlpTasks::SRL`] was requested and supported.
    pub frames: Vec<SrlFrame>,
    /// Sentence-level classifications. Empty unless [`NlpTasks::CLS`] was
    /// requested and supported.
    pub speech_acts: Vec<SpeechAct>,
}

/// One token in an [`NlpResult`].
#[derive(Debug, Clone)]
pub struct NlpToken {
    /// Surface form of the token.
    pub text: String,
    /// UTF-8 byte offset of the token start in the original input text.
    pub start: usize,
    /// UTF-8 byte offset of the token end (exclusive).
    pub end: usize,
    /// POS tag (typically Universal Dependencies tagset). `None` if POS not requested.
    pub pos: Option<String>,
    /// Named-entity type (e.g. `"PERSON"`, `"ORG"`). `None` outside any entity span.
    pub ner: Option<String>,
    /// Dependency-parse head and relation label. `None` if DEP not requested.
    pub dep: Option<DepLink>,
}

/// One sentence in an [`NlpResult`].
#[derive(Debug, Clone)]
pub struct NlpSentence {
    /// Inclusive `[first, last]` token indices into [`NlpResult::tokens`].
    pub token_range: (usize, usize),
    /// UTF-8 byte offset of the sentence start in the original text.
    pub start: usize,
    /// UTF-8 byte offset of the sentence end (exclusive).
    pub end: usize,
}

/// A dependency-parse arc attached to an [`NlpToken`].
#[derive(Debug, Clone)]
pub struct DepLink {
    /// Token index of the syntactic head, within the same sentence's tokens.
    pub head: usize,
    /// Relation label (e.g. `"nsubj"`, `"obj"`, `"amod"`).
    pub relation: String,
}

/// One semantic-role-labeling frame.
#[derive(Debug, Clone)]
pub struct SrlFrame {
    /// Index into [`NlpResult::tokens`] of the predicate.
    pub predicate_token: usize,
    /// Predicate sense identifier when known (e.g. `"buy.01"`).
    pub predicate_sense: Option<String>,
    /// Argument spans.
    pub roles: Vec<SrlRole>,
}

/// One argument span of an [`SrlFrame`].
#[derive(Debug, Clone)]
pub struct SrlRole {
    /// Inclusive `[first, last]` token indices.
    pub span: (usize, usize),
    /// Role label (e.g. `"ARG0"`, `"ARG1"`, `"ARGM-TMP"`).
    pub label: String,
}

/// One sentence-level classification result.
#[derive(Debug, Clone)]
pub struct SpeechAct {
    /// Index into [`NlpResult::sentences`].
    pub sentence_index: usize,
    /// Class label (e.g. `"STATEMENT"`, `"QUESTION"`, `"GREETING"`).
    pub label: String,
    /// Provider-reported confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A multi-head structured-NLP model.
///
/// # Canonical default model
///
/// The reference implementation is
/// [`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall):
/// a DeBERTa-v3-xsmall multi-head cascade producing POS / NER / DEP / SRL / CLS
/// in a single forward pass. It will ship as the canonical `nlp/default`
/// catalog alias once the `local/onnx` provider gains an `NlpModel`
/// implementation in a follow-up release.
///
/// Tagsets used by the reference model:
/// - POS: Universal Dependencies English EWT (17 classes)
/// - NER: OntoNotes (18 entity types)
/// - DEP: Universal Dependencies English EWT
/// - SRL: PropBank (42 classes)
/// - CLS: 8 dialog acts (internal dataset)
///
/// Language: English only. Maximum input: 128 tokens — provider-side
/// chunking is the implementation's responsibility; the trait contract
/// keeps token offsets stable in whole-document UTF-8 byte coordinates
/// regardless of internal chunking.
#[async_trait]
pub trait NlpModel: Send + Sync + Any {
    /// Analyze a batch of texts, returning one [`NlpResult`] per request.
    ///
    /// # Errors
    /// Returns an error if a requested task is not in
    /// [`supported_tasks`](NlpModel::supported_tasks), or if the provider
    /// fails internally.
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>>;

    /// Heads this model can populate. Callers may request a subset via
    /// [`NlpRequest::tasks`].
    fn supported_tasks(&self) -> NlpTasks;

    /// The underlying model identifier.
    fn model_id(&self) -> &str;

    /// Optional warmup hook. The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn nlp_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn NlpModel>) {}
    }

    #[test]
    fn nlp_tasks_all_includes_every_head() {
        let all = NlpTasks::ALL;
        for flag in [
            NlpTasks::POS,
            NlpTasks::NER,
            NlpTasks::DEP,
            NlpTasks::SRL,
            NlpTasks::CLS,
        ] {
            assert!(all.contains(flag), "{flag:?} missing from NlpTasks::ALL");
        }
    }
}
