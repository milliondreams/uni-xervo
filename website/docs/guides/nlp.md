# Structured NLP

`NlpModel` is Uni-Xervo's managed, multi-head structured-NLP task. One
`analyze` call runs a single transformer forward pass and returns
part-of-speech tags, named entities, a dependency parse, semantic-role frames,
and a sentence-level dialog-act classification — already decoded into typed
Rust structs, with provider-side tokenization, chunking, and label mapping
handled for you.

The canonical model is
[`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall)
(DeBERTa-v3-xsmall, five heads — POS / NER / DEP / SRL / dialog-act CLS — in one
forward pass, ~75 MB INT8 ONNX, English). It is served by `local/onnx`, the only
provider that ships `NlpModel` today.

## The five heads

| Head | `NlpTasks` flag | Where it lands in `NlpResult` |
| --- | --- | --- |
| Part-of-speech | `POS` | `tokens[i].pos` |
| Named-entity recognition | `NER` | `tokens[i].ner` (raw BIO) + `entities` (merged spans) |
| Dependency parse | `DEP` | `tokens[i].dep` |
| Semantic role labeling | `SRL` | `frames` |
| Dialog-act classification | `CLS` | `speech_acts` |

Callers request the subset they want via the `NlpTasks` bitflag; the provider
populates exactly those fields and leaves the rest empty.

## Quick example

```rust
use uni_xervo::traits::{NlpRequest, NlpTasks};

let model = runtime.nlp_model("nlp/default").await?;

let req = NlpRequest {
    text: "Marie Curie discovered radium in Paris.",
    tasks: NlpTasks::ALL,
};
let result = model.analyze(vec![req]).await?.remove(0);

// Merged named entities — no BIO bookkeeping required:
for e in &result.entities {
    println!("{} [{}] tokens {:?}", e.text, e.label, e.token_span);
}

// Dependency heads: `None` is the sentence root, `Some(i)` indexes `tokens`:
for tok in &result.tokens {
    match tok.dep.as_ref().and_then(|d| d.head) {
        None => println!("{:>12} ─▶ ROOT", tok.text),
        Some(h) => println!("{:>12} ─▶ {}", tok.text, result.tokens[h].text),
    }
}

// Gate on the full dialog-act distribution, self-described by the cls vocab:
if let (Some(act), Some(maps)) = (result.speech_acts.first(), model.label_maps()) {
    let informative = act
        .scores
        .iter()
        .zip(&maps.cls)
        .filter(|(_, label)| matches!(label.as_str(), "inform" | "status" | "request" | "offer"))
        .any(|(p, _)| *p >= 0.20);
    println!("informative: {informative}");
}

// Regroup subword tokens into words:
let words = result.tokens.last().map(|t| t.word_index + 1).unwrap_or(0);
println!("{} tokens across {} words", result.tokens.len(), words);
```

`analyze` takes a batch (`Vec<NlpRequest>`) and returns one `NlpResult` per
request, in order.

## Requesting heads

`NlpTasks` is a bitflag — combine the heads you need, or use `NlpTasks::ALL`:

```rust
pub struct NlpRequest<'a> {
    pub text: &'a str,
    pub tasks: NlpTasks,   // POS | NER | DEP | SRL | CLS, or ALL
}

// POS + NER only — DEP / SRL / CLS fields come back empty:
let req = NlpRequest {
    text: "Marie Curie discovered radium in Paris.",
    tasks: NlpTasks::POS | NlpTasks::NER,
};
```

Restricting the task set skips that head's post-processing, which saves wall
time on hot ingest paths where, for example, only NER is needed.

## Result types

`NlpModel::analyze` returns one `NlpResult` per request:

```rust
pub struct NlpResult {
    pub tokens: Vec<NlpToken>,        // per-token POS / NER / DEP, in input order
    pub sentences: Vec<NlpSentence>,  // boundaries with token-range indices
    pub frames: Vec<SrlFrame>,        // SRL frames (empty unless NlpTasks::SRL)
    pub speech_acts: Vec<SpeechAct>,  // dialog acts (empty unless NlpTasks::CLS)
    pub entities: Vec<NerEntity>,     // merged NER spans (empty unless NlpTasks::NER)
}
```

### `NlpToken`

```rust
pub struct NlpToken {
    pub text: String,
    pub start: usize,            // UTF-8 byte offset in the original text
    pub end: usize,              // UTF-8 byte offset (exclusive)
    pub pos: Option<String>,     // Universal Dependencies tag; None if POS not requested
    pub ner: Option<String>,     // raw BIO tag ("B-PERSON", "O", …); None if NER not requested
    pub dep: Option<DepLink>,    // dependency arc; None if DEP not requested
    pub word_index: usize,       // which word this subword belongs to
}
```

Tokens are subword units. `word_index` is **dense and monotonically
non-decreasing** over `tokens`: subwords of one word share an index, and each
new word increments it by one starting at `0`. Use it to regroup subwords into
words without parsing tokenizer metaspace markers.

### `DepLink`

```rust
pub struct DepLink {
    pub head: Option<usize>,   // Some(i): 0-based index into NlpResult::tokens; None: root
    pub relation: String,      // "nsubj", "obj", "amod", …
}
```

`head` is a **global** index into `NlpResult::tokens` (not sentence- or
chunk-local). `None` means the token attaches to the sentence root.

### `SrlFrame` and `SrlRole`

```rust
pub struct SrlFrame {
    pub predicate_token: usize,            // index into NlpResult::tokens
    pub predicate_sense: Option<String>,   // e.g. "buy.01" when known
    pub roles: Vec<SrlRole>,               // argument spans, excluding the predicate
}

pub struct SrlRole {
    pub span: (usize, usize),  // inclusive [first, last] token indices
    pub label: String,         // "ARG0", "ARG1", "ARGM-TMP", …
}
```

`span` endpoints are both inclusive; a single-token argument has `first == last`.
The predicate token is never itself included in `roles`. The provider runs one
forward pass per detected verb to fill every frame — this multi-pass
orchestration is invisible to callers.

### `SpeechAct`

```rust
pub struct SpeechAct {
    pub sentence_index: usize,
    pub label: String,       // == argmax of `scores`
    pub confidence: f32,     // == scores[argmax]
    pub scores: Vec<f32>,    // full per-class softmax, parallel to NlpLabelMaps::cls
}
```

`scores` is the complete softmax over the dialog-act vocabulary, so consumers
can do thresholded multi-label gating (admit a sentence if *any* relevant class
clears a probability floor) rather than relying only on the top-1 `label`.

### `NerEntity`

```rust
pub struct NerEntity {
    pub text: String,                // surface text spanning the original bytes
    pub label: String,               // entity type, BIO prefix stripped (e.g. "PERSON")
    pub token_span: (usize, usize),  // inclusive [first, last] token indices
    pub char_span: (usize, usize),   // half-open [start, end) UTF-8 byte offsets
}
```

`entities` is a BIO-collapsed view of the per-token `ner` tags: a `B-` (or
orphan `I-`) opens a span, matching `I-` tags extend it, and an `O` tag or a
label change closes it. The raw per-token `ner` tags remain available on each
`NlpToken`.

## Label vocabularies

The model decodes against fixed per-head vocabularies, exposed directly so you
do not have to embed a parallel copy of the model's `label_maps.json`:

```rust
pub struct NlpLabelMaps {
    pub pos: Vec<String>,
    pub ner: Vec<String>,
    pub deprel: Vec<String>,
    pub srl: Vec<String>,
    pub cls: Vec<String>,   // parallel to SpeechAct::scores
}

// Returns Some for models with a fixed label set (the ONNX cascade),
// None for models that don't expose vocabularies (remote / mock).
if let Some(maps) = model.label_maps() {
    // maps.cls[i] names the class whose probability is SpeechAct::scores[i]
}
```

Each vector is indexed by the model's internal class id, so `cls` lines up
index-for-index with `SpeechAct::scores`.

## Catalog entry

```json
{
  "alias": "nlp/default",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
  "warmup": "background",
  "options": {
    "onnx_path": "onnx/cascade.onnx",
    "max_seq_len": 128
  }
}
```

See the [`local/onnx` provider reference](../reference/providers/onnx.md) for the
full NLP option-key table (`onnx_path`, `tokenizer_path`, `label_maps_path`,
`max_seq_len`).

## Migration to 0.14.0

The 0.14.0 `NlpModel` surface is richer than 0.13.x. Two changes are breaking;
the rest is additive.

- **`DepLink.head` is now `Option<usize>`** (was `usize`). `Some(i)` is a 0-based
  global index into `NlpResult::tokens`; `None` is the sentence root. Match on
  `None` instead of inferring root from a sentinel value.
- **The result structs are `#[non_exhaustive]`.** Build them through `Default`
  (`NlpResult::default()` plus field assignment) rather than struct literals —
  this is what lets future heads be added without breaking callers. Reading
  existing fields is unaffected.
- **New, additive fields and methods:** `SpeechAct::scores` (full CLS
  distribution), `NlpResult::entities` (merged `NerEntity` spans),
  `NlpToken::word_index`, and `NlpModel::label_maps()`. Code that ignores them
  compiles unchanged.

## See also

- [`local/onnx` provider reference](../reference/providers/onnx.md) — NLP option
  keys and the canonical model details.
- [Multimodal trait surface](multimodal-traits.md) — the full set of traits
  introduced alongside `NlpModel`.
- [API reference (rustdoc)](../api/uni_xervo/index.html) — complete trait and
  type signatures.
