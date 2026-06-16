//! Deriving honest confidence for generative (VLM) output.
//!
//! The VLM emits no native confidence, so a [`Derived`](crate::Confidence::Derived)
//! score must be assembled from observable proxies: whether decoding finished
//! cleanly, how repetitive the output is (looping is the dominant VLM failure),
//! how far its length strays from a lower-tier reference, and how much it
//! diverges from the deterministic tier. The deterministic tiers do not use
//! this module — their confidence is [`Deterministic`](crate::Confidence::Deterministic)
//! or a [`Measured`](crate::Confidence::Measured) model probability.

use crate::result::{Confidence, DerivedSignals};
use std::collections::HashSet;

/// Multiplier applied when generation did not stop cleanly (truncation/looping).
///
/// Halving reflects that an unfinished generation is materially less
/// trustworthy without condemning it outright — the visible tokens may still be
/// correct up to the cut-off.
const UNCLEAN_FINISH_FACTOR: f32 = 0.5;

/// Multiplier applied when output length strays far from the lower-tier text.
///
/// Only triggers outside `[0.5, 2.0]`x; within that band, length variation is
/// normal (a VLM legitimately rewrites tables/markdown more verbosely).
const EXTREME_LENGTH_FACTOR: f32 = 0.7;

/// Fraction of repeated n-grams in `text`, in `[0.0, 1.0]`; higher means looping.
///
/// Uses `n`-word shingles. Texts shorter than `n` words (or `n == 0`) score
/// `0.0`. A score near `1.0` indicates the near-pathological repetition typical
/// of a VLM stuck in a loop.
#[must_use]
pub fn repetition_ratio(text: &str, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() - n + 1;
    let mut seen: HashSet<&[&str]> = HashSet::new();
    let mut unique = 0usize;
    for window in tokens.windows(n) {
        if seen.insert(window) {
            unique += 1;
        }
    }
    1.0 - (unique as f32 / total as f32)
}

/// Assemble a [`Derived`](Confidence::Derived) confidence from its signals.
///
/// The score starts at `1.0` and is multiplicatively penalized by each adverse
/// signal, then clamped to `[0.0, 1.0]`. The original `signals` are preserved on
/// the returned value so downstream can inspect *why* the score is what it is.
#[must_use]
pub fn from_signals(signals: DerivedSignals) -> Confidence {
    let mut score = 1.0f32;

    if !signals.finished_cleanly {
        score *= UNCLEAN_FINISH_FACTOR;
    }
    score *= 1.0 - signals.repetition_ratio.clamp(0.0, 1.0);
    if let Some(divergence) = signals.cross_tier_divergence {
        score *= 1.0 - divergence.clamp(0.0, 1.0);
    }
    if let Some(length_ratio) = signals.length_ratio
        && !(0.5..=2.0).contains(&length_ratio)
    {
        score *= EXTREME_LENGTH_FACTOR;
    }

    Confidence::Derived {
        score: score.clamp(0.0, 1.0),
        signals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signals() -> DerivedSignals {
        DerivedSignals {
            finished_cleanly: true,
            repetition_ratio: 0.0,
            length_ratio: None,
            cross_tier_divergence: None,
        }
    }

    #[test]
    fn repetition_high_for_looping_text() {
        assert!(repetition_ratio("the the the the the", 3) > 0.5);
    }

    #[test]
    fn repetition_zero_for_varied_text() {
        assert_eq!(repetition_ratio("a b c d e f g", 3), 0.0);
    }

    #[test]
    fn repetition_zero_for_short_text() {
        assert_eq!(repetition_ratio("a b", 3), 0.0);
    }

    #[test]
    fn clean_signals_score_high() {
        let c = from_signals(signals());
        assert!(c.score() > 0.99);
    }

    #[test]
    fn repetition_drives_score_down() {
        let c = from_signals(DerivedSignals {
            repetition_ratio: 0.9,
            ..signals()
        });
        assert!(c.score() < 0.2);
    }

    #[test]
    fn unclean_finish_halves_score() {
        let c = from_signals(DerivedSignals {
            finished_cleanly: false,
            ..signals()
        });
        assert!((c.score() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cross_tier_divergence_drives_score_down() {
        let c = from_signals(DerivedSignals {
            cross_tier_divergence: Some(0.8),
            ..signals()
        });
        assert!(c.score() < 0.25);
    }

    #[test]
    fn extreme_length_penalized() {
        let normal = from_signals(DerivedSignals {
            length_ratio: Some(1.2),
            ..signals()
        });
        let extreme = from_signals(DerivedSignals {
            length_ratio: Some(5.0),
            ..signals()
        });
        assert!(extreme.score() < normal.score());
    }
}
