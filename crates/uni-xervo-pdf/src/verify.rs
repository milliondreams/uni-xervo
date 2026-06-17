//! Cross-tier corroboration — diff generative output against a deterministic tier.
//!
//! The strongest guard against the failure that motivates the whole design — a
//! VLM silently altering a number — is not a confidence scalar but
//! *corroboration*: when a deterministic tier (native text or OCR) ran for the
//! same page, any number the generative tier emits that does not appear in the
//! deterministic text is suspect. [`numeric_divergence`] surfaces exactly those.
//!
//! Numbers are normalized before comparison so formatting differences do not
//! count as divergence: `"1,234"` and `"1234"` are equal, while `"1,234.56"`
//! and `"1,284.56"` are not. en-US digit grouping is assumed.

/// The result of comparing numbers across two texts.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericDivergence {
    /// Numbers present in the generative text but not the deterministic text.
    pub unmatched: Vec<f64>,
    /// Fraction of generative numbers that are unmatched, in `[0.0, 1.0]`.
    pub ratio: f32,
}

impl NumericDivergence {
    /// Whether every generative number was corroborated by the deterministic tier.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.unmatched.is_empty()
    }
}

/// Compare numbers in `generative` against those in `deterministic`.
///
/// Returns the generative numbers with no normalized match in the deterministic
/// text. An empty deterministic input (no lower tier ran) yields whatever the
/// generative text contains as unmatched — callers that lack a deterministic
/// tier should treat the page as *unverified* rather than failed.
#[must_use]
pub fn numeric_divergence(deterministic: &str, generative: &str) -> NumericDivergence {
    let det_nums = parse_numbers(deterministic);
    let gen_nums = parse_numbers(generative);
    let total = gen_nums.len();
    let unmatched: Vec<f64> = gen_nums
        .into_iter()
        .filter(|g| !det_nums.iter().any(|d| approx_eq(*d, *g)))
        .collect();
    let ratio = if total == 0 {
        0.0
    } else {
        unmatched.len() as f32 / total as f32
    };
    NumericDivergence { unmatched, ratio }
}

/// Relative-epsilon float comparison for normalized numbers.
fn approx_eq(a: f64, b: f64) -> bool {
    // Scale tolerance by magnitude so large currency figures still compare
    // exactly after normalization, while guarding tiny rounding noise.
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// Extract numeric tokens from `text`, normalized to `f64`.
///
/// Recognizes optional sign, ASCII digits, and `,` / `_` group separators with
/// an optional single decimal point. Tokens with more than one decimal point
/// (e.g. version strings like `1.2.3`) are skipped.
fn parse_numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let starts_number = c.is_ascii_digit()
            || ((c == '-' || c == '+')
                && i + 1 < bytes.len()
                && (bytes[i + 1] as char).is_ascii_digit());
        if starts_number {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let d = bytes[i] as char;
                if d.is_ascii_digit() || d == ',' || d == '.' || d == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            // Token chars are all ASCII, so `start..i` is a valid str boundary.
            if let Some(v) = normalize_number(&text[start..i]) {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Normalize a numeric token, stripping group separators.
fn normalize_number(tok: &str) -> Option<f64> {
    let cleaned: String = tok.chars().filter(|c| *c != ',' && *c != '_').collect();
    if cleaned.matches('.').count() > 1 {
        return None;
    }
    // A trailing dot ("12.") is a sentence boundary, not a decimal.
    let cleaned = cleaned.trim_end_matches('.');
    cleaned.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // R1: a VLM altering a number is flagged.
    #[test]
    fn altered_number_diverges() {
        let d = numeric_divergence("Total: $1,234.56", "Total: $1,284.56");
        assert!(!d.is_clean());
        assert_eq!(d.unmatched.len(), 1);
        assert!((d.unmatched[0] - 1284.56).abs() < 1e-6);
    }

    // Format-only differences must NOT be flagged.
    #[test]
    fn format_only_difference_is_clean() {
        let d = numeric_divergence("1,234 items", "1234 items");
        assert!(d.is_clean());
        assert_eq!(d.ratio, 0.0);
    }

    #[test]
    fn matching_numbers_are_clean() {
        let d = numeric_divergence("a 12 and 3.5 here", "result 3.5 then 12");
        assert!(d.is_clean());
    }

    #[test]
    fn no_numbers_is_clean() {
        let d = numeric_divergence("alpha beta", "gamma delta");
        assert!(d.is_clean());
        assert_eq!(d.ratio, 0.0);
    }

    #[test]
    fn one_of_several_altered() {
        let d = numeric_divergence("rows: 10, cols: 20", "rows: 10, cols: 21");
        assert_eq!(d.unmatched, vec![21.0]);
        assert!((d.ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn version_strings_are_ignored() {
        // "1.2.3" is not a number; only the real number diverges.
        let d = numeric_divergence("v1.2.3 build 42", "v1.2.3 build 99");
        assert_eq!(d.unmatched, vec![99.0]);
    }
}
