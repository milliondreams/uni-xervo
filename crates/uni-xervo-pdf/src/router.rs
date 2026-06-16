//! Per-page tier routing: pre-detection planning and failure-driven escalation.
//!
//! Two pure functions, deliberately model-free so the cost-proportionality and
//! reliability guarantees are unit-testable without inference or a GPU:
//!
//! - [`plan_tier`] chooses a page's starting tier from cheap signals, honoring
//!   the policy's bounds (including the `Ceiling` reliability knob).
//! - [`escalate_on_failure`] is the ParseFixer-style climb applied *after* a
//!   tier runs, when its output is empty or flatter than expected.
//!
//! Runtime caps (budget, absent GPU) are not handled here — the pipeline
//! narrows the effective policy before calling in, so this layer stays pure.

use crate::config::{DocExtractPolicy, LevelPolicy, OutputWant};
use crate::result::{Escalation, EscalationReason};
use crate::signals::PageSignals;
use crate::tier::Tier;

/// The router's pre-run decision for a page.
#[derive(Debug, Clone, PartialEq)]
pub struct TierPlan {
    /// The tier to run for this page.
    pub target: Tier,
    /// Why the router climbed above [`Native`](Tier::Native), in climb order.
    pub escalations: Vec<Escalation>,
}

/// What a tier's output looked like, for failure-driven escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TierObservation {
    /// The tier produced no usable content.
    pub empty: bool,
    /// The tier produced flat text where layout structure was expected.
    pub flat: bool,
}

/// Choose a page's starting tier from cheap signals, bounded by `policy`.
///
/// A [`Fixed`](LevelPolicy::Fixed) policy pins the tier and ignores signals.
/// Otherwise the desired tier is raised by `Native -> Ocr` triggers (no/garbled/
/// sparse text) and the `Ocr -> Vlm` trigger (structure wanted on a structured
/// page), then forced up to the policy `min`, then clamped to the policy `max`
/// / ceiling. Climbs above the clamped target are dropped, which is exactly how
/// `Ceiling(Ocr)` forbids the generative tier.
#[must_use]
pub fn plan_tier(signals: PageSignals, policy: &DocExtractPolicy) -> TierPlan {
    if let LevelPolicy::Fixed(t) = policy.level {
        return TierPlan {
            target: t,
            escalations: Vec::new(),
        };
    }

    let mut escalations = Vec::new();
    let mut desired = Tier::Native;

    // Native -> Ocr: the text layer is missing, garbled, or too sparse.
    if signals.needs_ocr() {
        desired = Tier::Ocr;
        escalations.push(Escalation {
            from: Tier::Native,
            to: Tier::Ocr,
            reason: signals.ocr_reason(),
        });
    }

    // Ocr -> Vlm: structured output wanted on a structurally complex page.
    if policy.want == OutputWant::Structure && signals.looks_structured {
        // A structured page needs the image tiers; ensure Ocr is on the path.
        // (`desired` is unconditionally raised to Vlm below, so we only need to
        // record the intervening Native->Ocr climb here.)
        if desired < Tier::Ocr {
            escalations.push(Escalation {
                from: Tier::Native,
                to: Tier::Ocr,
                reason: EscalationReason::StructureRequested,
            });
        }
        desired = Tier::Vlm;
        escalations.push(Escalation {
            from: Tier::Ocr,
            to: Tier::Vlm,
            reason: EscalationReason::StructureRequested,
        });
    }

    // An Auto `min` forces at least that tier (a policy choice, not a signal).
    desired = desired.max(policy.level.min_tier());

    // Clamp to the policy ceiling/max; drop any climbs above the result.
    let target = policy.level.clamp(desired);
    escalations.retain(|e| e.to <= target);

    TierPlan {
        target,
        escalations,
    }
}

/// Decide whether to climb after observing `current`'s output.
///
/// Returns the single next-step [`Escalation`] when the policy permits it, or
/// `None` when the output was adequate or no higher tier is allowed. A
/// [`Fixed`](LevelPolicy::Fixed) policy never escalates.
#[must_use]
pub fn escalate_on_failure(
    current: Tier,
    observation: TierObservation,
    policy: &DocExtractPolicy,
) -> Option<Escalation> {
    if matches!(policy.level, LevelPolicy::Fixed(_)) {
        return None;
    }

    let step = |from: Tier, to: Tier, reason: EscalationReason| -> Option<Escalation> {
        policy
            .level
            .allows(to)
            .then_some(Escalation { from, to, reason })
    };

    match current {
        // Native produced nothing -> try reading the pixels.
        Tier::Native if observation.empty => {
            step(Tier::Native, Tier::Ocr, EscalationReason::EmptyOutput)
        }
        // OCR produced nothing -> let the VLM attempt the page.
        Tier::Ocr if observation.empty => step(Tier::Ocr, Tier::Vlm, EscalationReason::EmptyOutput),
        // OCR gave flat text where structure was expected -> escalate for layout.
        Tier::Ocr if policy.want == OutputWant::Structure && observation.flat => step(
            Tier::Ocr,
            Tier::Vlm,
            EscalationReason::FlatWhereStructureExpected,
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DocExtractPolicy;

    /// A born-digital page: text present, well covered, not garbled.
    fn born_digital() -> PageSignals {
        PageSignals {
            has_text_layer: true,
            text_coverage: 0.5,
            garbled: false,
            looks_structured: false,
        }
    }

    /// A scanned page: no text layer.
    fn scanned() -> PageSignals {
        PageSignals {
            has_text_layer: false,
            text_coverage: 0.0,
            garbled: false,
            looks_structured: false,
        }
    }

    // C1: born-digital + want:Text -> stays Native, no climbs.
    #[test]
    fn c1_born_digital_stays_native() {
        let plan = plan_tier(born_digital(), &DocExtractPolicy::auto_up_to(Tier::Vlm));
        assert_eq!(plan.target, Tier::Native);
        assert!(plan.escalations.is_empty());
    }

    // C2: no text layer -> Native->Ocr.
    #[test]
    fn c2_no_text_layer_escalates_to_ocr() {
        let plan = plan_tier(scanned(), &DocExtractPolicy::auto_up_to(Tier::Vlm));
        assert_eq!(plan.target, Tier::Ocr);
        assert_eq!(plan.escalations.len(), 1);
        assert_eq!(plan.escalations[0].reason, EscalationReason::NoTextLayer);
    }

    // C3: low coverage -> Ocr.
    #[test]
    fn c3_low_coverage_escalates_to_ocr() {
        let signals = PageSignals {
            text_coverage: 0.01,
            ..born_digital()
        };
        let plan = plan_tier(signals, &DocExtractPolicy::auto_up_to(Tier::Vlm));
        assert_eq!(plan.target, Tier::Ocr);
        assert_eq!(
            plan.escalations[0].reason,
            EscalationReason::LowTextCoverage
        );
    }

    // C4: garbled -> Ocr.
    #[test]
    fn c4_garbled_escalates_to_ocr() {
        let signals = PageSignals {
            garbled: true,
            ..born_digital()
        };
        let plan = plan_tier(signals, &DocExtractPolicy::auto_up_to(Tier::Vlm));
        assert_eq!(plan.target, Tier::Ocr);
        assert_eq!(plan.escalations[0].reason, EscalationReason::GarbledText);
    }

    // C5: structured page but want:Text -> does NOT escalate to Vlm.
    #[test]
    fn c5_structure_ignored_when_only_text_wanted() {
        let signals = born_digital().with_structured(true);
        let plan = plan_tier(signals, &DocExtractPolicy::auto_up_to(Tier::Vlm));
        assert_eq!(plan.target, Tier::Native);
        assert!(plan.escalations.is_empty());
    }

    // C6: structured page + want:Structure -> climbs through Ocr to Vlm.
    #[test]
    fn c6_structure_wanted_climbs_to_vlm() {
        let signals = born_digital().with_structured(true);
        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;
        let plan = plan_tier(signals, &policy);
        assert_eq!(plan.target, Tier::Vlm);
        assert_eq!(plan.escalations.len(), 2);
        assert_eq!(plan.escalations[1].to, Tier::Vlm);
    }

    // C8: Auto{min:Ocr} forces at least Ocr on a plain page.
    #[test]
    fn c8_auto_min_forces_floor() {
        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.level = LevelPolicy::Auto {
            min: Tier::Ocr,
            max: Tier::Vlm,
        };
        let plan = plan_tier(born_digital(), &policy);
        assert_eq!(plan.target, Tier::Ocr);
    }

    // C9: Fixed pins the tier and ignores signals.
    #[test]
    fn c9_fixed_pins_tier() {
        let plan = plan_tier(scanned(), &DocExtractPolicy::fixed(Tier::Native));
        assert_eq!(plan.target, Tier::Native);
        assert!(plan.escalations.is_empty());
    }

    // R5 (planning half): Ceiling(Ocr) forbids the VLM even on a structured page.
    #[test]
    fn r5_ceiling_forbids_vlm() {
        let signals = born_digital().with_structured(true);
        let mut policy = DocExtractPolicy::ceiling(Tier::Ocr);
        policy.want = OutputWant::Structure;
        let plan = plan_tier(signals, &policy);
        assert_eq!(plan.target, Tier::Ocr);
        // The Ocr->Vlm climb is dropped by the ceiling.
        assert!(plan.escalations.iter().all(|e| e.to <= Tier::Ocr));
    }

    // C7a: Native empty -> failure-driven climb to Ocr.
    #[test]
    fn c7a_empty_native_escalates() {
        let obs = TierObservation {
            empty: true,
            flat: false,
        };
        let esc = escalate_on_failure(Tier::Native, obs, &DocExtractPolicy::auto_up_to(Tier::Vlm));
        let esc = esc.expect("should escalate");
        assert_eq!(esc.to, Tier::Ocr);
        assert_eq!(esc.reason, EscalationReason::EmptyOutput);
    }

    // C7b: Ocr flat + want:Structure -> failure-driven climb to Vlm.
    #[test]
    fn c7b_flat_ocr_escalates_for_structure() {
        let obs = TierObservation {
            empty: false,
            flat: true,
        };
        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;
        let esc = escalate_on_failure(Tier::Ocr, obs, &policy).expect("should escalate");
        assert_eq!(esc.to, Tier::Vlm);
        assert_eq!(esc.reason, EscalationReason::FlatWhereStructureExpected);
    }

    // Failure-driven respects the ceiling: a flat OCR page under Ceiling(Ocr)
    // does not climb to the VLM.
    #[test]
    fn failure_escalation_respects_ceiling() {
        let obs = TierObservation {
            empty: false,
            flat: true,
        };
        let mut policy = DocExtractPolicy::ceiling(Tier::Ocr);
        policy.want = OutputWant::Structure;
        assert!(escalate_on_failure(Tier::Ocr, obs, &policy).is_none());
    }
}
