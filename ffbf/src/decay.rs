use serde::{Deserialize, Serialize};

/// Define the decay behavior for synaptic weights in the model.
/// - `EdgeOnly`: Only recover epsilon on `add()`, no passive decay over time.
/// - `EdgeAndFront`: Recover epsilon on `add()` and also passively decay of delta
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecayMode {
    /// Only recover epsilon on `add()`, no passive decay over time.
    EdgeOnly,
    /// Epsilon recovery on `add()` + decay delta on tick().
    EdgeAndFront,
}

/// Shape of the tick decay curve, setting how the forgetting speed depends on the current weight.
/// A low weight is a freshly learned memory, a weight near 1.0 one that is already almost forgotten.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TickShape {
    /// Uniform step: every synapse forgets at the same absolute speed, whatever it holds.
    Lin,
    /// Step scaled by the weight: a fresh memory barely moves, an old one accelerates away.
    /// Long retention followed by a fast wipe.
    Exp,
    /// Step scaled by the distance left to run: a fresh memory fades fast, then a long tail.
    Log,
}

/// Smallest weight `add()` may produce.
/// `TickShape::Exp` scales its step by the weight itself, so a weight reaching exactly 0.0 would
/// never recover. Flooring the depression keeps every synapse reachable by `tick()`.
pub(crate) const MIN_WEIGHT: f32 = 1e-3;

/// Overshoot factor for depressed weights, allowing the forgotten inputs to get an emotional boost when remembered. 
/// Returns a positive peak only for weights significantly below 1.0.
/// The more its learned, the more it can be boosted when remembered.
// pub fn reminiscence_peak(w: f32, factor: f32) -> f32 {
//     let depression = (1.0 - w).max(0.0);
//     depression * factor
// }

/// Calculate the new weight value after a tick, if decay EdgeAndFront is enabled.
/// `tick_rate` is a coefficient, and `shape` picks what it multiplies: nothing for `Lin`, the weight
/// for `Exp`, the remaining distance for `Log`.
/// Parameters:
///     `w` is the current weight to update.
///     `tick_rate` controls how quickly weights move toward the target (base value 1).
///     `shape` determines how the step scales with `w`, i.e. which memories are forgotten first.
///     `w_max` is the maximum allowed weight.
/// Returns the new weight after applying the tick decay, capped at the target and at `w_max`.
pub fn tick_step(w: f32, tick_rate: f32, shape: &TickShape, w_max: f32) -> f32 {
    //let target = 1.0 + reminiscence_peak(w, reminiscence_factor);
    let target = 1.0;
    let step = match shape {
        TickShape::Lin => tick_rate,
        TickShape::Exp => tick_rate * w,
        TickShape::Log => tick_rate * (target - w),
    };
    //Lin has a constant step, so unlike the other two it can overshoot the target on its own
    (w + step).min(target).min(w_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn decay_mode_clone() {
        let m = DecayMode::EdgeOnly;
        let _ = m.clone();
    }

    #[test]
    fn tick_shape_clone() {
        let s = TickShape::Exp;
        let _ = s.clone();
    }

    #[test]
    fn tick_linear_moves_toward_1() {
        let w = tick_step(0.0, 0.1, &TickShape::Lin, 2.0);
        assert!(w > 0.0 && w < 1.0, "w={w}");
    }

    #[test]
    fn tick_exponential_moves_toward_1() {
        let w = tick_step(0.5, 0.1, &TickShape::Exp, 2.0);
        assert!(w > 0.5 && w < 1.0, "w={w}");
    }

    #[test]
    fn exp_is_a_no_op_at_zero_weight() {
        //Exp scales its step by the weight, so 0.0 is a fixed point; add() floors weights above it
        assert_abs_diff_eq!(tick_step(0.0, 0.1, &TickShape::Exp, 2.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn lin_step_is_independent_of_weight() {
        let fresh = tick_step(0.1, 0.05, &TickShape::Lin, 2.0) - 0.1;
        let old = tick_step(0.8, 0.05, &TickShape::Lin, 2.0) - 0.8;
        assert_abs_diff_eq!(fresh, old, epsilon = 1e-6);
    }

    #[test]
    fn exp_forgets_an_old_memory_faster_than_a_fresh_one() {
        let fresh = tick_step(0.1, 0.1, &TickShape::Exp, 2.0) - 0.1;
        let old = tick_step(0.9, 0.1, &TickShape::Exp, 2.0) - 0.9;
        assert!(old > fresh, "old={old}, fresh={fresh}");
    }

    #[test]
    fn log_forgets_a_fresh_memory_faster_than_an_old_one() {
        let fresh = tick_step(0.1, 0.1, &TickShape::Log, 2.0) - 0.1;
        let old = tick_step(0.9, 0.1, &TickShape::Log, 2.0) - 0.9;
        assert!(fresh > old, "fresh={fresh}, old={old}");
    }

    #[test]
    fn tick_never_overshoots_target() {
        assert_abs_diff_eq!(tick_step(0.95, 0.5, &TickShape::Lin, 2.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn tick_does_not_exceed_w_max() {
        let w = tick_step(0.0, 0.9, &TickShape::Lin, 1.5);
        assert!(w <= 1.5, "w={w}");
    }

    // #[test]
    // fn reminiscence_peak_full_for_zero_weight() {
    //     assert_abs_diff_eq!(reminiscence_peak(0.0, 1.0), 1.0, epsilon = 1e-6);
    // }

    // #[test]
    // fn reminiscence_peak_zero_for_high_weight() {
    //     assert_abs_diff_eq!(reminiscence_peak(1.0, 1.0), 0.0, epsilon = 1e-6);
    // }

    #[test]
    fn logarithmic_slower_than_linear_far_from_target() {
        let linear = tick_step(0.1, 0.5, &TickShape::Lin, 2.0);
        let log    = tick_step(0.1, 0.5, &TickShape::Log, 2.0);
        assert!(log < linear, "log={log}, linear={linear}");
    }
}
