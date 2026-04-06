use serde::{Deserialize, Serialize};

/// Define the decay behavior for synaptic weights in the model.
/// - `EdgeOnly`: Only recover epsilon on `add()`, no passive decay over time.
/// - `EdgeAndFront`: Recover epsilon on `add()` and also passively decay of delta
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecayMode {
    /// Only recover epsilon on `add()` — no passive decay over time.
    EdgeOnly,
    /// Epsilon recovery on `add()` + decay delta on tick().
    EdgeAndFront,
}

/// Shape of the tick decay curve, affecting how weights return to the target value over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TickShape {
    /// Constant speed toward the target
    Lin,
    /// Decay quickly at the beginning, then slower at the end.
    Exp,
    /// Decay slowly at the beginning, then faster at the end.
    Log,
}

/// Overshoot factor for depressed weights. 
/// Returns a positive peak only for weights significantly below 1.0.
pub fn reminiscence_peak(w: f32, factor: f32) -> f32 {
    let depression = (1.0 - w).max(0.0);
    depression * factor
}

/// Update the weight `w` from one tick.
/// `w_max` is the maximum allowed weight.
pub fn tick_step(w: f32, tick_rate: f32, reminiscence_factor: f32, shape: &TickShape, w_max: f32) -> f32 {
    let target = 1.0 + reminiscence_peak(w, reminiscence_factor);
    let delta = target - w;
    let step = match shape {
        TickShape::Lin      => tick_rate * delta,
        TickShape::Exp => tick_rate * delta * delta.abs(),
        TickShape::Log => tick_rate * delta / (1.0 + delta.abs()),
    };
    (w + step).min(w_max)
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
        let w = tick_step(0.0, 0.1, 0.0, &TickShape::Lin, 2.0);
        assert!(w > 0.0 && w < 1.0, "w={w}");
    }

    #[test]
    fn tick_exponential_moves_toward_1() {
        let w = tick_step(0.0, 0.1, 0.0, &TickShape::Exp, 2.0);
        assert!(w > 0.0 && w < 1.0, "w={w}");
    }

    #[test]
    fn tick_does_not_exceed_w_max() {
        let w = tick_step(0.0, 0.9, 1.0, &TickShape::Lin, 1.5);
        assert!(w <= 1.5, "w={w}");
    }

    #[test]
    fn reminiscence_peak_full_for_zero_weight() {
        assert_abs_diff_eq!(reminiscence_peak(0.0, 1.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn reminiscence_peak_zero_for_high_weight() {
        assert_abs_diff_eq!(reminiscence_peak(1.0, 1.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn logarithmic_slower_than_linear_far_from_target() {
        let linear = tick_step(0.1, 0.5, 0.0, &TickShape::Lin, 2.0);
        let log    = tick_step(0.1, 0.5, 0.0, &TickShape::Log, 2.0);
        assert!(log < linear, "log={log}, linear={linear}");
    }
}
