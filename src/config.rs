use crate::decay::{DecayMode, TickShape};
use serde::{Deserialize, Serialize};

/// Fruit Fly Bloom Filter hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFBFConfig {
    /// Input vector dimension, which must match all inserted vectors.
    pub input_dim: usize,
    /// Number of KC->MBON synapses representing the size of the filter. Recommended: `30 * expected_n`.
    pub m: usize,
    /// Number of activated KCs per stimulus (= number of indices on which project hashed inputs, typically ~5% of m).
    pub k: usize,
    /// PNs ratio mapped for each KC (= fraction of inputs connected per KC, matrix sparsity).
    pub projection_sparsity: f32,
    /// Depression factor for active synapses on `add()` (= learning factor per input, range: [0.0, 1.0]).
    pub delta: f32,
    /// Increment for recovering inactive synapses on `add()` (= forgetting factor per input by memory saturation effect, range: [0.0, 1.0]).
    pub epsilon: f32,
    /// Maximum weight value. Default to 1.0, but can be set above 1.0 to allow for reminiscence overshoot [TODO feature]. 
    pub w_max: f32,
    /// Decay mode for forgetting behavior. 
    /// EdgeOnly = only recover epsilon on `add()` for inactive synapses (memory saturation effect only), 
    /// EdgeAndFront = also recover epsilon on `add()` + passive decay of delta on `tick()` (entropy effect), for both active and inactive synapses.
    pub decay_mode: DecayMode,
    /// Shape for the tick decay curve, affecting how weights return to the target value over time.
    /// Lin = constant speed toward the target,
    /// Exp = decay quickly at the beginning, then slower at the end (strengthens strong memories and weakens weak ones),
    /// Log = decay slowly at the beginning, then faster at the end (strengthens weak memories and weakens strong ones).
    pub tick_shape: TickShape,
    /// Decay speed for passive decay on `tick()`. Range: (0.0, 1.0).
    pub tick_rate: f32,
    /// Overshoot factor for reminiscence above 1.0. Range: [0.0, 1.0]. [TODO feature]
    // pub reminiscence_factor: f32,
    /// Number of recent novelty scores to keep in the ring buffer for dynamic thresholding. 
    pub window_size: usize,
    /// Optional RNG seed for deterministic projection matrix generation, for reproducibility. 
    pub seed: Option<u64>,
}

impl FFBFConfig {
    /// Build default config based on input dimension and expected number of items to store.
    pub fn default_for(input_dim: usize, expected_n: usize) -> Self {
        let m = 30 * expected_n.max(1);
        let k = ((m as f32 * 0.05).round() as usize).max(1);
        
        Self {
            input_dim,
            m,
            k,
            projection_sparsity: 0.12,
            delta: 0.5,
            epsilon: 0.05,
            w_max: 1.0,
            decay_mode: DecayMode::EdgeOnly,
            tick_shape: TickShape::Exp,
            tick_rate: 0.01,
            // pub reminiscence_factor: f32,
            window_size: 100,
            seed: None,
        }
    }

    /// Self validate all hyperparameters ranges. 
    /// Returns `Err` with description if invalid, otherwise `Ok(())`.
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("input_dim must be >= 1".to_string());
        }
        if self.k == 0 {
            return Err("k must be >= 1".to_string());
        }
        if self.k >= self.m {
            return Err(format!("k ({}) must be < m ({})", self.k, self.m));
        }
        if self.delta < 0.0 || self.delta >= 1.0 {
            return Err(format!("delta ({}) must be in [0.0, 1.0[", self.delta));
        }
        if self.epsilon < 0.0 || self.epsilon > 1.0 {
            return Err(format!("epsilon ({}) must be in [0.0, 1.0]", self.epsilon));
        }
        if self.w_max < 1.0 {
            return Err(format!("w_max ({}) must be >= 1.0, which is the base weight value", self.w_max));
        }
        if self.tick_rate <= 0.0 {
            return Err(format!("tick_rate ({}) must be > 0.0", self.tick_rate));
        }
        if self.projection_sparsity <= 0.0 || self.projection_sparsity > 1.0 {
            return Err(format!(
                "projection_sparsity ({}) must be in ]0.0, 1.0]",
                self.projection_sparsity
            ));
        }
        if self.window_size < 2 {
            return Err(format!("window_size ({}) must be >= 2", self.window_size));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = FFBFConfig::default_for(128, 1000);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn default_k_is_5_percent_of_m() {
        let cfg = FFBFConfig::default_for(128, 1000);
        let expected_k = ((cfg.m as f32 * 0.05).round() as usize).max(1);
        assert_eq!(cfg.k, expected_k);
    }

    #[test]
    fn validate_rejects_zero_input_dim() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.input_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_k() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_k_ge_m() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.k = cfg.m;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_delta_out_of_range() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.delta = 1.0;
        assert!(cfg.validate().is_err());
        cfg.delta = -0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_w_max_below_1() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.w_max = 0.9;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_small_window() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.window_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_tick_rate() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.tick_rate = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_projection_sparsity() {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.projection_sparsity = 0.0;
        assert!(cfg.validate().is_err());
        cfg.projection_sparsity = 1.1;
        assert!(cfg.validate().is_err());
    }
}
