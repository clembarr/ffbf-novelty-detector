use crate::analysis::{novelty_score, NoveltyWindow};
use crate::config::FFBFConfig;
use crate::decay::{tick_step, DecayMode};
use crate::projection::Projection;

/// Fruit Fly Bloom Filter — novelty detector based on sparse random projection
/// and Hebbian-like synaptic weight updates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FFBF {
    pub(crate) cfg: FFBFConfig,
    pub(crate) projection: Projection,
    /// Synaptic weights over KCs, initialised to 1.0.
    pub(crate) weights: Vec<f32>,
    pub(crate) window: NoveltyWindow,
}

impl FFBF {
    /// Build a new FFBF from `cfg`.
    ///
    /// Parameters:
    ///   cfg: Hyperparameter config. Must pass `validate()`.
    /// Returns:
    ///   `Ok(FFBF)` on success.
    /// Errors:
    ///   `Err(String)` if `cfg.validate()` fails.
    pub fn new(cfg: FFBFConfig) -> Result<Self, String> {
        cfg.validate()?;
        let projection = Projection::new(&cfg);
        let weights = vec![1.0f32; cfg.m];
        let window = NoveltyWindow::new(cfg.window_size);
        Ok(Self { cfg, projection, weights, window })
    }

    /// Process one input: score novelty, push to window, then update weights.
    ///
    /// Parameters:
    ///   input: Slice of length `cfg.input_dim`.
    /// Panics:
    ///   If `input.len() != cfg.input_dim`.
    pub fn add(&mut self, input: &[f32]) {
        assert_eq!(
            input.len(), self.cfg.input_dim,
            "input length {} != input_dim {}", input.len(), self.cfg.input_dim
        );
        let active = self.projection.project(input);
        let score = novelty_score(&self.weights, &active);
        self.window.push(score);
        //depress active synapses, recover inactive
        let scale = 1.0 - self.cfg.delta;
        for (i, w) in self.weights.iter_mut().enumerate() {
            if active.binary_search(&i).is_ok() {
                *w *= scale;
            } else {
                *w = (*w + self.cfg.epsilon).min(self.cfg.w_max);
            }
        }
    }

    /// Passive decay step: move all weights toward 1.0 (or reminiscence peak).
    /// No-op when `cfg.decay_mode` is `EdgeOnly`.
    pub fn tick(&mut self) {
        if self.cfg.decay_mode == DecayMode::EdgeOnly {
            return;
        }
        for w in self.weights.iter_mut() {
            *w = tick_step(*w, self.cfg.tick_rate, self.cfg.reminiscence_factor, &self.cfg.tick_shape, self.cfg.w_max);
        }
    }

    /// Compute novelty score for `input` without modifying state.
    ///
    /// Parameters:
    ///   input: Slice of length `cfg.input_dim`.
    /// Returns:
    ///   Mean weight at active KC positions. Higher = more novel.
    pub fn novelty(&self, input: &[f32]) -> f32 {
        assert_eq!(
            input.len(), self.cfg.input_dim,
            "input length {} != input_dim {}", input.len(), self.cfg.input_dim
        );
        let active = self.projection.project(input);
        novelty_score(&self.weights, &active)
    }

    /// Whether `input` is novel relative to the adaptive window baseline.
    ///
    /// Parameters:
    ///   input: Slice of length `cfg.input_dim`.
    ///   threshold: Multiplier on the window mean; `1.0` = at or above mean.
    /// Returns:
    ///   `true` if window is empty or `novelty(input) >= window_mean * threshold`.
    pub fn is_novel(&self, input: &[f32], threshold: f32) -> bool {
        self.window.is_novel(self.novelty(input), threshold)
    }

    /// Current synaptic weights (read-only view).
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Number of scores in the novelty window.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn cfg_base() -> FFBFConfig {
        let mut cfg = FFBFConfig::default_for(64, 50);
        cfg.seed = Some(0);
        cfg
    }

    #[test]
    fn new_rejects_invalid_config() {
        let mut cfg = cfg_base();
        cfg.k = cfg.m;
        assert!(FFBF::new(cfg).is_err());
    }

    #[test]
    fn weights_initialised_to_one() {
        let ffbf = FFBF::new(cfg_base()).unwrap();
        assert!(ffbf.weights().iter().all(|&w| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn add_depresses_active_weights() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        let active = ffbf.projection.project(&input);
        ffbf.add(&input);
        for &i in &active {
            assert!(ffbf.weights[i] < 1.0, "active weight should be depressed");
        }
    }

    #[test]
    fn add_recovers_inactive_weights_only_after_second_add() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        let active = ffbf.projection.project(&input);
        ffbf.add(&input);
        //inactive weights were already at w_max=1.0 before first add; epsilon pushes them above 1.0
        for (i, &w) in ffbf.weights.iter().enumerate() {
            if active.binary_search(&i).is_err() {
                assert!(w > 1.0 || (w - ffbf.cfg.w_max).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn add_pushes_score_to_window() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        assert_eq!(ffbf.window_len(), 0);
        ffbf.add(&vec![0.5f32; ffbf.cfg.input_dim]);
        assert_eq!(ffbf.window_len(), 1);
    }

    #[test]
    fn novelty_fresh_filter_is_high() {
        let ffbf = FFBF::new(cfg_base()).unwrap();
        let score = ffbf.novelty(&vec![1.0f32; ffbf.cfg.input_dim]);
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn novelty_decreases_after_repeated_adds() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        let initial = ffbf.novelty(&input);
        for _ in 0..10 {
            ffbf.add(&input);
        }
        let after = ffbf.novelty(&input);
        assert!(after < initial, "score should decrease: initial={initial}, after={after}");
    }

    #[test]
    fn tick_no_op_for_edge_only() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        ffbf.add(&input);
        let before: Vec<f32> = ffbf.weights().to_vec();
        ffbf.tick();
        assert_eq!(ffbf.weights(), before.as_slice());
    }

    #[test]
    fn tick_moves_weights_for_edge_and_front() {
        let mut cfg = cfg_base();
        cfg.decay_mode = DecayMode::EdgeAndFront;
        let mut ffbf = FFBF::new(cfg).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        ffbf.add(&input);
        let depressed: Vec<f32> = ffbf.weights().to_vec();
        ffbf.tick();
        //at least one weight should have changed (active KCs were depressed)
        assert_ne!(ffbf.weights(), depressed.as_slice());
    }

    #[test]
    fn is_novel_true_on_empty_window() {
        let ffbf = FFBF::new(cfg_base()).unwrap();
        assert!(ffbf.is_novel(&vec![0.0f32; ffbf.cfg.input_dim], 1.0));
    }

    #[test]
    fn repeated_input_eventually_not_novel() {
        let mut ffbf = FFBF::new(cfg_base()).unwrap();
        let input = vec![1.0f32; ffbf.cfg.input_dim];
        for _ in 0..50 {
            ffbf.add(&input);
        }
        assert!(!ffbf.is_novel(&input, 1.0));
    }
}
