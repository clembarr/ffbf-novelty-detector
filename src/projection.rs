use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::config::FFBFConfig;

/// Sparse random projection from input space to KC activations.
///
/// Each KC connects to a random subset of input neurons (PNs).
/// Projection computes KC activations and returns the top-k winner indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    /// For each KC, sorted indices of connected input neurons.
    pub(crate) connections: Vec<Vec<usize>>,
    pub(crate) k: usize,
}

impl Projection {
    /// Build a sparse random projection from the given config.
    ///
    /// Parameters:
    ///   cfg: Validated `FFBFConfig` providing `input_dim`, `m`, `k`, `projection_sparsity`, `seed`.
    pub fn new(cfg: &FFBFConfig) -> Self {
        let fan_in = ((cfg.projection_sparsity * cfg.input_dim as f32).round() as usize).max(1);
        let mut rng: ChaCha8Rng = match cfg.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let mut indices: Vec<usize> = (0..cfg.input_dim).collect();
        let connections = (0..cfg.m)
            .map(|_| {
                indices.shuffle(&mut rng);
                let mut c = indices[..fan_in].to_vec();
                c.sort_unstable();
                c
            })
            .collect();
        Self { connections, k: cfg.k }
    }

    /// Project `input` to KC space and return the top-k active KC indices.
    ///
    /// Parameters:
    ///   input: Slice of length `input_dim`.
    /// Returns:
    ///   Sorted vec of `k` KC indices with the highest activation.
    /// Panics:
    ///   If `input.len()` does not match `input_dim` used at construction.
    pub fn project(&self, input: &[f32]) -> Vec<usize> {
        assert!(
            self.connections.iter().all(|c| c.iter().all(|&i| i < input.len())),
            "input length {} too small for this projection", input.len()
        );
        let mut activations: Vec<(usize, f32)> = self
            .connections
            .iter()
            .enumerate()
            .map(|(kc, conns)| {
                let act: f32 = conns.iter().map(|&i| input[i]).sum();
                (kc, act)
            })
            .collect();
        //partial sort: only top-k needed
        activations.select_nth_unstable_by(self.k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut top_k: Vec<usize> = activations[..self.k].iter().map(|&(kc, _)| kc).collect();
        top_k.sort_unstable();
        top_k
    }

    /// Number of KCs (output dimension).
    pub fn m(&self) -> usize {
        self.connections.len()
    }

    /// Number of active KCs per projection.
    pub fn k(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FFBFConfig;

    fn cfg_base() -> FFBFConfig {
        let mut cfg = FFBFConfig::default_for(128, 100);
        cfg.seed = Some(42);
        cfg
    }

    #[test]
    fn project_returns_k_indices() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        let input = vec![1.0f32; cfg.input_dim];
        let result = proj.project(&input);
        assert_eq!(result.len(), cfg.k);
    }

    #[test]
    fn project_indices_in_range() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        let input = vec![1.0f32; cfg.input_dim];
        let result = proj.project(&input);
        assert!(result.iter().all(|&i| i < cfg.m));
    }

    #[test]
    fn project_indices_sorted_and_unique() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        let input = vec![1.0f32; cfg.input_dim];
        let result = proj.project(&input);
        for w in result.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let cfg = cfg_base();
        let p1 = Projection::new(&cfg);
        let p2 = Projection::new(&cfg);
        let input = vec![0.5f32; cfg.input_dim];
        assert_eq!(p1.project(&input), p2.project(&input));
    }

    #[test]
    fn different_inputs_can_differ() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        let mut a = vec![0.0f32; cfg.input_dim];
        let mut b = vec![0.0f32; cfg.input_dim];
        a[0] = 10.0;
        b[cfg.input_dim - 1] = 10.0;
        //not guaranteed, but very likely with a reasonable projection
        let ra = proj.project(&a);
        let rb = proj.project(&b);
        assert_ne!(ra, rb);
    }

    #[test]
    fn fan_in_respects_sparsity() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        let expected_fan_in = ((cfg.projection_sparsity * cfg.input_dim as f32).round() as usize).max(1);
        assert!(proj.connections.iter().all(|c| c.len() == expected_fan_in));
    }

    #[test]
    fn accessors() {
        let cfg = cfg_base();
        let proj = Projection::new(&cfg);
        assert_eq!(proj.m(), cfg.m);
        assert_eq!(proj.k(), cfg.k);
    }
}
