use crate::decay::{DecayMode, TickShape};
use serde::{Deserialize, Serialize};

/// Hyperparamètres d'un filtre FFBF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFBFConfig {
    /// Dimension du vecteur d'entrée — doit correspondre à tous les vecteurs insérés.
    pub input_dim: usize,
    /// Nombre de synapses KC→MBON (taille du filtre). Recommandé : `30 * expected_n`.
    pub m: usize,
    /// Nombre de KCs actifs par stimulus (~5% de m).
    pub k: usize,
    /// Ratio de PNs connectés par KC (sparsité de la matrice). Défaut biologique : 0.12.
    pub projection_sparsity: f32,
    /// Facteur de décroissance des synapses actives sur `add()`. Plage : [0.0, 1.0).
    pub delta: f32,
    /// Incrément de récupération des synapses inactives sur `add()`. Plage : [0.0, 1.0].
    pub epsilon: f32,
    /// Valeur maximale d'un poids. Défaut : 2.0.
    pub w_max: f32,
    /// Mode de décroissance : edge only ou edge + front.
    pub decay_mode: DecayMode,
    /// Forme de la courbe de récupération tick().
    pub tick_shape: TickShape,
    /// Vitesse de décroissance passive par appel tick(). Plage : (0.0, 1.0].
    pub tick_rate: f32,
    /// Amplitude du dépassement de réminiscence au-dessus de 1.0. Plage : [0.0, 1.0].
    pub reminiscence_factor: f32,
    /// Nombre de scores de nouveauté récents dans le ring buffer. Min : 2.
    pub window_size: usize,
    /// Graine RNG optionnelle pour une matrice de projection déterministe.
    pub seed: Option<u64>,
}

impl FFBFConfig {
    /// Construit une config sensée pour des vecteurs de dimension `input_dim`
    /// et une population attendue de `expected_n` éléments.
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
            w_max: 2.0,
            decay_mode: DecayMode::EdgeOnly,
            tick_shape: TickShape::Exp,
            tick_rate: 0.01,
            reminiscence_factor: 0.0,
            window_size: 100,
            seed: None,
        }
    }

    /// Valide toutes les plages d'hyperparamètres. Retourne `Err` avec description si invalide.
    pub fn validate(&self) -> Result<(), String> {
        if self.k >= self.m {
            return Err(format!("k ({}) must be < m ({})", self.k, self.m));
        }
        if self.delta < 0.0 || self.delta >= 1.0 {
            return Err(format!("delta ({}) must be in [0.0, 1.0)", self.delta));
        }
        if self.epsilon < 0.0 || self.epsilon > 1.0 {
            return Err(format!("epsilon ({}) must be in [0.0, 1.0]", self.epsilon));
        }
        if self.w_max < 1.0 {
            return Err(format!("w_max ({}) must be >= 1.0", self.w_max));
        }
        if self.tick_rate <= 0.0 {
            return Err(format!("tick_rate ({}) must be > 0.0", self.tick_rate));
        }
        if self.projection_sparsity <= 0.0 || self.projection_sparsity > 1.0 {
            return Err(format!(
                "projection_sparsity ({}) must be in (0.0, 1.0]",
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
