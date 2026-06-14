/// Compute the raw novelty score from synaptic weights at active KC positions.
/// Parameters:
///   weights: full weight vector over all KCs.
///   active_kcs: sorted indices of active KCs (top-k from projection).
/// Returns:
///   Mean weight at active positions. Higher means more novel (weights not yet depressed).
/// Panics:
///   If any index in `active_kcs` is out of bounds for `weights`.
pub fn novelty_score(weights: &[f32], active_kcs: &[usize]) -> f32 {
    assert!(!active_kcs.is_empty(), "active_kcs must not be empty");

    let sum: f32 = active_kcs.iter().map(|&i| weights[i]).sum();

    sum / active_kcs.len() as f32
}

/// Ring buffer tracking recent novelty scores for adaptive thresholding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoveltyWindow {
    buf: Vec<f32>,
    head: usize,
    len: usize,
}

impl NoveltyWindow {
    /// Create a new window with the given capacity.
    /// Parameters:
    ///   capacity: maximum number of scores retained.
    /// Panics:
    ///   If `capacity` < 2.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 2, "window capacity must be >= 2");
        Self { buf: vec![0.0; capacity], head: 0, len: 0 }
    }

    /// Push a new score into the window, evicting the oldest if full.
    pub fn push(&mut self, score: f32) {
        self.buf[self.head] = score;
        self.head = (self.head + 1) % self.buf.len();

        if self.len < self.buf.len() {
            self.len += 1;
        }
    }

    /// Mean of scores currently in the window. Returns `None` if empty.
    pub fn mean(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }

        let sum: f32 = self.buf[..self.len].iter().sum();

        Some(sum / self.len as f32)
    }

    /// Number of scores currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the window has no scores yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `score` exceeds the window mean by at least `threshold` ratio.
    /// Parameters:
    ///   score: Candidate novelty score.
    ///   threshold: Multiplier on the mean; e.g. `1.0` means "above mean".
    /// Returns:
    ///   `true` if window is empty (no baseline yet) or `score >= mean * threshold`.
    pub fn is_novel(&self, score: f32, threshold: f32) -> bool {
        match self.mean() {
            None => true,
            Some(m) => score >= m * threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn novelty_score_uniform_weights() {
        let weights = vec![1.0f32; 10];
        let active = vec![0, 2, 5];
        assert_abs_diff_eq!(novelty_score(&weights, &active), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn novelty_score_depressed_weights() {
        let mut weights = vec![1.0f32; 10];
        weights[2] = 0.4;
        weights[5] = 0.6;
        let active = vec![2, 5];
        assert_abs_diff_eq!(novelty_score(&weights, &active), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn novelty_score_single_kc() {
        let weights = vec![0.7f32; 5];
        assert_abs_diff_eq!(novelty_score(&weights, &[3]), 0.7, epsilon = 1e-6);
    }

    #[test]
    fn window_mean_empty() {
        let w = NoveltyWindow::new(4);
        assert!(w.mean().is_none());
    }

    #[test]
    fn window_mean_partial() {
        let mut w = NoveltyWindow::new(4);
        w.push(1.0);
        w.push(3.0);
        assert_abs_diff_eq!(w.mean().unwrap(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn window_evicts_oldest_when_full() {
        let mut w = NoveltyWindow::new(3);
        w.push(1.0);
        w.push(1.0);
        w.push(1.0);
        w.push(4.0); // evicts first 1.0
        assert_abs_diff_eq!(w.mean().unwrap(), 2.0, epsilon = 1e-6);
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn window_is_novel_empty_baseline() {
        let w = NoveltyWindow::new(4);
        assert!(w.is_novel(0.0, 1.0));
    }

    #[test]
    fn window_is_novel_above_mean() {
        let mut w = NoveltyWindow::new(4);
        w.push(1.0);
        w.push(1.0);
        assert!(w.is_novel(1.1, 1.0));
        assert!(!w.is_novel(0.9, 1.0));
    }

    #[test]
    fn window_is_empty_flag() {
        let mut w = NoveltyWindow::new(4);
        assert!(w.is_empty());
        w.push(0.5);
        assert!(!w.is_empty());
    }
}
