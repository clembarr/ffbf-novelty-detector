use ffbf::{DecayMode, FFBFConfig, TickShape};
use pyo3::prelude::*;

/// Python-facing enum mirroring [`ffbf::DecayMode`].
///
/// Defines decay behavior for synaptic weights in the FFBF model.
/// - `EdgeOnly`: Only recover epsilon on `add()`, no passive decay over time.
/// - `EdgeAndFront`: Recover epsilon on `add()` + decay delta on `tick()`.
//skip_from_py_object prevents auto-generated FromPyObject impl that conflicts with the From<T> conversions used for Rust↔Python round-tripping
#[pyclass(name = "DecayMode", eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum PyDecayMode {
    EdgeOnly,
    EdgeAndFront,
}

impl From<PyDecayMode> for DecayMode {
    fn from(v: PyDecayMode) -> Self {
        match v {
            PyDecayMode::EdgeOnly => DecayMode::EdgeOnly,
            PyDecayMode::EdgeAndFront => DecayMode::EdgeAndFront,
        }
    }
}

impl From<DecayMode> for PyDecayMode {
    fn from(v: DecayMode) -> Self {
        match v {
            DecayMode::EdgeOnly => PyDecayMode::EdgeOnly,
            DecayMode::EdgeAndFront => PyDecayMode::EdgeAndFront,
        }
    }
}

/// Python-facing enum mirroring [`ffbf::TickShape`].
///
/// Shape of the tick decay curve, affecting how weights return to target value over time.
/// - `Lin`: Constant speed toward the target.
/// - `Exp`: Decays quickly at the start, then slower at the end.
/// - `Log`: Decays slowly at the start, then faster at the end.
//skip_from_py_object prevents auto-generated FromPyObject impl that conflicts with the From<T> conversions used for Rust↔Python round-tripping
#[pyclass(name = "TickShape", eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum PyTickShape {
    Lin,
    Exp,
    Log,
}

impl From<PyTickShape> for TickShape {
    fn from(v: PyTickShape) -> Self {
        match v {
            PyTickShape::Lin => TickShape::Lin,
            PyTickShape::Exp => TickShape::Exp,
            PyTickShape::Log => TickShape::Log,
        }
    }
}

impl From<TickShape> for PyTickShape {
    fn from(v: TickShape) -> Self {
        match v {
            TickShape::Lin => PyTickShape::Lin,
            TickShape::Exp => PyTickShape::Exp,
            TickShape::Log => PyTickShape::Log,
        }
    }
}

/// Python-facing wrapper for [`ffbf::FFBFConfig`].
///
/// Holds all hyperparameters for the FFBF filter. Build a default config with
/// `FFBFConfig.default_for(input_dim, expected_n)`, then adjust fields as needed
/// before passing to `FFBF(cfg)`.
#[pyclass(name = "FFBFConfig", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFFBFConfig(pub(crate) FFBFConfig);

#[pymethods]
impl PyFFBFConfig {
    /// Build a default config for the given input dimension and expected item count.
    #[staticmethod]
    fn default_for(input_dim: usize, expected_n: usize) -> Self {
        PyFFBFConfig(FFBFConfig::default_for(input_dim, expected_n))
    }

    fn __repr__(&self) -> String {
        format!(
            "FFBFConfig(input_dim={}, m={}, k={}, delta={:.3}, epsilon={:.3})",
            self.0.input_dim, self.0.m, self.0.k, self.0.delta, self.0.epsilon
        )
    }

    #[getter] fn input_dim(&self) -> usize { self.0.input_dim }
    #[setter] fn set_input_dim(&mut self, v: usize) { self.0.input_dim = v; }

    #[getter] fn m(&self) -> usize { self.0.m }
    #[setter] fn set_m(&mut self, v: usize) { self.0.m = v; }

    #[getter] fn k(&self) -> usize { self.0.k }
    #[setter] fn set_k(&mut self, v: usize) { self.0.k = v; }

    #[getter] fn projection_sparsity(&self) -> f32 { self.0.projection_sparsity }
    #[setter] fn set_projection_sparsity(&mut self, v: f32) { self.0.projection_sparsity = v; }

    #[getter] fn delta(&self) -> f32 { self.0.delta }
    #[setter] fn set_delta(&mut self, v: f32) { self.0.delta = v; }

    #[getter] fn epsilon(&self) -> f32 { self.0.epsilon }
    #[setter] fn set_epsilon(&mut self, v: f32) { self.0.epsilon = v; }

    #[getter] fn w_max(&self) -> f32 { self.0.w_max }
    #[setter] fn set_w_max(&mut self, v: f32) { self.0.w_max = v; }

    #[getter]
    fn decay_mode(&self) -> PyDecayMode { self.0.decay_mode.clone().into() }
    #[setter]
    fn set_decay_mode(&mut self, v: Bound<'_, PyDecayMode>) {
        self.0.decay_mode = v.borrow().clone().into();
    }

    #[getter]
    fn tick_shape(&self) -> PyTickShape { self.0.tick_shape.clone().into() }
    #[setter]
    fn set_tick_shape(&mut self, v: Bound<'_, PyTickShape>) {
        self.0.tick_shape = v.borrow().clone().into();
    }

    #[getter] fn tick_rate(&self) -> f32 { self.0.tick_rate }
    #[setter] fn set_tick_rate(&mut self, v: f32) { self.0.tick_rate = v; }

    #[getter] fn window_size(&self) -> usize { self.0.window_size }
    #[setter] fn set_window_size(&mut self, v: usize) { self.0.window_size = v; }

    #[getter] fn seed(&self) -> Option<u64> { self.0.seed }
    #[setter] fn set_seed(&mut self, v: Option<u64>) { self.0.seed = v; }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecayMode>()?;
    m.add_class::<PyTickShape>()?;
    m.add_class::<PyFFBFConfig>()?;
    Ok(())
}
