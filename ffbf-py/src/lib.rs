use ffbf::{DecayMode, FFBFConfig, TickShape, FFBF};
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecayMode>()?;
    m.add_class::<PyTickShape>()?;
    Ok(())
}
