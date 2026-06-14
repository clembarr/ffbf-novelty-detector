use ffbf::{DecayMode, FFBFConfig, TickShape, FFBF};
use pyo3::prelude::*;

/// Python-facing enum mirroring `ffbf::DecayMode`.
#[pyclass(name = "DecayMode", eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
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

/// Python-facing enum mirroring `ffbf::TickShape`.
#[pyclass(name = "TickShape", eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
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
