import pytest
from ffbf import DecayMode, FFBFConfig, TickShape


def test_decay_mode_variants_exist():
    assert DecayMode.EdgeOnly is not None
    assert DecayMode.EdgeAndFront is not None


def test_decay_mode_equality():
    assert DecayMode.EdgeOnly == DecayMode.EdgeOnly
    assert DecayMode.EdgeOnly != DecayMode.EdgeAndFront


def test_tick_shape_variants_exist():
    assert TickShape.Lin is not None
    assert TickShape.Exp is not None
    assert TickShape.Log is not None


def test_tick_shape_equality():
    assert TickShape.Exp == TickShape.Exp
    assert TickShape.Lin != TickShape.Log


def test_ffbfconfig_default_for():
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    assert cfg.input_dim == 64
    assert cfg.m == 1500   # 30 * 50
    assert cfg.k >= 1
    assert cfg.window_size == 100


def test_ffbfconfig_field_setters():
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    cfg.delta = 0.3
    assert abs(cfg.delta - 0.3) < 1e-6
    cfg.seed = 42
    assert cfg.seed == 42
    cfg.seed = None
    assert cfg.seed is None


def test_ffbfconfig_enum_setters():
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    cfg.decay_mode = DecayMode.EdgeAndFront
    assert cfg.decay_mode == DecayMode.EdgeAndFront
    cfg.tick_shape = TickShape.Exp
    assert cfg.tick_shape == TickShape.Exp


def test_ffbfconfig_repr():
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    r = repr(cfg)
    assert "FFBFConfig" in r
    assert "64" in r
