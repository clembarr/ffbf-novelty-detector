import json
import os
import tempfile

import numpy as np
import pytest
from ffbf import DecayMode, FFBF, FFBFConfig, TickShape


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


def _make_ffbf() -> FFBF:
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    cfg.seed = 42
    return FFBF(cfg)


def test_ffbf_new_valid():
    f = _make_ffbf()
    assert f.window_len() == 0


def test_ffbf_new_invalid_config_raises():
    cfg = FFBFConfig.default_for(input_dim=64, expected_n=50)
    cfg.k = cfg.m
    with pytest.raises(ValueError):
        FFBF(cfg)


def test_ffbf_add_list():
    f = _make_ffbf()
    f.add([0.5] * 64)
    assert f.window_len() == 1


def test_ffbf_add_numpy():
    f = _make_ffbf()
    f.add(np.ones(64, dtype=np.float32))
    assert f.window_len() == 1


def test_ffbf_add_wrong_dtype_raises():
    f = _make_ffbf()
    with pytest.raises(ValueError):
        f.add(np.ones(64, dtype=np.float64))


def test_ffbf_novelty_fresh_is_one():
    f = _make_ffbf()
    score = f.novelty(np.ones(64, dtype=np.float32))
    assert abs(score - 1.0) < 1e-6


def test_ffbf_novelty_decreases_after_adds():
    f = _make_ffbf()
    vec = np.ones(64, dtype=np.float32)
    initial = f.novelty(vec)
    for _ in range(10):
        f.add(vec)
    assert f.novelty(vec) < initial


def test_ffbf_is_novel_empty_window():
    f = _make_ffbf()
    assert f.is_novel([0.5] * 64, 1.0) is True


def test_ffbf_is_novel_false_after_repetition():
    f = _make_ffbf()
    vec = [1.0] * 64
    for _ in range(50):
        f.add(vec)
    assert f.is_novel(vec, 1.0) is False


def test_ffbf_weights_returns_float32_numpy():
    f = _make_ffbf()
    w = f.weights()
    assert isinstance(w, np.ndarray)
    assert w.dtype == np.float32


def test_ffbf_weights_are_copy():
    f = _make_ffbf()
    w = f.weights()
    w[0] = 999.0
    assert f.weights()[0] != 999.0


def test_ffbf_tick_no_op_edge_only():
    f = _make_ffbf()
    f.add(np.ones(64, dtype=np.float32))
    before = f.weights().copy()
    f.tick()
    np.testing.assert_array_equal(f.weights(), before)


def test_ffbf_repr():
    f = _make_ffbf()
    assert "FFBF" in repr(f)


def test_ffbf_to_json_is_valid_json():
    f = _make_ffbf()
    f.add(np.ones(64, dtype=np.float32))
    j = f.to_json()
    parsed = json.loads(j)
    assert "weights" in parsed


def test_ffbf_from_json_roundtrip():
    f = _make_ffbf()
    vec = np.ones(64, dtype=np.float32)
    f.add(vec)
    j = f.to_json()
    f2 = FFBF.from_json(j)
    assert abs(f.novelty(vec) - f2.novelty(vec)) < 1e-6
    assert f.window_len() == f2.window_len()


def test_ffbf_from_json_invalid_raises():
    with pytest.raises(ValueError):
        FFBF.from_json("not json")


def test_ffbf_save_load_roundtrip():
    f = _make_ffbf()
    f.add(np.ones(64, dtype=np.float32))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        f.save(path)
        f2 = FFBF.load(path)
        np.testing.assert_array_almost_equal(f.weights(), f2.weights())
    finally:
        os.unlink(path)


def test_ffbf_load_invalid_path_raises():
    with pytest.raises(ValueError):
        FFBF.load("/nonexistent/path/ffbf.json")
