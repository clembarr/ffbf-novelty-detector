import pytest
from ffbf import DecayMode, TickShape


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
