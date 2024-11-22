import numpy as np
import pytest

from roseau.load_flow import Q_, RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow_single.models import Bus, Transformer, TransformerParameters


def test_max_power():
    tp = TransformerParameters.from_catalogue(name="FT_Standard_Standard_100kVA")
    assert tp.sn == Q_(100, "kVA")

    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    transformer = Transformer(id="transformer", bus1=bus1, bus2=bus2, parameters=tp, max_loading=1)
    assert transformer.sn == Q_(100, "kVA")
    assert transformer.max_power == Q_(100, "kVA")

    transformer.max_loading = 0.5
    assert transformer.sn == Q_(100, "kVA")
    assert transformer.max_power == Q_(50, "kVA")


def test_max_loading():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    tp = TransformerParameters.from_open_and_short_circuit_tests(
        id="tp", psc=1350.0, p0=145.0, i0=1.8 / 100, us=400, up=20000, sn=50 * 1e3, vsc=4 / 100, type="yzn11"
    )
    transformer = Transformer(id="transformer", bus1=bus1, bus2=bus2, parameters=tp)

    # Value must be positive
    with pytest.raises(RoseauLoadFlowException) as e:
        transformer.max_loading = -1
    assert e.value.msg == "Maximum loading must be positive: -1 was provided."
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_MAX_LOADING_VALUE

    with pytest.raises(RoseauLoadFlowException) as e:
        transformer.max_loading = 0
    assert e.value.msg == "Maximum loading must be positive: 0 was provided."
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_MAX_LOADING_VALUE


def test_res_violated():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    tp = TransformerParameters.from_open_and_short_circuit_tests(
        id="tp", psc=1350.0, p0=145.0, i0=1.8 / 100, us=400, up=20000, sn=50 * 1e3, vsc=4 / 100, type="yzn11"
    )
    transformer = Transformer(id="transformer", bus1=bus1, bus2=bus2, parameters=tp)

    bus1._res_potential = 20_000
    bus2._res_potential = 230
    transformer._res_currents = 0.8, -65

    # Default value
    assert transformer.max_loading == Q_(1, "")
    assert transformer.res_violated is False

    # No constraint violated
    transformer.max_loading = 1
    assert transformer.res_violated is False
    assert np.allclose(transformer.res_loading, 0.8 * 20 * 3 / 50)

    # Two violations
    transformer.max_loading = 4 / 5
    assert transformer.res_violated is True
    assert np.allclose(transformer.res_loading, 0.8 * 20 * 3 / 40)

    # Primary side violation
    transformer.max_loading = Q_(45, "%")
    assert transformer.res_violated is True
    assert np.allclose(transformer.res_loading, 0.8 * 20 * 3 / (50 * 0.45))

    # Secondary side violation
    transformer.max_loading = 1
    transformer._res_currents = 0.8, -80
    assert transformer.res_violated is True
    assert np.allclose(transformer.res_loading, 80 * 230 * 3 / 50_000)


def test_transformer_results():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    tp = TransformerParameters.from_open_and_short_circuit_tests(
        id="tp", psc=1350, p0=145, i0=0.018, us=400, up=20e3, sn=50e3, vsc=0.04, type="yzn11"
    )
    transformer = Transformer(id="transformer", bus1=bus1, bus2=bus2, parameters=tp)

    bus1._res_potential = 20_000
    bus2._res_potential = 230
    transformer._res_currents = np.complex128(0.8 + 0j), np.complex128(-65 + 0j)

    res_p1, res_p2 = (p.m for p in transformer.res_powers)

    np.testing.assert_allclose(
        res_p1, transformer.res_voltages[0].m / np.sqrt(3.0) * transformer.res_currents[0].m.conj() * 3.0
    )
    np.testing.assert_allclose(
        res_p2, transformer.res_voltages[1].m / np.sqrt(3.0) * transformer.res_currents[1].m.conj() * 3.0
    )

    expected_total_losses = res_p1 + res_p2
    actual_total_losses = transformer.res_power_losses.m
    assert np.isscalar(actual_total_losses)
    np.testing.assert_allclose(actual_total_losses, expected_total_losses)
