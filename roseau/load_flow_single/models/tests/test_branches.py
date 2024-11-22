import numpy as np

from roseau.load_flow_single import Bus, Line, LineParameters


def test_res_branches_voltages():
    # Same phases
    bus1 = Bus("bus1")
    bus2 = Bus("bus2")
    lp = LineParameters("lp", z_line=1)
    line = Line("line", bus1, bus2, length=1, parameters=lp)
    bus1._res_potential = 230.0 + 0.0j
    bus2._res_potential = 225.47405027 + 0.0j
    line_v1, line_v2 = line.res_voltages
    assert np.isclose(line_v1, bus1.res_voltage)
    assert np.isclose(line_v2, bus2.res_voltage)


def test_powers_equal(network_with_results):
    line: Line = network_with_results.lines["line"]
    vs = network_with_results.sources["vs"]
    pl = network_with_results.loads["load"]
    power1, power2 = line.res_powers
    assert np.allclose(power1, -vs.res_power)
    assert np.allclose(power2, -pl.res_power)
    assert np.allclose(power1 + power2, line.res_power_losses)
