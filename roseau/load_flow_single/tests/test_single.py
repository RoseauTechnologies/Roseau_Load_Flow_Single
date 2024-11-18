import numpy as np

import roseau.load_flow as rlf
import roseau.load_flow_single as rlfs


def test_single():
    # Single phase network
    voltage = 20e3
    source_bus = rlfs.Bus(id="source_bus")
    junction_bus = rlfs.Bus(id="junction_bus")
    load_bus = rlfs.Bus(id="bus1")
    vs = rlfs.VoltageSource(id="vs", bus=source_bus, voltage=voltage)
    load = rlfs.PowerLoad(id="load", bus=load_bus, power=9e6 + 3e6j)
    switch = rlfs.Switch(id="switch", bus1=source_bus, bus2=junction_bus)
    lp = rlfs.LineParameters.from_catalogue(name="U_AM_148", id="lp", nb_phases=1)
    line = rlfs.Line(id="line", bus1=junction_bus, bus2=load_bus, parameters=lp, length=1.0)

    en = rlfs.ElectricalNetwork.from_element(initial_bus=source_bus)
    en.solve_load_flow()

    # 3-phase balanced network
    ground_tri = rlf.Ground(id="ground")
    source_bus_tri = rlf.Bus(id="source_bus", phases="abcn")
    rlf.PotentialRef(id="pref", element=source_bus_tri)
    junction_bus_tri = rlf.Bus(id="junction_bus", phases="abcn")
    load_bus_tri = rlf.Bus(id="bus1", phases="abcn")
    vs_tri = rlf.VoltageSource(id="vs", bus=source_bus_tri, voltages=voltage / np.sqrt(3.0))
    load_tri = rlf.PowerLoad(id="load", bus=load_bus_tri, powers=[3e6 + 1e6j, 3e6 + 1e6j, 3e6 + 1e6j])
    switch_tri = rlf.Switch(id="switch", bus1=source_bus_tri, bus2=junction_bus_tri)
    lp_tri = rlf.LineParameters.from_catalogue(name="U_AM_148", id="lp", nb_phases=4)
    line_tri = rlf.Line(
        id="line", bus1=junction_bus_tri, bus2=load_bus_tri, parameters=lp_tri, length=1.0, ground=ground_tri
    )

    en_3 = rlf.ElectricalNetwork.from_element(initial_bus=source_bus_tri)
    en_3.solve_load_flow()

    # Check voltages
    assert np.isclose(source_bus.res_voltage, source_bus_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(load_bus.res_voltage, load_bus_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(vs.res_voltage, vs_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(load.res_voltage, load_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(line.res_voltages[0], line_tri.res_voltages[0][0] * np.sqrt(3.0))
    assert np.isclose(line.res_voltages[1], line_tri.res_voltages[1][0] * np.sqrt(3.0))
    assert np.isclose(switch.res_voltages[1], switch_tri.res_voltages[1][0] * np.sqrt(3.0))
    assert np.isclose(switch.res_voltages[1], switch_tri.res_voltages[1][0] * np.sqrt(3.0))

    # Check currents
    assert np.isclose(load.res_current, load_tri.res_currents[0])
    assert np.isclose(vs.res_current, vs_tri.res_currents[0])
    assert np.isclose(line.res_currents[0], line_tri.res_currents[0][0])
    assert np.isclose(line.res_currents[1], line_tri.res_currents[1][0])
    assert np.isclose(line.res_shunt_currents[0], line_tri.res_shunt_currents[0][0])
    assert np.isclose(line.res_shunt_currents[1], line_tri.res_shunt_currents[1][0])
    assert np.isclose(line.res_series_currents, line_tri.res_series_currents[0])
    assert np.isclose(line.res_currents[1], line_tri.res_currents[1][0])
    assert np.isclose(switch.res_currents[0], switch_tri.res_currents[0][0])
    assert np.isclose(switch.res_currents[1], switch_tri.res_currents[1][0])

    # Check powers
    assert np.isclose(load.res_power, np.sum(load_tri.res_powers))
    assert np.isclose(vs.res_power, np.sum(vs_tri.res_powers))
    assert np.isclose(line.res_powers[0], np.sum(line_tri.res_powers[0]))
    assert np.isclose(line.res_powers[1], np.sum(line_tri.res_powers[1]))
    assert np.isclose(line.res_series_power_losses, np.sum(line_tri.res_series_power_losses))
    assert np.isclose(line.res_shunt_power_losses, np.sum(line_tri.res_shunt_power_losses))
    assert np.isclose(line.res_power_losses, np.sum(line_tri.res_power_losses))
    assert np.isclose(switch.res_powers[0], np.sum(switch_tri.res_powers[0]))
    assert np.isclose(switch.res_powers[1], np.sum(switch_tri.res_powers[1]))


def test_single_transformer(three_phases_transformer_type):
    # Single phase network
    voltage = 20e3
    source_bus = rlfs.Bus(id="source_bus")
    load_bus = rlfs.Bus(id="bus1")
    vs = rlfs.VoltageSource(id="vs", bus=source_bus, voltage=voltage)
    load = rlfs.PowerLoad(id="load", bus=load_bus, power=90e3 + 30e3j)
    tr = rlfs.TransformerParameters(
        id="TP", type=three_phases_transformer_type, up=20e3, us=400, sn=630, z2=0.0005, ym=0.001
    )
    transformer = rlfs.Transformer(id="transformer", bus1=source_bus, bus2=load_bus, parameters=tr)

    en = rlfs.ElectricalNetwork.from_element(initial_bus=source_bus)
    en.solve_load_flow()

    # 3-phase balanced network
    source_bus_tri = rlf.Bus(id="source_bus", phases="abcn")
    rlf.PotentialRef(id="pref", element=source_bus_tri)
    vs_tri = rlf.VoltageSource(id="vs", bus=source_bus_tri, voltages=voltage / np.sqrt(3.0))

    load_bus_tri = rlf.Bus(id="bus1", phases="abcn")
    rlf.PotentialRef(id="pref2", element=load_bus_tri)
    load_tri = rlf.PowerLoad(id="load", bus=load_bus_tri, powers=[30e3 + 10e3j, 30e3 + 10e3j, 30e3 + 10e3j])

    tr_tri = rlf.TransformerParameters(
        id="TP", type=three_phases_transformer_type, up=20e3, us=400, sn=630, z2=0.0005, ym=0.001
    )
    transformer_tri = rlf.Transformer(id="transformer", bus1=source_bus_tri, bus2=load_bus_tri, parameters=tr_tri)

    en_tri = rlf.ElectricalNetwork.from_element(initial_bus=source_bus_tri)
    en_tri.solve_load_flow()

    # Check voltages
    assert np.isclose(source_bus.res_voltage, source_bus_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(load_bus.res_voltage, load_bus_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(vs.res_voltage, vs_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(load.res_voltage, load_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(transformer.res_voltages[0], source_bus_tri.res_voltages[0] * np.sqrt(3.0))
    assert np.isclose(transformer.res_voltages[1], load_bus_tri.res_voltages[0] * np.sqrt(3.0))

    # Check currents
    assert np.isclose(load.res_current, load_tri.res_currents[0])
    assert np.isclose(vs.res_current, vs_tri.res_currents[0])
    assert np.isclose(transformer.res_currents[0], transformer_tri.res_currents[0][0])
    assert np.isclose(transformer.res_currents[1], transformer_tri.res_currents[1][0])

    # Check powers
    assert np.isclose(load.res_power, np.sum(load_tri.res_powers))
    assert np.isclose(vs.res_power, np.sum(vs_tri.res_powers))
    assert np.isclose(transformer.res_powers[0], np.sum(transformer_tri.res_powers[0]))
    assert np.isclose(transformer.res_powers[1], np.sum(transformer_tri.res_powers[1]))
