from importlib import resources
from pathlib import Path

from shapely import LineString, Point

import roseau.load_flow as rlf
import roseau.load_flow_single as rlfs

TEST_NETWORKS_PATH = Path(resources.files("roseau.load_flow_single")) / "tests" / "data" / "networks"


def generate_small_network() -> None:
    # Build a small network
    point1 = Point(-1.318375372111463, 48.64794139348595)
    point2 = Point(-1.320149235966572, 48.64971306653889)
    line_string = LineString([point1, point2])

    source_bus = rlfs.Bus(id="bus0", geometry=point1)
    load_bus = rlfs.Bus(id="bus1", geometry=point2)

    voltage = 20000.0 + 0.0j
    vs = rlfs.VoltageSource(id="vs", bus=source_bus, voltage=voltage)
    load = rlfs.PowerLoad(id="load", bus=load_bus, power=100)

    lp = rlfs.LineParameters(id="test", z_line=10)
    line = rlfs.Line(id="line", bus1=source_bus, bus2=load_bus, parameters=lp, length=1.0, geometry=line_string)

    en = rlfs.ElectricalNetwork(
        buses=[source_bus, load_bus],
        lines=[line],
        transformers=[],
        switches=[],
        loads=[load],
        sources=[vs],
    )
    en.solve_load_flow()
    en.to_json(TEST_NETWORKS_PATH / "small_network.json")


def generate_all_element_network(apl=None) -> None:
    # Buses
    bus0 = rlfs.Bus(id="bus0")
    bus1 = rlfs.Bus(id="bus1")
    bus2 = rlfs.Bus(id="bus2")
    bus3 = rlfs.Bus(id="bus3")
    bus4 = rlfs.Bus(id="bus4")

    # Voltage source
    voltage_source0 = rlfs.VoltageSource(id="voltage_source0", bus=bus0, voltage=20e3)

    # Line between bus0 and bus1 (with shunt)
    lp0 = rlfs.LineParameters.from_catalogue(name="U_AM_148", id="lp0", nb_phases=1)
    line0 = rlfs.Line(id="line0", bus1=bus0, bus2=bus1, parameters=lp0, length=rlf.Q_(1.5, "km"))

    # Transformer between bus1 and bus2
    tp0 = rlfs.TransformerParameters(id="630kVA", type="single", sn=630e3, uhv=20e3, ulv=400, z2=0.02, ym=1e-7)
    transformer0 = rlfs.Transformer(id="transformer0", bus1=bus1, bus2=bus2, parameters=tp0, tap=1.0)

    # Switch between the bus2 and the bus3
    switch0 = rlfs.Switch(id="switch0", bus1=bus2, bus2=bus3)

    # Line between bus3 and bus4 (without shunt)
    lp1_tmp = rlfs.LineParameters.from_catalogue(name="T_AL_75", id="lp1", nb_phases=1)
    lp1 = rlfs.LineParameters(
        id=lp1_tmp.id,
        z_line=lp1_tmp.z_line,
        y_shunt=None,  # <---- No shunt
        max_current=lp1_tmp.max_current,
        line_type=lp1_tmp.line_type,
        conductor_type=lp1_tmp.conductor_type,
        insulator_type=lp1_tmp.insulator_type,
        section=lp1_tmp.section,
    )
    line1 = rlfs.Line(id="line1", bus1=bus3, bus2=bus4, parameters=lp1, length=rlf.Q_(100, "m"))

    # Loads
    load0 = rlfs.PowerLoad(id="load0", bus=bus4, power=rlf.Q_(100 + 5j, "W"))
    load1 = rlfs.CurrentLoad(id="load1", bus=bus4, current=rlf.Q_(1 + 0.1j, "A"))
    load2 = rlfs.ImpedanceLoad(id="load2", bus=bus4, impedance=rlf.Q_(1, "ohm"))

    fp0 = rlfs.FlexibleParameter.constant()
    fp1 = rlfs.FlexibleParameter.p_max_u_consumption(
        u_min=rlf.Q_(380, "V"),
        u_down=rlf.Q_(385, "V"),
        s_max=rlf.Q_(150, "VA"),
    )
    fp2 = rlfs.FlexibleParameter.pq_u_consumption(
        up_min=rlf.Q_(380, "V"),
        up_down=rlf.Q_(385, "V"),
        uq_min=rlf.Q_(385, "V"),
        uq_down=rlf.Q_(390, "V"),
        uq_up=rlf.Q_(415, "V"),
        uq_max=rlf.Q_(420, "V"),
        s_max=rlf.Q_(150, "VA"),
    )
    fp3 = rlfs.FlexibleParameter.p_max_u_production(
        u_up=rlf.Q_(415, "V"), u_max=rlf.Q_(420, "V"), s_max=rlf.Q_(150, "VA")
    )
    fp4 = rlfs.FlexibleParameter.pq_u_production(
        up_up=rlf.Q_(415, "V"),
        up_max=rlf.Q_(420, "V"),
        uq_min=rlf.Q_(385, "V"),
        uq_down=rlf.Q_(390, "V"),
        uq_up=rlf.Q_(410, "V"),
        uq_max=rlf.Q_(415, "V"),
        s_max=rlf.Q_(150, "VA"),
    )
    fp5 = rlfs.FlexibleParameter.q_u(
        u_min=rlf.Q_(385, "V"),
        u_down=rlf.Q_(390, "V"),
        u_up=rlf.Q_(410, "V"),
        u_max=rlf.Q_(415, "V"),
        s_max=rlf.Q_(150, "VA"),
        q_min=rlf.Q_(-100, "VAr"),
        q_max=rlf.Q_(100, "VAr"),
    )
    load3 = rlfs.PowerLoad(id="load3", bus=bus4, power=rlf.Q_(100, "W"), flexible_param=fp0)
    load4 = rlfs.PowerLoad(id="load4", bus=bus4, power=rlf.Q_(100, "W"), flexible_param=fp1)
    load5 = rlfs.PowerLoad(id="load5", bus=bus4, power=rlf.Q_(100, "W"), flexible_param=fp2)
    load6 = rlfs.PowerLoad(id="load6", bus=bus4, power=rlf.Q_(-100, "W"), flexible_param=fp3)
    load7 = rlfs.PowerLoad(id="load7", bus=bus4, power=rlf.Q_(-100, "W"), flexible_param=fp4)
    load8 = rlfs.PowerLoad(id="load8", bus=bus4, power=rlf.Q_(-100, "W"), flexible_param=fp5)

    en = rlfs.ElectricalNetwork(
        buses=[bus0, bus1, bus2, bus3, bus4],
        lines=[line0, line1],
        transformers=[transformer0],
        switches=[switch0],
        loads=[load0, load1, load2, load3, load4, load5, load6, load7, load8],
        sources=[voltage_source0],
    )
    en.solve_load_flow()
    en.to_json(TEST_NETWORKS_PATH / "all_element_network.json")


if __name__ == "__main__":
    generate_small_network()
    generate_all_element_network()