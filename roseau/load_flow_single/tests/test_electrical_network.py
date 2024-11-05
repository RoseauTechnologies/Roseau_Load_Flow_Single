import itertools as it
import json
import re
import warnings
from contextlib import contextmanager

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.units import Q_
from roseau.load_flow.utils import LoadTypeDtype
from roseau.load_flow_single.models import (
    Bus,
    CurrentLoad,
    FlexibleParameter,
    ImpedanceLoad,
    Line,
    LineParameters,
    PowerLoad,
    Switch,
    Transformer,
    TransformerParameters,
    VoltageSource,
)
from roseau.load_flow_single.network import ElectricalNetwork

# The following networks are generated using the scripts/generate_test_networks.py script


@pytest.fixture
def all_element_network(test_networks_path) -> ElectricalNetwork:
    # Load the network from the JSON file (without results)
    return ElectricalNetwork.from_json(path=test_networks_path / "all_element_network.json", include_results=False)


@pytest.fixture
def all_element_network_with_results(test_networks_path) -> ElectricalNetwork:
    # Load the network from the JSON file (with results, no need to invoke the solver)
    return ElectricalNetwork.from_json(path=test_networks_path / "all_element_network.json", include_results=True)


@pytest.fixture
def small_network(test_networks_path) -> ElectricalNetwork:
    # Load the network from the JSON file (without results)
    return ElectricalNetwork.from_json(path=test_networks_path / "small_network.json", include_results=False)


@pytest.fixture
def small_network_with_results(test_networks_path) -> ElectricalNetwork:
    # Load the network from the JSON file (with results, no need to invoke the solver)
    return ElectricalNetwork.from_json(path=test_networks_path / "small_network.json", include_results=True)


@contextmanager
def check_result_warning(expected_message: str | re.Pattern[str]):
    with warnings.catch_warnings(record=True) as records:
        yield
    assert len(records) == 1
    assert re.match(expected_message, records[0].message.args[0])
    assert records[0].category is UserWarning


def test_connect_and_disconnect():
    vn = 400
    source_bus = Bus(id="source")
    load_bus = Bus(id="load bus")
    vs = VoltageSource(id="vs", bus=source_bus, voltage=vn)
    load = PowerLoad(id="power load", bus=load_bus, power=100 + 0j)
    lp = LineParameters(id="test", z_line=1.0)
    line = Line(id="line", bus1=source_bus, bus2=load_bus, parameters=lp, length=10)
    en = ElectricalNetwork.from_element(source_bus)

    # Connection of a new connected component
    load_bus2 = Bus(id="load_bus2")
    tp = TransformerParameters(id="630kVA", type="single", sn=630e3, uhv=20e3, ulv=400, z2=0.02, ym=1e-7)
    Transformer(id="transfo", bus1=load_bus, bus2=load_bus2, parameters=tp)

    # Disconnection of a load
    assert load.network == en
    load.disconnect()
    assert load.network is None
    assert load.bus is None
    with pytest.raises(RoseauLoadFlowException) as e:
        load.to_dict()
    assert e.value.msg == "The load 'power load' is disconnected and cannot be used anymore."
    assert e.value.code == RoseauLoadFlowExceptionCode.DISCONNECTED_ELEMENT
    new_load = PowerLoad(id="power load", bus=load_bus, power=100 + 0j)
    assert new_load.network == en

    # Disconnection of a source
    assert vs.network == en
    vs.disconnect()
    assert vs.network is None
    assert vs.bus is None
    with pytest.raises(RoseauLoadFlowException) as e:
        vs.to_dict()
    assert e.value.msg == "The voltage source 'vs' is disconnected and cannot be used anymore."
    assert e.value.code == RoseauLoadFlowExceptionCode.DISCONNECTED_ELEMENT

    # Adding unknown element
    with pytest.raises(RoseauLoadFlowException) as e:
        en._connect_element(3)
    assert "Unknown element" in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT

    # Remove line => impossible
    with pytest.raises(RoseauLoadFlowException) as e:
        en._disconnect_element(line)
    assert e.value.msg == (
        "<Line: id='line', bus1='source', bus2='load bus'> is a Line and it cannot be disconnected from a network."
    )
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT


def test_recursive_connect_disconnect():
    vn = 400 / np.sqrt(3)
    source_bus = Bus(id="source")
    load_bus = Bus(id="load bus")
    VoltageSource(id="vs", bus=source_bus, voltage=vn)
    load = PowerLoad(id="power load", bus=load_bus, power=100 + 0j)
    lp = LineParameters(id="test", z_line=1.0)
    line = Line(id="line", bus1=source_bus, bus2=load_bus, parameters=lp, length=10)
    en = ElectricalNetwork.from_element(source_bus)

    # Create new elements (without connecting them to the existing network)
    new_bus2 = Bus(id="new_bus2")
    new_load2 = PowerLoad(id="new_load2", bus=new_bus2, power=Q_(100, "VA"))
    new_bus = Bus(id="new_bus")
    new_load = PowerLoad(id="new_load", bus=new_bus, power=Q_(100, "VA"))
    lp = LineParameters(id="U_AL_240_without_shunt", z_line=Q_(0.1, "ohm/km"), y_shunt=None)
    new_line2 = Line(
        id="new_line2",
        bus1=new_bus2,
        bus2=new_bus,
        parameters=lp,
        length=0.5,
    )
    assert new_bus.network is None
    assert new_bus.id not in en.buses
    assert new_load.network is None
    assert new_load.id not in en.loads
    assert new_bus2.network is None
    assert new_bus2.id not in en.buses
    assert new_line2.network is None
    assert new_line2.id not in en.lines
    assert new_load2.network is None
    assert new_load2.id not in en.loads

    # Connect them to the first part of the network using a Line
    new_line = Line(
        id="new_line",
        bus1=new_bus,  # new part of the network
        bus2=load_bus,  # first part of the network
        parameters=lp,
        length=0.5,
    )
    assert load_bus._connected_elements == [load, line, new_line]
    assert new_bus.network == en
    assert new_bus._connected_elements == [new_load, new_line2, new_line]
    assert new_bus.id in en.buses
    assert new_line.network == en
    assert new_line._connected_elements == [new_bus, load_bus]
    assert new_line.id in en.lines
    assert new_load.network == en
    assert new_load._connected_elements == [new_bus]
    assert new_load.id in en.loads
    assert new_bus2.network == en
    assert new_bus2._connected_elements == [new_load2, new_line2]
    assert new_bus2.id in en.buses
    assert new_line2.network == en
    assert new_line2._connected_elements == [new_bus2, new_bus]
    assert new_line2.id in en.lines
    assert new_load2.network == en
    assert new_load2._connected_elements == [new_bus2]
    assert new_load2.id in en.loads

    # Disconnect a load
    new_load.disconnect()
    assert load_bus._connected_elements == [load, line, new_line]
    assert new_bus.network == en
    assert new_bus._connected_elements == [new_line2, new_line]
    assert new_bus.id in en.buses
    assert new_line.network == en
    assert new_line._connected_elements == [new_bus, load_bus]
    assert new_line.id in en.lines
    assert new_load.network is None
    assert new_load._connected_elements == []
    assert new_load.id not in en.loads
    assert new_bus2.network == en
    assert new_bus2._connected_elements == [new_load2, new_line2]
    assert new_bus2.id in en.buses
    assert new_line2.network == en
    assert new_line2._connected_elements == [new_bus2, new_bus]
    assert new_line2.id in en.lines
    assert new_load2.network == en
    assert new_load2._connected_elements == [new_bus2]
    assert new_load2.id in en.loads


def test_bad_networks():
    # No source
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    lp = LineParameters(id="test", z_line=1.0)
    line = Line(id="line", bus1=bus1, bus2=bus2, parameters=lp, length=10)
    with pytest.raises(RoseauLoadFlowException) as e:
        ElectricalNetwork.from_element(bus1)
    assert e.value.msg == "There is no voltage source provided in the network, you must provide at least one."
    assert e.value.code == RoseauLoadFlowExceptionCode.NO_VOLTAGE_SOURCE

    # No network has been assigned
    assert bus1.network is None
    assert line.network is None

    # Bad constructor
    bus0 = Bus(id="bus0")
    vs = VoltageSource(id="vs", bus=bus0, voltage=20e3)
    switch = Switch(id="switch", bus1=bus0, bus2=bus1)
    with pytest.raises(RoseauLoadFlowException) as e:
        ElectricalNetwork(
            buses=[bus0, bus1],  # no bus2
            lines=[line],
            transformers=[],
            switches=[switch],
            loads=[],
            sources=[vs],
        )
    assert "but has not been added to the network. It must be added with 'connect'." in e.value.msg
    assert bus2.id in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.UNKNOWN_ELEMENT

    # No network has been assigned
    assert bus0.network is None
    assert bus1.network is None
    assert line.network is None
    assert switch.network is None
    assert vs.network is None

    # No potential reference
    bus3 = Bus(id="bus3")
    tp = TransformerParameters.from_open_and_short_circuit_tests(
        id="t", type="single", uhv=20000, ulv=400, sn=160 * 1e3, p0=460, i0=2.3 / 100, psc=2350, vsc=4 / 100
    )
    t = Transformer(id="transfo", bus1=bus2, bus2=bus3, parameters=tp)

    # No network has been assigned
    assert bus0.network is None
    assert bus1.network is None
    assert line.network is None
    assert switch.network is None
    assert vs.network is None
    assert bus3.network is None
    assert t.network is None

    # Bad ID
    src_bus = Bus(id="sb")
    load_bus = Bus(id="lb")
    lp = LineParameters(id="test", z_line=1.0)
    line = Line(id="ln", bus1=src_bus, bus2=load_bus, parameters=lp, length=10)
    vs = VoltageSource(id="vs", bus=src_bus, voltage=230)
    load = PowerLoad(id="pl", bus=load_bus, power=1000)
    with pytest.raises(RoseauLoadFlowException) as e:
        ElectricalNetwork(
            buses={"foo": src_bus, "lb": load_bus},  # <-- ID of src_bus is wrong
            lines={"ln": line},
            transformers={},
            switches={},
            loads={"pl": load},
            sources={"vs": vs},
        )
    assert e.value.msg == "Bus ID 'sb' does not match its key in the dictionary 'foo'."
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_BUS_ID


def test_poorly_connected_elements():
    bus1 = Bus(id="b1")
    bus2 = Bus(id="b2")
    bus3 = Bus(id="b3")
    bus4 = Bus(id="b4")
    lp = LineParameters.from_catalogue(name="U_AL_150", nb_phases=1)
    line1 = Line(id="l1", bus1=bus1, bus2=bus2, parameters=lp, length=1)
    line2 = Line(id="l2", bus1=bus3, bus2=bus4, parameters=lp, length=1)
    vs = VoltageSource(id="vs1", bus=bus1, voltage=20e3)
    with pytest.raises(RoseauLoadFlowException) as e:
        ElectricalNetwork(
            buses=[bus1, bus2, bus3, bus4],
            lines=[line1, line2],
            transformers={},
            switches={},
            loads={},
            sources=[vs],
        )
    assert (
        e.value.msg
        == "The elements [\"Bus('b3'), Bus('b4'), Line('l2')\"] are not electrically connected to a voltage source."
    )
    assert e.value.code == RoseauLoadFlowExceptionCode.POORLY_CONNECTED_ELEMENT


def test_invalid_element_overrides():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    lp = LineParameters(id="lp", z_line=1.0)
    Line(id="line", bus1=bus1, bus2=bus2, parameters=lp, length=1)
    VoltageSource(id="source", bus=bus1, voltage=230)
    old_load = PowerLoad(id="load", bus=bus2, power=1000)
    ElectricalNetwork.from_element(bus1)

    # Case of a different load type on a different bus
    with pytest.raises(RoseauLoadFlowException) as e:
        CurrentLoad(id="load", bus=bus1, current=1)
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT
    assert e.value.msg == (
        "A load of ID 'load' is already connected to the network. Disconnect the old load first "
        "if you meant to replace it."
    )

    # Disconnect the old element first: OK
    old_load.disconnect()
    ImpedanceLoad(id="load", bus=bus1, impedance=500)

    # Case of a source (also suggests disconnecting first)
    with pytest.raises(RoseauLoadFlowException) as e:
        VoltageSource(id="source", bus=bus2, voltage=230)
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT
    assert e.value.msg == (
        "A source of ID 'source' is already connected to the network. Disconnect the old source first "
        "if you meant to replace it."
    )


def test_frame(small_network: ElectricalNetwork):
    # Buses
    buses_gdf = small_network.buses_frame
    assert isinstance(buses_gdf, gpd.GeoDataFrame)
    assert buses_gdf.shape == (2, 3)
    assert buses_gdf.columns.tolist() == ["min_voltage", "max_voltage", "geometry"]
    assert buses_gdf.index.name == "id"

    # Lines
    lines_gdf = small_network.lines_frame
    assert isinstance(lines_gdf, gpd.GeoDataFrame)
    assert lines_gdf.shape == (1, 6)
    assert lines_gdf.columns.tolist() == [
        "bus1_id",
        "bus2_id",
        "parameters_id",
        "length",
        "max_current",
        "geometry",
    ]

    # Transformers
    transformers_gdf = small_network.transformers_frame
    assert isinstance(transformers_gdf, gpd.GeoDataFrame)
    assert transformers_gdf.shape == (0, 5)
    assert transformers_gdf.columns.tolist() == [
        "bus1_id",
        "bus2_id",
        "parameters_id",
        "max_power",
        "geometry",
    ]
    assert transformers_gdf.index.name == "id"

    # Switches
    switches_gdf = small_network.switches_frame
    assert isinstance(switches_gdf, gpd.GeoDataFrame)
    assert switches_gdf.shape == (0, 3)
    assert switches_gdf.columns.tolist() == ["bus1_id", "bus2_id", "geometry"]

    # Loads
    loads_df = small_network.loads_frame
    assert isinstance(loads_df, pd.DataFrame)
    assert loads_df.shape == (1, 3)
    assert loads_df.columns.tolist() == ["type", "bus_id", "flexible"]
    assert loads_df.index.name == "id"

    # Sources
    sources_df = small_network.sources_frame
    assert isinstance(sources_df, pd.DataFrame)
    assert sources_df.shape == (1, 1)
    assert sources_df.columns.tolist() == ["bus_id"]
    assert sources_df.index.name == "id"


def test_empty_network():
    with pytest.raises(RoseauLoadFlowException) as exc_info:
        ElectricalNetwork(
            buses={},
            lines={},
            transformers={},
            switches={},
            loads={},
            sources={},
        )
    assert exc_info.value.code == RoseauLoadFlowExceptionCode.EMPTY_NETWORK
    assert exc_info.value.msg == "Cannot create a network without elements."


def test_buses_voltages(small_network_with_results):
    assert isinstance(small_network_with_results, ElectricalNetwork)
    en = small_network_with_results
    en.buses["bus0"].max_voltage = 21_000
    en.buses["bus1"].min_voltage = 20_000

    voltage_records = [
        {
            "bus_id": "bus0",
            "voltage": 20000.0 + 0.0j,
            "min_voltage": np.nan,
            "max_voltage": 21000,
            "violated": False,
        },
        {
            "bus_id": "bus1",
            "voltage": 19999.949999875 + 0.0j,
            "min_voltage": 20000,
            "max_voltage": np.nan,
            "violated": True,
        },
    ]

    buses_voltages = en.res_buses_voltages
    expected_buses_voltages = (
        pd.DataFrame.from_records(voltage_records)
        .astype(
            {
                "bus_id": str,
                "voltage": complex,
                "min_voltage": float,
                "max_voltage": float,
                "violated": pd.BooleanDtype(),
            }
        )
        .set_index("bus_id")
    )

    assert isinstance(buses_voltages, pd.DataFrame)
    assert buses_voltages.shape == (2, 4)
    assert buses_voltages.index.names == ["bus_id"]
    assert list(buses_voltages.columns) == ["voltage", "min_voltage", "max_voltage", "violated"]
    assert_frame_equal(buses_voltages, expected_buses_voltages, check_exact=False)


def test_to_from_dict_roundtrip(small_network: ElectricalNetwork):
    net_dict = small_network.to_dict()
    new_net = ElectricalNetwork.from_dict(net_dict)
    assert_frame_equal(small_network.buses_frame, new_net.buses_frame)
    assert_frame_equal(small_network.lines_frame, new_net.lines_frame)
    assert_frame_equal(small_network.transformers_frame, new_net.transformers_frame)
    assert_frame_equal(small_network.switches_frame, new_net.switches_frame)
    assert_frame_equal(small_network.loads_frame, new_net.loads_frame)
    assert_frame_equal(small_network.sources_frame, new_net.sources_frame)


def test_network_elements(small_network: ElectricalNetwork):
    # Add a line to the network ("bus2" constructor belongs to the network)
    bus1 = small_network.buses["bus1"]
    bus2 = Bus(id="bus2")
    assert bus2.network is None
    lp = LineParameters(id="test", z_line=10 * 1.0)
    l2 = Line(id="line2", bus1=bus2, bus2=bus1, parameters=lp, length=Q_(0.3, "km"))
    assert l2.network == small_network
    assert bus2.network == small_network

    # Add a switch ("bus1" constructor belongs to the network)
    bus3 = Bus(id="bus3")
    assert bus3.network is None
    s = Switch(id="switch", bus1=bus2, bus2=bus3)
    assert s.network == small_network
    assert bus3.network == small_network

    # Create a second network
    bus_vs = Bus(id="bus_vs")
    VoltageSource(id="vs2", bus=bus_vs, voltage=15e3)
    small_network_2 = ElectricalNetwork.from_element(initial_bus=bus_vs)

    # Connect the two networks
    with pytest.raises(RoseauLoadFlowException) as e:
        Switch(id="switch2", bus1=bus2, bus2=bus_vs)
    assert e.value.msg == "The Bus 'bus_vs' is already assigned to another network."
    assert e.value.code == RoseauLoadFlowExceptionCode.SEVERAL_NETWORKS

    # Every object have their good network after this failure
    for element in it.chain(
        small_network.buses.values(),
        small_network.lines.values(),
        small_network.transformers.values(),
        small_network.switches.values(),
        small_network.loads.values(),
    ):
        assert element.network == small_network
    for element in it.chain(
        small_network_2.buses.values(),
        small_network_2.lines.values(),
        small_network_2.transformers.values(),
        small_network_2.switches.values(),
        small_network_2.loads.values(),
    ):
        assert element.network == small_network_2


def test_network_results_warning(small_network, small_network_with_results, recwarn):  # noqa: C901
    en = small_network
    # network well-defined using the constructor
    for bus in en.buses.values():
        assert bus.network == en
    for load in en.loads.values():
        assert load.network == en
    for source in en.sources.values():
        assert source.network == en
    for line in en.lines.values():
        assert line.network == en
    for transformer in en.transformers.values():
        assert transformer.network == en
    for switch in en.switches.values():
        assert switch.network == en

    # All the results function raises an exception
    result_field_names_dict = {
        "buses": ("res_potential", "res_voltage", "res_violated"),
        "lines": (
            "res_currents",
            "res_violated",
            "res_voltage",
            "res_power_losses",
            "res_potentials",
            "res_powers",
            "res_series_currents",
            "res_series_power_losses",
            "res_shunt_current",
            "res_shunt_power_losses",
        ),
        "transformers": (
            "res_currents",
            "res_powers",
            "res_potentials",
            "res_power_losses",
            "res_violated",
            "res_voltages",
        ),
        "switches": ("res_currents", "res_potentials", "res_powers", "res_voltages"),
        "loads": ("res_current", "res_power", "res_potential", "res_voltage"),
        "sources": ("res_current", "res_potential", "res_power"),
    }
    for bus in en.buses.values():
        for result_field_name in result_field_names_dict["buses"]:
            if result_field_name == "res_violated" and bus.min_voltage is None and bus.max_voltage is None:
                continue  # No min or max voltages so no call to results
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(bus, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
    for line in en.lines.values():
        for result_field_name in result_field_names_dict["lines"]:
            if result_field_name == "res_violated":
                continue  # No max_currents
            if not line.with_shunt and "shunt" in result_field_name:
                continue  # No results if no shunt
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(line, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
    for transformer in en.transformers.values():
        for result_field_name in result_field_names_dict["transformers"]:
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(transformer, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
    for switch in en.switches.values():
        for result_field_name in result_field_names_dict["switches"]:
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(switch, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
    for load in en.loads.values():
        for result_field_name in result_field_names_dict["loads"]:
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(load, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
        if load.is_flexible and isinstance(load, PowerLoad):
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = load.res_flexible_powers
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
    for source in en.sources.values():
        for result_field_name in result_field_names_dict["sources"]:
            with pytest.raises(RoseauLoadFlowException) as e:
                _ = getattr(source, result_field_name)
            assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN

    # Network with results
    en = small_network_with_results

    # No warning when getting results (they are up-to-date)
    recwarn.clear()
    for bus in en.buses.values():
        for result_field_name in result_field_names_dict["buses"]:
            if result_field_name == "res_violated" and bus.min_voltage is None and bus.max_voltage is None:
                continue  # No min or max voltages so no call to results
            _ = getattr(bus, result_field_name)
    for line in en.lines.values():
        for result_field_name in result_field_names_dict["lines"]:
            if result_field_name == "res_violated":
                continue  # No max_currents
            if not line.with_shunt and "shunt" in result_field_name:
                continue  # No results if no shunt
            _ = getattr(line, result_field_name)
    for transformer in en.transformers.values():
        for result_field_name in result_field_names_dict["transformers"]:
            _ = getattr(transformer, result_field_name)
    for switch in en.switches.values():
        for result_field_name in result_field_names_dict["switches"]:
            _ = getattr(switch, result_field_name)
    for load in en.loads.values():
        for result_field_name in result_field_names_dict["loads"]:
            _ = getattr(load, result_field_name)
        if load.is_flexible and isinstance(load, PowerLoad):
            _ = load.res_flexible_powers
    for source in en.sources.values():
        for result_field_name in result_field_names_dict["sources"]:
            _ = getattr(source, result_field_name)
    assert len(recwarn) == 0

    # Modify something
    load = en.loads["load"]
    load.power = 200

    # Ensure that a warning is raised no matter which result is requested
    expected_message = (
        r"The results of \w+ '\w+' may be outdated. Please re-run a load flow to ensure the validity of results."
    )
    for bus in en.buses.values():
        for result_field_name in result_field_names_dict["buses"]:
            if result_field_name == "res_violated" and bus.min_voltage is None and bus.max_voltage is None:
                continue  # No min or max voltages so no call to results
            with check_result_warning(expected_message=expected_message):
                _ = getattr(bus, result_field_name)
    for line in en.lines.values():
        for result_field_name in result_field_names_dict["lines"]:
            if result_field_name == "res_violated":
                continue  # No max_currents
            if not line.with_shunt and "shunt" in result_field_name:
                continue  # No results if no shunt
            with check_result_warning(expected_message=expected_message):
                _ = getattr(line, result_field_name)
    for transformer in en.transformers.values():
        for result_field_name in result_field_names_dict["transformers"]:
            with check_result_warning(expected_message=expected_message):
                _ = getattr(transformer, result_field_name)
    for switch in en.switches.values():
        for result_field_name in result_field_names_dict["switches"]:
            with check_result_warning(expected_message=expected_message):
                _ = getattr(switch, result_field_name)
    for load in en.loads.values():
        for result_field_name in result_field_names_dict["loads"]:
            with check_result_warning(expected_message=expected_message):
                _ = getattr(load, result_field_name)
        if load.is_flexible and isinstance(load, PowerLoad):
            with check_result_warning(expected_message=expected_message):
                _ = load.res_flexible_powers
    for source in en.sources.values():
        for result_field_name in result_field_names_dict["sources"]:
            with check_result_warning(expected_message=expected_message):
                _ = getattr(source, result_field_name)

    # Ensure that a single warning is raised when having a data frame result
    expected_message = (
        "The results of this network may be outdated. Please re-run a load flow to ensure the validity of results."
    )
    with check_result_warning(expected_message=expected_message):
        _ = en.res_buses
    with check_result_warning(expected_message=expected_message):
        _ = en.res_buses_voltages
    with check_result_warning(expected_message=expected_message):
        _ = en.res_lines
    with check_result_warning(expected_message=expected_message):
        _ = en.res_transformers
    with check_result_warning(expected_message=expected_message):
        _ = en.res_switches
    with check_result_warning(expected_message=expected_message):
        _ = en.res_loads
    with check_result_warning(expected_message=expected_message):
        _ = en.res_sources
    with check_result_warning(expected_message=expected_message):
        _ = en.res_loads_flexible_powers


def test_network_results_error(small_network):
    en = small_network

    # Test all results
    for attr_name in dir(en):
        if not attr_name.startswith("res_"):
            continue
        with pytest.raises(RoseauLoadFlowException) as e:
            getattr(en, attr_name)
        assert e.value.code == RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN
        assert e.value.msg == "The load flow results are not available because the load flow has not been run yet."


def test_load_flow_results_frames(small_network_with_results):
    en = small_network_with_results
    en.buses["bus0"].min_voltage = 21_000

    # Buses results
    expected_res_buses = (
        pd.DataFrame.from_records(
            [
                {"bus_id": "bus0", "potential": 20000 + 2.89120338e-18j},
                {"bus_id": "bus1", "potential": 19999.949999875 + 2.891196e-18j},
            ]
        )
        .astype({"bus_id": object, "potential": complex})
        .set_index("bus_id")
    )
    assert_frame_equal(en.res_buses, expected_res_buses, rtol=1e-5)

    # Buses voltages results
    expected_res_buses_voltages = (
        pd.DataFrame.from_records(
            [
                {
                    "bus_id": "bus0",
                    "voltage": (20000 + 2.89120e-18j) - (-1.34764e-12 + 2.89120e-18j),
                    "min_voltage": 21_000,
                    "max_voltage": np.nan,
                    "violated": True,
                },
                {
                    "bus_id": "bus1",
                    "voltage": (19999.94999 + 2.89119e-18j) - (0j),
                    "min_voltage": np.nan,
                    "max_voltage": np.nan,
                    "violated": None,
                },
            ]
        )
        .astype(
            {
                "bus_id": object,
                "voltage": complex,
                "min_voltage": float,
                "max_voltage": float,
                "violated": pd.BooleanDtype(),
            }
        )
        .set_index(
            "bus_id",
        )
    )
    assert_frame_equal(en.res_buses_voltages, expected_res_buses_voltages, rtol=1e-5)

    # Transformers results
    expected_res_transformers = (
        pd.DataFrame.from_records(
            data=[],
            columns=[
                "transformer_id",
                "current1",
                "current2",
                "power1",
                "power2",
                "potential1",
                "potential2",
                "max_power",
                "violated",
            ],
        )
        .astype(
            {
                "transformer_id": object,
                "current1": complex,
                "current2": complex,
                "power1": complex,
                "power2": complex,
                "potential1": complex,
                "potential2": complex,
                "max_power": float,
                "violated": pd.BooleanDtype(),
            }
        )
        .set_index("transformer_id")
    )
    assert_frame_equal(en.res_transformers, expected_res_transformers)

    # Lines results
    expected_res_lines_records = [
        {
            "line_id": "line",
            "current1": 0.00500 + 7.22799e-25j,
            "current2": -0.00500 - 7.22799e-25j,
            "power1": (20000 + 2.89120e-18j) * (0.00500 + 7.22799e-25j).conjugate(),
            "power2": (19999.94999 + 2.89119e-18j) * (-0.00500 - 7.22799e-25j).conjugate(),
            "potential1": 20000 + 2.89120e-18j,
            "potential2": 19999.94999 + 2.89119e-18j,
            "series_losses": (
                (20000 + 2.89120e-18j) * (0.00500 + 7.22799e-25j).conjugate()
                + (19999.94999 + 2.89119e-18j) * (-0.00500 - 7.22799e-25j).conjugate()
            ),
            "series_current": 0.00500 + 7.22799e-25j,
            "max_current": np.nan,
            "violated": None,
        },
    ]
    expected_res_lines_dtypes = {
        "line_id": object,
        "current1": complex,
        "current2": complex,
        "power1": complex,
        "power2": complex,
        "potential1": complex,
        "potential2": complex,
        "series_losses": complex,
        "series_current": complex,
        "max_current": float,
        "violated": pd.BooleanDtype(),
    }
    expected_res_lines = (
        pd.DataFrame.from_records(expected_res_lines_records).astype(expected_res_lines_dtypes).set_index("line_id")
    )
    assert_frame_equal(en.res_lines, expected_res_lines, rtol=1e-4, atol=1e-5)

    # Lines with violated max current
    en.lines["line"].parameters.max_current = 0.002
    expected_res_lines_violated_records = [
        d | {"max_current": 0.002, "violated": True} for d in expected_res_lines_records
    ]
    expected_res_violated_lines = (
        pd.DataFrame.from_records(expected_res_lines_violated_records)
        .astype(expected_res_lines_dtypes)
        .set_index("line_id")
    )
    assert_frame_equal(en.res_lines, expected_res_violated_lines, rtol=1e-4, atol=1e-5)

    # Switches results
    expected_res_switches = (
        pd.DataFrame.from_records(
            data=[],
            columns=[
                "switch_id",
                "current1",
                "current2",
                "power1",
                "power2",
                "potential1",
                "potential2",
            ],
        )
        .astype(
            {
                "switch_id": object,
                "current1": complex,
                "current2": complex,
                "power1": complex,
                "power2": complex,
                "potential1": complex,
                "potential2": complex,
            }
        )
        .set_index("switch_id")
    )
    assert_frame_equal(en.res_switches, expected_res_switches)

    # Loads results
    expected_res_loads = (
        pd.DataFrame.from_records(
            [
                {
                    "load_id": "load",
                    "type": "power",
                    "current": 0.00500 + 7.22802e-25j,
                    "power": (19999.94999 + 2.89119e-18j) * (0.00500 + 7.22802e-25j).conjugate(),
                    "potential": 19999.94999 + 2.89119e-18j,
                },
            ]
        )
        .astype(
            {
                "load_id": object,
                "type": LoadTypeDtype,
                "current": complex,
                "power": complex,
                "potential": complex,
            }
        )
        .set_index("load_id")
    )
    assert_frame_equal(en.res_loads, expected_res_loads, rtol=1e-4)

    # Sources results
    expected_res_sources = (
        pd.DataFrame.from_records(
            [
                {
                    "source_id": "vs",
                    "current": -0.00500 + 0j,
                    "power": (20000 + 2.89120e-18j) * (-0.00500 + 0j).conjugate(),
                    "potential": 20000 + 2.89120e-18j,
                }
            ]
        )
        .astype(
            {
                "source_id": object,
                "current": complex,
                "power": complex,
                "potential": complex,
            }
        )
        .set_index("source_id")
    )
    assert_frame_equal(en.res_sources, expected_res_sources, rtol=1e-4)

    # No flexible loads
    assert en.res_loads_flexible_powers.empty

    # Let's add a flexible load
    fp = FlexibleParameter.p_max_u_consumption(u_min=16000, u_down=17000, s_max=1000)
    load = en.loads["load"]
    assert isinstance(load, PowerLoad)
    load._flexible_param = fp
    load._res_flexible_power = 100
    load._fetch_results = False
    expected_res_flex_powers = (
        pd.DataFrame.from_records(
            [
                {
                    "load_id": "load",
                    "flexible_power": 99.99999999999994 + 0j,
                }
            ]
        )
        .astype({"load_id": object, "flexible_power": complex})
        .set_index("load_id")
    )
    assert_frame_equal(en.res_loads_flexible_powers, expected_res_flex_powers, rtol=1e-5)


def test_solver_warm_start(small_network: ElectricalNetwork, monkeypatch):
    load: PowerLoad = small_network.loads["load"]
    load_bus = small_network.buses["bus1"]

    original_propagate_potentials = small_network._propagate_potentials
    original_reset_inputs = small_network._reset_inputs

    def _propagate_potentials():
        nonlocal propagate_potentials_called
        propagate_potentials_called = True
        return original_propagate_potentials()

    def _reset_inputs():
        nonlocal reset_inputs_called
        reset_inputs_called = True
        return original_reset_inputs()

    monkeypatch.setattr(small_network, "_propagate_potentials", _propagate_potentials)
    monkeypatch.setattr(small_network, "_reset_inputs", _reset_inputs)
    monkeypatch.setattr(small_network._solver, "solve_load_flow", lambda *_, **__: (1, 1e-20))

    # First case: network is valid, no results yet -> no warm start
    propagate_potentials_called = False
    reset_inputs_called = False
    assert small_network._valid
    assert not small_network._results_valid  # Results are not valid by default
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Make sure there is no warning
        small_network.solve_load_flow(warm_start=True)
    assert not propagate_potentials_called  # Is not called because it was already called in the constructor
    assert not reset_inputs_called

    # Second case: the user requested no warm start (even though the network and results are valid)
    propagate_potentials_called = False
    reset_inputs_called = False
    assert small_network._valid
    assert small_network._results_valid
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Make sure there is no warning
        small_network.solve_load_flow(warm_start=False)
    assert not propagate_potentials_called
    assert reset_inputs_called

    # Third case: network is valid, results are valid -> warm start
    propagate_potentials_called = False
    reset_inputs_called = False
    assert small_network._valid
    assert small_network._results_valid
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Make sure there is no warning
        small_network.solve_load_flow(warm_start=True)
    assert not propagate_potentials_called
    assert not reset_inputs_called

    # Fourth case (load powers changes): network is valid, results are not valid -> warm start
    propagate_potentials_called = False
    reset_inputs_called = False
    load.power = load.power + Q_(1 + 1j, "VA")
    assert small_network._valid
    assert not small_network._results_valid
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Make sure there is no warning
        small_network.solve_load_flow(warm_start=True)
    assert not propagate_potentials_called
    assert not reset_inputs_called

    # Fifth case: network is not valid -> no warm start
    propagate_potentials_called = False
    reset_inputs_called = False
    new_load = PowerLoad("new_load", load_bus, power=100)
    assert new_load.network is small_network
    assert not small_network._valid
    assert not small_network._results_valid
    with warnings.catch_warnings():
        # We could warn here that the user requested warm start but the network is not valid
        # but this will be disruptive for the user especially that warm start is the default
        warnings.simplefilter("error")  # Make sure there is no warning
        small_network.solve_load_flow(warm_start=True)
    assert propagate_potentials_called
    assert not reset_inputs_called


def test_propagate_potentials():
    # Delta source
    source_bus = Bus(id="source_bus")
    _ = VoltageSource(id="source", bus=source_bus, voltage=20e3)
    load_bus = Bus(id="load_bus")
    _ = Switch(id="switch", bus1=source_bus, bus2=load_bus)

    assert not load_bus._initialized
    assert not source_bus._initialized
    _ = ElectricalNetwork.from_element(source_bus)
    assert load_bus._initialized
    assert source_bus._initialized
    expected_potentials = 20e3
    assert np.allclose(load_bus.potential.m, expected_potentials)
    assert np.allclose(source_bus.potential.m, expected_potentials)


def test_to_graph(small_network: ElectricalNetwork):
    g = small_network.to_graph()
    assert isinstance(g, nx.Graph)
    assert sorted(g.nodes) == sorted(small_network.buses)
    assert sorted(g.edges) == sorted(
        (b.bus1.id, b.bus2.id)
        for b in it.chain(
            small_network.lines.values(), small_network.transformers.values(), small_network.switches.values()
        )
    )

    for bus in small_network.buses.values():
        node_data = g.nodes[bus.id]
        assert node_data["geom"] == bus.geometry

    for line in small_network.lines.values():
        edge_data = g.edges[line.bus1.id, line.bus2.id]
        max_current = line.max_current.magnitude if line.max_current is not None else None
        assert edge_data == {
            "id": line.id,
            "type": "line",
            "parameters_id": line.parameters.id,
            "max_current": max_current,
            "geom": line.geometry,
        }

    for transformer in small_network.transformers.values():
        edge_data = g.edges[transformer.bus1.id, transformer.bus2.id]
        max_power = transformer.max_power.magnitude if transformer.max_power is not None else None
        assert edge_data == {
            "id": transformer.id,
            "type": "transformer",
            "parameters_id": transformer.parameters.id,
            "max_power": max_power,
            "geom": transformer.geometry,
        }

    for switch in small_network.switches.values():
        edge_data = g.edges[switch.bus1.id, switch.bus2.id]
        assert edge_data == {"id": switch.id, "type": "switch", "geom": switch.geometry}


def test_serialization(all_element_network, all_element_network_with_results):
    def assert_results(en_dict: dict, included: bool):
        for bus_data in en_dict["buses"]:
            assert ("results" in bus_data) == included
        for line_data in en_dict["lines"]:
            assert ("results" in line_data) == included
        for transformer_data in en_dict["transformers"]:
            assert ("results" in transformer_data) == included
        for switch_data in en_dict["switches"]:
            assert ("results" in switch_data) == included
        for source_data in en_dict["sources"]:
            assert ("results" in source_data) == included
        for load_data in en_dict["loads"]:
            assert ("results" in load_data) == included

    # No results: include_results is ignored
    en = all_element_network
    en_dict_with_results = en.to_dict(include_results=True)
    en_dict_without_results = en.to_dict(include_results=False)
    assert_results(en_dict_with_results, included=False)
    assert_results(en_dict_without_results, included=False)
    assert en_dict_with_results == en_dict_without_results
    new_en = ElectricalNetwork.from_dict(en_dict_without_results)
    assert new_en.to_dict() == en_dict_without_results

    # Has results: include_results is respected
    en = all_element_network_with_results
    en_dict_with_results = en.to_dict(include_results=True)
    en_dict_without_results = en.to_dict(include_results=False)
    assert_results(en_dict_with_results, included=True)
    assert_results(en_dict_without_results, included=False)
    assert en_dict_with_results != en_dict_without_results
    # round tripping
    assert ElectricalNetwork.from_dict(en_dict_with_results).to_dict() == en_dict_with_results
    assert ElectricalNetwork.from_dict(en_dict_without_results).to_dict() == en_dict_without_results
    # default is to include the results
    assert en.to_dict() == en_dict_with_results

    # Has invalid results: cannot include them
    en.loads["load0"].power += Q_(1, "VA")  # <- invalidate the results
    with pytest.raises(RoseauLoadFlowException) as e:
        en.to_dict(include_results=True)
    assert e.value.msg == (
        "Trying to convert ElectricalNetwork with invalid results to a dict. Either call "
        "`en.solve_load_flow()` before converting or pass `include_results=False`."
    )
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_LOAD_FLOW_RESULT
    en_dict_without_results = en.to_dict(include_results=False)
    # round tripping without the results should still work
    assert ElectricalNetwork.from_dict(en_dict_without_results).to_dict() == en_dict_without_results


def test_results_to_dict(all_element_network_with_results):
    en = all_element_network_with_results

    # By default full=False
    res_network = en.results_to_dict()
    assert set(res_network) == {
        "buses",
        "lines",
        "transformers",
        "switches",
        "loads",
        "sources",
    }
    for v in res_network.values():
        assert isinstance(v, list)
    for res_bus in res_network["buses"]:
        bus = en.buses[res_bus["id"]]
        complex_potentials = res_bus["potential"][0] + 1j * res_bus["potential"][1]
        np.testing.assert_allclose(complex_potentials, bus.res_potential.m)
    for res_line in res_network["lines"]:
        line = en.lines[res_line["id"]]
        complex_currents1 = res_line["current1"][0] + 1j * res_line["current1"][1]
        np.testing.assert_allclose(complex_currents1, line.res_currents[0].m)
        complex_currents2 = res_line["current2"][0] + 1j * res_line["current2"][1]
        np.testing.assert_allclose(complex_currents2, line.res_currents[1].m)
    for res_transformer in res_network["transformers"]:
        transformer = en.transformers[res_transformer["id"]]
        complex_currents1 = res_transformer["current1"][0] + 1j * res_transformer["current1"][1]
        np.testing.assert_allclose(complex_currents1, transformer.res_currents[0].m)
        complex_currents2 = res_transformer["current2"][0] + 1j * res_transformer["current2"][1]
        np.testing.assert_allclose(complex_currents2, transformer.res_currents[1].m)
    for res_switch in res_network["switches"]:
        switch = en.switches[res_switch["id"]]
        complex_currents1 = res_switch["current1"][0] + 1j * res_switch["current1"][1]
        np.testing.assert_allclose(complex_currents1, switch.res_currents[0].m)
        complex_currents2 = res_switch["current2"][0] + 1j * res_switch["current2"][1]
        np.testing.assert_allclose(complex_currents2, switch.res_currents[1].m)
    for res_load in res_network["loads"]:
        load = en.loads[res_load["id"]]
        complex_currents = res_load["current"][0] + 1j * res_load["current"][1]
        np.testing.assert_allclose(complex_currents, load.res_current.m)
    for res_source in res_network["sources"]:
        source = en.sources[res_source["id"]]
        complex_currents = res_source["current"][0] + 1j * res_source["current"][1]
        np.testing.assert_allclose(complex_currents, source.res_current.m)


def test_results_to_dict_full(all_element_network_with_results):
    en = all_element_network_with_results

    # Here, `full` is True
    res_network = en.results_to_dict(full=True)
    assert set(res_network) == {
        "buses",
        "lines",
        "transformers",
        "switches",
        "loads",
        "sources",
    }
    for v in res_network.values():
        assert isinstance(v, list)
    for res_bus in res_network["buses"]:
        bus = en.buses[res_bus["id"]]
        complex_potentials = res_bus["potential"][0] + 1j * res_bus["potential"][1]
        np.testing.assert_allclose(complex_potentials, bus.res_potential.m)
        complex_voltages = res_bus["voltage"][0] + 1j * res_bus["voltage"][1]
        np.testing.assert_allclose(complex_voltages, bus.res_voltage.m)
    for res_line in res_network["lines"]:
        line = en.lines[res_line["id"]]
        # Currents
        complex_currents1 = res_line["current1"][0] + 1j * res_line["current1"][1]
        np.testing.assert_allclose(complex_currents1, line.res_currents[0].m)
        complex_currents2 = res_line["current2"][0] + 1j * res_line["current2"][1]
        np.testing.assert_allclose(complex_currents2, line.res_currents[1].m)
        # Potentials
        complex_potentials1 = res_line["potential1"][0] + 1j * res_line["potential1"][1]
        np.testing.assert_allclose(complex_potentials1, line.res_potentials[0].m)
        complex_potentials2 = res_line["potential2"][0] + 1j * res_line["potential2"][1]
        np.testing.assert_allclose(complex_potentials2, line.res_potentials[1].m)
        # Powers
        complex_powers1 = res_line["power1"][0] + 1j * res_line["power1"][1]
        np.testing.assert_allclose(complex_powers1, line.res_powers[0].m)
        complex_powers2 = res_line["power2"][0] + 1j * res_line["power2"][1]
        np.testing.assert_allclose(complex_powers2, line.res_powers[1].m)
        # Voltages
        complex_voltages1 = res_line["voltage1"][0] + 1j * res_line["voltage1"][1]
        np.testing.assert_allclose(complex_voltages1, line.res_voltage[0].m)
        complex_voltages2 = res_line["voltage2"][0] + 1j * res_line["voltage2"][1]
        np.testing.assert_allclose(complex_voltages2, line.res_voltage[1].m)
        # Power losses
        complex_power_losses = res_line["power_losses"][0] + 1j * res_line["power_losses"][1]
        np.testing.assert_allclose(complex_power_losses, line.res_power_losses.m)
        # Series currents
        complex_series_currents = res_line["series_current"][0] + 1j * res_line["series_current"][1]
        np.testing.assert_allclose(complex_series_currents, line.res_series_currents.m)
        # Shunt currents
        complex_shunt_currents1 = res_line["shunt_current1"][0] + 1j * res_line["shunt_current1"][1]
        np.testing.assert_allclose(complex_shunt_currents1, line.res_shunt_currents[0].m)
        complex_shunt_currents2 = res_line["shunt_current2"][0] + 1j * res_line["shunt_current2"][1]
        np.testing.assert_allclose(complex_shunt_currents2, line.res_shunt_currents[1].m)
        # Shunt power losses
        complex_shunt_power_losses = res_line["shunt_power_losses"][0] + 1j * res_line["shunt_power_losses"][1]
        np.testing.assert_allclose(complex_shunt_power_losses, line.res_shunt_power_losses.m)

    for res_transformer in res_network["transformers"]:
        transformer = en.transformers[res_transformer["id"]]
        # Currents
        complex_currents1 = res_transformer["current1"][0] + 1j * res_transformer["current1"][1]
        np.testing.assert_allclose(complex_currents1, transformer.res_currents[0].m)
        complex_currents2 = res_transformer["current2"][0] + 1j * res_transformer["current2"][1]
        np.testing.assert_allclose(complex_currents2, transformer.res_currents[1].m)
        # Power losses
        complex_power_losses = complex(*res_transformer["power_losses"])
        np.testing.assert_allclose(complex_power_losses, transformer.res_power_losses.m)
    for res_switch in res_network["switches"]:
        switch = en.switches[res_switch["id"]]
        # Currents
        complex_currents1 = res_switch["current1"][0] + 1j * res_switch["current1"][1]
        np.testing.assert_allclose(complex_currents1, switch.res_currents[0].m)
        complex_currents2 = res_switch["current2"][0] + 1j * res_switch["current2"][1]
        np.testing.assert_allclose(complex_currents2, switch.res_currents[1].m)
        # Potentials
        complex_potentials1 = res_switch["potential1"][0] + 1j * res_switch["potential1"][1]
        np.testing.assert_allclose(complex_potentials1, switch.res_potentials[0].m)
        complex_potentials2 = res_switch["potential2"][0] + 1j * res_switch["potential2"][1]
        np.testing.assert_allclose(complex_potentials2, switch.res_potentials[1].m)
        # Powers
        complex_powers1 = res_switch["power1"][0] + 1j * res_switch["power1"][1]
        np.testing.assert_allclose(complex_powers1, switch.res_powers[0].m)
        complex_powers2 = res_switch["power2"][0] + 1j * res_switch["power2"][1]
        np.testing.assert_allclose(complex_powers2, switch.res_powers[1].m)
        # Voltages
        complex_voltages1 = res_switch["voltage1"][0] + 1j * res_switch["voltage1"][1]
        np.testing.assert_allclose(complex_voltages1, switch.res_voltage[0].m)
        complex_voltages2 = res_switch["voltage2"][0] + 1j * res_switch["voltage2"][1]
        np.testing.assert_allclose(complex_voltages2, switch.res_voltage[1].m)
    for res_load in res_network["loads"]:
        load = en.loads[res_load["id"]]
        # Currents
        complex_currents = res_load["current"][0] + 1j * res_load["current"][1]
        np.testing.assert_allclose(complex_currents, load.res_current.m)
        # Powers
        complex_powers = res_load["power"][0] + 1j * res_load["power"][1]
        np.testing.assert_allclose(complex_powers, load.res_power.m)
        # Potentials
        if "potentials" in res_load:
            complex_potentials = res_load["potential"][0] + 1j * res_load["potential"][1]
            np.testing.assert_allclose(complex_potentials, load.res_potential.m)
        # Flexible powers
        if "flexible_powers" in res_load:
            complex_flexible_powers = res_load["flexible_power"][0] + 1j * res_load["flexible_power"][1]
            np.testing.assert_allclose(complex_flexible_powers, load.res_flexible_power.m)
    for res_source in res_network["sources"]:
        source = en.sources[res_source["id"]]
        # Currents
        complex_currents = res_source["current"][0] + 1j * res_source["current"][1]
        np.testing.assert_allclose(complex_currents, source.res_current.m)
        # Powers
        complex_powers = res_source["power"][0] + 1j * res_source["power"][1]
        np.testing.assert_allclose(complex_powers, source.res_power.m)
        # Potentials
        if "potentials" in res_source:
            complex_potentials = res_source["potential"][0] + 1j * res_source["potential"][1]
            np.testing.assert_allclose(complex_potentials, source.res_potential.m)


def test_results_to_json(small_network_with_results, tmp_path):
    en = small_network_with_results
    res_network_expected = en.results_to_dict()
    tmp_file = tmp_path / "results.json"
    en.results_to_json(tmp_file)

    with tmp_file.open() as fp:
        res_network = json.load(fp)

    assert res_network == res_network_expected
