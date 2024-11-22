import pytest

from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow_single.models import Bus, Line, LineParameters, Switch, VoltageSource


def test_switch_loop():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    bus3 = Bus(id="bus3")

    Switch(id="switch1", bus1=bus1, bus2=bus2)
    lp = LineParameters(id="test", z_line=1.0)
    Line(id="line", bus1=bus1, bus2=bus3, parameters=lp, length=10)

    with pytest.raises(RoseauLoadFlowException) as e:
        Switch(id="switch2", bus1=bus1, bus2=bus2)
    assert "There is a loop of switch" in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.SWITCHES_LOOP

    with pytest.raises(RoseauLoadFlowException) as e:
        Switch(id="switch3", bus1=bus2, bus2=bus1)
    assert "There is a loop of switch" in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.SWITCHES_LOOP

    Switch(id="switch4", bus1=bus2, bus2=bus3)
    with pytest.raises(RoseauLoadFlowException) as e:
        Switch(id="switch5", bus1=bus1, bus2=bus3)
    assert "There is a loop of switch" in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.SWITCHES_LOOP


def test_switch_connection():
    bus1 = Bus(id="bus1")
    bus2 = Bus(id="bus2")
    VoltageSource(id="vs1", bus=bus1, voltage=230 + 0j)
    VoltageSource(id="vs2", bus=bus2, voltage=230 + 0j)
    with pytest.raises(RoseauLoadFlowException) as e:
        Switch(id="switch", bus1=bus1, bus2=bus2)
    assert "are connected with the switch" in e.value.msg
    assert e.value.code == RoseauLoadFlowExceptionCode.BAD_VOLTAGES_SOURCES_CONNECTION
