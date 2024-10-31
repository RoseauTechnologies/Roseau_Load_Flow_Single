import logging
from abc import ABC
from typing import ClassVar, Final, Literal

import numpy as np

from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.models.loads.flexible_parameters import FlexibleParameter
from roseau.load_flow.typing import Complex, Id, JsonDict
from roseau.load_flow.units import Q_, ureg_wraps
from roseau.load_flow_engine.cy_engine import (
    CyFlexibleLoad,
    CyPowerLoad,
)
from roseau.load_flow_single.models.buses import Bus
from roseau.load_flow_single.models.core import Element

logger = logging.getLogger(__name__)


class AbstractLoad(Element, ABC):
    """An abstract class of an electric load."""

    type: ClassVar[Literal["power", "current", "impedance"]]

    def __init__(self, id: Id, bus: Bus) -> None:
        """AbstractLoad constructor.

        Args:
            id:
                A unique ID of the load in the network loads.

            bus:
                The bus to connect the load to.
        """
        if type(self) is AbstractLoad:
            raise TypeError("Can't instantiate abstract class AbstractLoad")
        super().__init__(id)
        self._connect(bus)

        self._bus = bus
        self._n = 2
        self._symbol = {"power": "S", "current": "I", "impedance": "Z"}[self.type]

        # Results
        self._res_current: Complex | None = None
        self._res_potential: Complex | None = None

    def __repr__(self) -> str:
        bus_id = self.bus.id if self.bus is not None else None
        return f"<{type(self).__name__}: id={self.id!r}, bus={bus_id!r}>"

    @property
    def bus(self) -> Bus:
        """The bus of the load."""
        return self._bus

    @property
    def is_flexible(self) -> bool:
        """Whether the load is flexible or not. Only :class:`PowerLoad` can be flexible."""
        return False

    def _refresh_results(self) -> None:
        self._res_current = self._cy_element.get_currents(self._n)[0]
        self._res_potential = self._cy_element.get_potentials(self._n)[0]

    def _res_current_getter(self, warning: bool) -> Complex:
        if self._fetch_results:
            self._refresh_results()
        return self._res_getter(value=self._res_current, warning=warning)

    @property
    @ureg_wraps("A", (None,))
    def res_current(self) -> Q_[Complex]:
        """The load flow result of the load currents (A)."""
        return self._res_current_getter(warning=True)

    def _validate_value(self, value: Complex) -> Complex:
        # A load cannot have any zero impedance
        if self.type == "impedance" and np.isclose(value, 0).any():
            msg = f"An impedance of the load {self.id!r} is null"
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_Z_VALUE)
        return value

    def _res_potential_getter(self, warning: bool) -> Complex:
        if self._fetch_results:
            self._refresh_results()
        return self._res_getter(value=self._res_potential, warning=warning)

    def _res_voltage_getter(self, warning: bool) -> Complex:
        return self._res_potential_getter(warning)

    @property
    @ureg_wraps("V", (None,))
    def res_voltage(self) -> Q_[Complex]:
        """The load flow result of the load voltages (V)."""
        return self._res_voltage_getter(warning=True)

    def _res_power_getter(
        self, warning: bool, current: Complex | None = None, potential: Complex | None = None
    ) -> Complex:
        if current is None:
            current = self._res_current_getter(warning=warning)
            warning = False  # we warn only one
        if potential is None:
            potential = self._res_potential_getter(warning=warning)
        return potential * current.conj()

    @property
    @ureg_wraps("VA", (None,))
    def res_power(self) -> Q_[Complex]:
        """The load flow result of the "line powers" flowing into the load (VA)."""
        return self._res_power_getter(warning=True)

    def _cy_connect(self):
        connections = []
        for i in range(self._n):
            connections.append((i, i))
        self.bus._cy_element.connect(self._cy_element, connections)

    #
    # Disconnect
    #
    def disconnect(self) -> None:
        """Disconnect this load from the network. It cannot be used afterwards."""
        self._disconnect()
        self._bus = None

    def _raise_disconnected_error(self) -> None:
        """Raise an error if the load is disconnected."""
        if self.bus is None:
            msg = f"The load {self.id!r} is disconnected and cannot be used anymore."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.DISCONNECTED_ELEMENT)

    #
    # Json Mixin interface
    #
    @classmethod
    def from_dict(cls, data: JsonDict, *, include_results: bool = True) -> "AbstractLoad":
        load_type: Literal["power", "current", "impedance"] = data["type"]
        if load_type == "power":
            power = complex(data["power"][0], data["power"][1])
            if (fp_data := data.get("flexible_param")) is not None:
                fp = FlexibleParameter.from_dict(data=fp_data, include_results=include_results)
            else:
                fp = None
            self = PowerLoad(
                id=data["id"],
                bus=data["bus"],
                power=power,
                flexible_param=fp,
            )
        # elif load_type == "current": TODO
        #     currents = [complex(i[0], i[1]) for i in data["currents"]]
        #     self = CurrentLoad(id=data["id"], bus=data["bus"], currents=currents, phases=data["phases"])
        # elif load_type == "impedance":
        #     impedances = [complex(z[0], z[1]) for z in data["impedances"]]
        #     self = ImpedanceLoad(
        #         id=data["id"],
        #         bus=data["bus"],
        #         impedances=impedances,
        #         phases=data["phases"],
        #         connect_neutral=data.get("connect_neutral"),
        #     )
        else:
            msg = f"Unknown load type {load_type!r} for load {data['id']!r}"
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_LOAD_TYPE)
        if include_results and "results" in data:
            self._res_current = complex(data["results"]["current"][0], data["results"]["current"][1])
            self._res_potential = complex(data["results"]["potential"][0], data["results"]["potential"][1])
            if "flexible_power" in data["results"]:
                self._res_flexible_power = complex(
                    data["results"]["flexible_power"][0], data["results"]["flexible_power"][1]
                )

            self._fetch_results = False
            self._no_results = False
        return self

    def _to_dict(self, include_results: bool) -> JsonDict:
        self._raise_disconnected_error()
        complex_value = getattr(self, f"_{self.type}")
        res = {
            "id": self.id,
            "bus": self.bus.id,
            "type": self.type,
            f"{self.type}s": [complex_value.real, complex_value.imag],
        }
        if include_results:
            current = self._res_current_getter(warning=True)
            res["results"] = {"current": [current.real, current.imag]}
            potential = self._res_potential_getter(warning=True)
            res["results"]["potential"] = [potential.real, potential.imag]
        return res

    def _results_to_dict(self, warning: bool, full: bool) -> JsonDict:
        current = self._res_current_getter(warning)
        results = {
            "id": self.id,
            "type": self.type,
            "current": [current.real, current.imag],
        }
        potential = self._res_potential_getter(warning=False)
        results["potential"] = [potential.real, potential.imag]
        if full:
            power = self._res_power_getter(warning=False, current=current, potential=potential)
            results["power"] = [power.real, power.imag]
        return results


class PowerLoad(AbstractLoad):
    """A constant power load."""

    type: Final = "power"

    def __init__(
        self,
        id: Id,
        bus: Bus,
        *,
        power: Complex,
        flexible_param: FlexibleParameter | None = None,
    ) -> None:
        """PowerLoad constructor.

        Args:
            id:
                A unique ID of the load in the network loads.

            bus:
                The bus to connect the load to.

            power:
                A single power value, either complex value (VA) or a :class:`Quantity <roseau.load_flow.units.Q_>` of
                complex value.

            flexible_param:
                A :class:`FlexibleParameters` object. When provided, the load is considered as flexible
                (or controllable) and the parameters are used to compute the flexible power of the load.
        """
        super().__init__(id=id, bus=bus)

        self._flexible_param = flexible_param
        self.power = power
        self._res_flexible_power: Complex | None = None

        if self.is_flexible:
            cy_parameters = np.array([flexible_param._cy_fp])  # type: ignore
            self._cy_element = CyFlexibleLoad(
                n=self._n, powers=np.array([self._power], dtype=np.complex128), parameters=cy_parameters
            )
        else:
            self._cy_element = CyPowerLoad(n=self._n, powers=np.array([self._power], dtype=np.complex128))
        self._cy_connect()

    @property
    def flexible_param(self) -> FlexibleParameter | None:
        return self._flexible_param

    @property
    def is_flexible(self) -> bool:
        return self._flexible_param is not None

    @property
    @ureg_wraps("VA", (None,))
    def power(self) -> Q_[Complex]:
        """The powers of the load (VA).

        Setting the powers will update the load's power values and invalidate the network results.
        """
        return self._power

    @power.setter
    @ureg_wraps(None, (None, "VA"))
    def power(self, value: Complex) -> None:
        value = self._validate_value(value)
        if self._flexible_param is not None:
            power, fp = value, self._flexible_param
            if fp.control_p.type != "constant" or fp.control_q.type != "constant":
                if abs(power) > fp._s_max:
                    msg = f"The power is greater than the parameter s_max for flexible load {self.id!r}"
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_S_VALUE)
                if power.imag < fp._q_min:
                    msg = f"The reactive power is lower than the parameter q_min for flexible load {self.id!r}"
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_S_VALUE)
                if power.imag > fp._q_max:
                    msg = f"The reactive power is greater than the parameter q_max for flexible load {self.id!r}"
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_S_VALUE)
                if fp.control_p.type == "p_max_u_production" and power.real > 0:
                    msg = f"There is a production control but a positive power for flexible load {self.id!r}"
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_S_VALUE)
                if fp.control_p.type == "p_max_u_consumption" and power.real < 0:
                    msg = f"There is a consumption control but a negative power for flexible load {self.id!r}"
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_S_VALUE)
        self._power = value
        self._invalidate_network_results()
        if self._cy_element is not None:
            self._cy_element.update_powers([self._power])

    def _refresh_results(self) -> None:
        super()._refresh_results()
        if self.is_flexible:
            self._res_flexible_power = self._cy_element.get_powers(self._n)[0]

    def _res_flexible_power_getter(self, warning: bool) -> Complex:
        if self._fetch_results:
            self._refresh_results()
        return self._res_getter(value=self._res_flexible_power, warning=warning)

    @property
    @ureg_wraps("VA", (None,))
    def res_flexible_power(self) -> Q_[Complex]:
        """The load flow result of the load flexible powers (VA).

        This property is only available for flexible loads.

        It returns the powers actually consumed or produced by each component of the load instead
        of the "line powers" flowing into the load connection points (as the :meth:`res_powers`
        property does). The two properties are the same for Wye-connected loads but are different
        for Delta-connected loads.
        """
        if not self.is_flexible:
            msg = f"The load {self.id!r} is not flexible and does not have flexible powers"
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_LOAD_TYPE)
        return self._res_flexible_power_getter(warning=True)

    #
    # Json Mixin interface
    #
    def _to_dict(self, include_results: bool) -> JsonDict:
        res = super()._to_dict(include_results=include_results)
        if self.flexible_param is not None:
            res["flexible_param"] = self.flexible_param.to_dict(include_results=include_results)
            if include_results:
                power = self._res_flexible_power_getter(warning=False)
                res["results"]["flexible_power"] = [power.real, power.imag]
        return res

    def _results_to_dict(self, warning: bool, full: bool) -> JsonDict:
        if self.is_flexible:
            power = self._res_flexible_power_getter(warning=False)
            return {
                **super()._results_to_dict(warning=warning, full=full),
                "flexible_power": [power.real, power.imag],
            }
        else:
            return super()._results_to_dict(warning=warning, full=full)
