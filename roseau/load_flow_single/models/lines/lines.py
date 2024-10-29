import logging

import numpy as np
from shapely.geometry.base import BaseGeometry

from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.typing import Complex, ComplexArray, Id, JsonDict
from roseau.load_flow.units import Q_, ureg_wraps
from roseau.load_flow_engine.cy_engine import CyShuntLine, CySimplifiedLine
from roseau.load_flow_single.models.branches import AbstractBranch
from roseau.load_flow_single.models.buses import Bus
from roseau.load_flow_single.models.lines.parameters import LineParameters

logger = logging.getLogger(__name__)


class Line(AbstractBranch):
    """An electrical line PI model with series impedance and optional shunt admittance."""

    def __init__(
        self,
        id: Id,
        bus1: Bus,
        bus2: Bus,
        *,
        parameters: LineParameters,
        length: float | Q_[float],
        geometry: BaseGeometry | None = None,
    ) -> None:
        """Line constructor.

        Args:
            id:
                A unique ID of the line in the network branches.

            bus1:
                The first bus (aka `"from_bus"`) to connect to the line.

            bus2:
                The second bus (aka `"to_bus"`) to connect to the line.

            parameters:
                Parameters defining the electric model of the line using its impedance and shunt
                admittance matrices. This is an instance of the :class:`LineParameters` class and
                can be used by multiple lines.

            length:
                The length of the line (in km).

            geometry:
                The geometry of the line i.e. the linestring.
        """
        self._initialized = False
        super().__init__(id=id, bus1=bus1, bus2=bus2, geometry=geometry)
        self.length = length
        self.parameters = parameters
        self._initialized = True
        self._ground = None

        if parameters.with_shunt:
            self._cy_element = CyShuntLine(
                n=1,
                y_shunt=parameters._y_shunt.reshape(1) * self._length,
                z_line=parameters._z_line.reshape(1) * self._length,
            )
        else:
            self._cy_element = CySimplifiedLine(n=1, z_line=parameters._z_line.reshape(1) * self._length)
        self._cy_connect()

        # Cache values used in results calculations
        self._z_line = parameters._z_line * self._length
        self._y_shunt = parameters._y_shunt * self._length
        self._z_line_inv = np.linalg.inv(self._z_line)
        self._yg = self._y_shunt.sum(axis=1)  # y_ig = Y_ia + Y_ib + Y_ic + Y_in for i in {a, b, c, n}

    def _update_internal_parameters(self, parameters: LineParameters, length: float) -> None:
        """Update the internal parameters of the line."""
        self._parameters = parameters
        self._length = length

        self._z_line = parameters._z_line * length
        self._y_shunt = parameters._y_shunt * length
        self._z_line_inv = np.linalg.inv(self._z_line)
        self._yg = self._y_shunt.sum(axis=1)

        if self._cy_element is not None:
            if self._parameters.with_shunt:
                self._cy_element.update_line_parameters(
                    y_shunt=self._y_shunt.reshape(1), z_line=self._z_line.reshape(1)
                )
            else:
                self._cy_element.update_line_parameters(z_line=self._z_line.reshape(1))

    @property
    @ureg_wraps("km", (None,))
    def length(self) -> Q_[float]:
        """The length of the line (in km)."""
        return self._length

    @length.setter
    @ureg_wraps(None, (None, "km"))
    def length(self, value: float | Q_[float]) -> None:
        if value <= 0:
            msg = f"A line length must be greater than 0. {value:.2f} km provided."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_LENGTH_VALUE)
        self._invalidate_network_results()
        self._length = value
        if self._initialized:
            self._update_internal_parameters(self._parameters, value)

    @property
    def parameters(self) -> LineParameters:
        """The parameters defining the impedance and shunt admittance matrices of line model."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: LineParameters) -> None:
        shape = (1, 1)
        if value._z_line.shape != shape:
            msg = f"Incorrect z_line dimensions for line {self.id!r}: {value._z_line.shape} instead of {shape}"
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_Z_LINE_SHAPE)

        if value.with_shunt:
            if self._initialized and not self.with_shunt:
                msg = "Cannot set line parameters with a shunt to a line that does not have shunt components."
                logger.error(msg)
                raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_LINE_MODEL)
            if value._y_shunt.shape != shape:
                msg = f"Incorrect y_shunt dimensions for line {self.id!r}: {value._y_shunt.shape} instead of {shape}"
                logger.error(msg)
                raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_Y_SHUNT_SHAPE)
        else:
            if self._initialized and self.with_shunt:
                msg = "Cannot set line parameters without a shunt to a line that has shunt components."
                logger.error(msg)
                raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_LINE_MODEL)
        self._invalidate_network_results()
        self._parameters = value
        if self._initialized:
            self._update_internal_parameters(value, self._length)

    @property
    @ureg_wraps("ohm", (None,))
    def z_line(self) -> Q_[ComplexArray]:
        """Impedance of the line (in Ohm)."""
        return self._parameters._z_line * self._length

    @property
    @ureg_wraps("S", (None,))
    def y_shunt(self) -> Q_[ComplexArray]:
        """Shunt admittance of the line (in Siemens)."""
        return self._parameters._y_shunt * self._length

    @property
    def max_current(self) -> Q_[float] | None:
        """The maximum current loading of the line (in A)."""
        # Do not add a setter. The user must know that if they change the max_current, it changes
        # for all lines that share the parameters. It is better to set it on the parameters.
        return self._parameters.max_current

    @property
    def with_shunt(self) -> bool:
        return self._parameters.with_shunt

    def _res_series_values_getter(self, warning: bool) -> tuple[ComplexArray, ComplexArray]:
        pot1, pot2 = self._res_potentials_getter(warning)  # V
        du_line = pot1 - pot2
        i_line = self._z_line_inv @ du_line  # Zₗ x Iₗ = ΔU -> I = Zₗ⁻¹ x ΔU
        return du_line, i_line

    def _res_series_currents_getter(self, warning: bool) -> ComplexArray:
        _, i_line = self._res_series_values_getter(warning)
        return i_line

    @property
    @ureg_wraps("A", (None,))
    def res_series_currents(self) -> Q_[Complex]:
        """Get the current in the series elements of the line (in A)."""
        return self._res_series_currents_getter(warning=True)[0]

    def _res_series_power_losses_getter(self, warning: bool) -> ComplexArray:
        du_line, i_line = self._res_series_values_getter(warning)
        return du_line * i_line.conj()  # Sₗ = ΔU.Iₗ*

    @property
    @ureg_wraps("VA", (None,))
    def res_series_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the series elements of the line (in VA)."""
        return self._res_series_power_losses_getter(warning=True)[0]

    def _res_shunt_values_getter(self, warning: bool) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        assert self.with_shunt, "This method only works when there is a shunt"
        assert self._ground is not None
        pot1, pot2 = self._res_potentials_getter(warning)
        vg = self._ground._res_potential_getter(warning=False)
        ig = self._yg * vg
        i1_shunt = (self._y_shunt @ pot1 - ig) / 2
        i2_shunt = (self._y_shunt @ pot2 - ig) / 2
        return pot1, pot2, i1_shunt, i2_shunt

    def _res_shunt_currents_getter(self, warning: bool) -> tuple[ComplexArray, ComplexArray]:
        if not self.with_shunt:
            zeros = np.zeros(1, dtype=np.complex128)
            return zeros[:], zeros[:]
        _, _, cur1, cur2 = self._res_shunt_values_getter(warning)
        return cur1, cur2

    @property
    @ureg_wraps(("A", "A"), (None,))
    def res_shunt_currents(self) -> tuple[Q_[ComplexArray], Q_[ComplexArray]]:
        """Get the currents in the shunt elements of the line (in A)."""
        cur1, cur2 = self._res_shunt_currents_getter(warning=True)
        return cur1[0], cur2[0]

    def _res_shunt_power_losses_getter(self, warning: bool) -> ComplexArray:
        if not self.with_shunt:
            return np.zeros(1, dtype=np.complex128)
        pot1, pot2, cur1, cur2 = self._res_shunt_values_getter(warning)
        return pot1 * cur1.conj() + pot2 * cur2.conj()

    @property
    @ureg_wraps("VA", (None,))
    def res_shunt_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the shunt elements of the line (in VA)."""
        return self._res_shunt_power_losses_getter(warning=True)[0]

    def _res_power_losses_getter(self, warning: bool) -> ComplexArray:
        series_losses = self._res_series_power_losses_getter(warning)
        shunt_losses = self._res_shunt_power_losses_getter(warning=False)  # we warn on the previous line
        return series_losses + shunt_losses

    @property
    @ureg_wraps("VA", (None,))
    def res_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the line (in VA)."""
        return self._res_power_losses_getter(warning=True)[0]

    @property
    def res_violated(self) -> bool | None:
        """Whether the line current exceeds the maximum current (loading > 100%).

        Returns ``None`` if the maximum current is not set.
        """
        i_max = self._parameters._max_current
        if i_max is None:
            return None
        currents1, currents2 = self._res_currents_getter(warning=True)
        return float(np.max([abs(currents1), abs(currents2)])) > i_max

    #
    # Json Mixin interface
    #
    def _to_dict(self, include_results: bool) -> JsonDict:
        res = {
            "id": self.id,
            "bus1": self.bus1.id,
            "bus2": self.bus2.id,
            "length": self._length,
            "params_id": self._parameters.id,
        }
        if self.geometry is not None:
            res["geometry"] = self.geometry.__geo_interface__
        if include_results:
            currents1, currents2 = self._res_currents_getter(warning=True)
            res["results"] = {
                "currents1": [[i.real, i.imag] for i in currents1],
                "currents2": [[i.real, i.imag] for i in currents2],
            }
        return res

    def _results_to_dict(self, warning: bool, full: bool) -> JsonDict:
        currents1, currents2 = self._res_currents_getter(warning)
        results = {
            "id": self.id,
            "currents1": [[i.real, i.imag] for i in currents1],
            "currents2": [[i.real, i.imag] for i in currents2],
        }
        if full:
            potentials1, potentials2 = self._res_potentials_getter(warning=False)
            results["potentials1"] = [[v.real, v.imag] for v in potentials1]
            results["potentials2"] = [[v.real, v.imag] for v in potentials2]
            powers1, powers2 = self._res_powers_getter(
                warning=False,
                potentials1=potentials1,
                potentials2=potentials2,
                currents1=currents1,
                currents2=currents2,
            )
            results["powers1"] = [[s.real, s.imag] for s in powers1]
            results["powers2"] = [[s.real, s.imag] for s in powers2]
            voltages1, voltages2 = self._res_voltages_getter(
                warning=False, potentials1=potentials1, potentials2=potentials2
            )
            results["voltages1"] = [[v.real, v.imag] for v in voltages1]
            results["voltages2"] = [[v.real, v.imag] for v in voltages2]
            results["power_losses"] = [[s.real, s.imag] for s in self._res_power_losses_getter(warning=False)]
            results["series_currents"] = [[i.real, i.imag] for i in self._res_series_currents_getter(warning=False)]
            results["series_power_losses"] = [
                [s.real, s.imag] for s in self._res_series_power_losses_getter(warning=False)
            ]
            shunt_currents1, shunt_currents2 = self._res_shunt_currents_getter(warning=False)
            results["shunt_currents1"] = [[i.real, i.imag] for i in shunt_currents1]
            results["shunt_currents2"] = [[i.real, i.imag] for i in shunt_currents2]
            results["shunt_power_losses"] = [
                [s.real, s.imag] for s in self._res_shunt_power_losses_getter(warning=False)
            ]
        return results
