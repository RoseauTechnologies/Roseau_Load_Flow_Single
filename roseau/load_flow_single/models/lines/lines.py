import logging

import numpy as np
from shapely.geometry.base import BaseGeometry

from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.typing import Complex, Id, JsonDict
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
        self._with_shunt = parameters.with_shunt
        super().__init__(id=id, bus1=bus1, bus2=bus2, n=1, geometry=geometry)
        self.length = length
        self.parameters = parameters
        self._initialized = True

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
        self._z_line = parameters._z_line[0][0] * self._length
        self._y_shunt = parameters._y_shunt[0][0] * self._length
        self._z_line_inv = 1.0 / self._z_line
        self._yg = self._y_shunt  # y_ig = Y_ia + Y_ib + Y_ic + Y_in for i in {a, b, c, n}

    def _update_internal_parameters(self, parameters: LineParameters, length: float) -> None:
        """Update the internal parameters of the line."""
        self._parameters = parameters
        self._length = length

        self._z_line = parameters._z_line[0][0] * length
        self._y_shunt = parameters._y_shunt[0][0] * length
        self._z_line_inv = 1.0 / self._z_line
        self._yg = self._y_shunt.sum(axis=1)

        if self._cy_element is not None:
            if self._parameters.with_shunt:
                self._cy_element.update_line_parameters(
                    y_shunt=np.array([self._y_shunt], dtype=np.complex128),
                    z_line=np.array([self._z_line], dtype=np.complex128),
                )
            else:
                self._cy_element.update_line_parameters(z_line=np.array([self._z_line], dtype=np.complex128))

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
    def z_line(self) -> Q_[Complex]:
        """Impedance of the line (in Ohm)."""
        return self._parameters._z_line[0][0] * self._length

    @property
    @ureg_wraps("S", (None,))
    def y_shunt(self) -> Q_[Complex]:
        """Shunt admittance of the line (in Siemens)."""
        return self._parameters._y_shunt[0][0] * self._length

    @property
    def max_current(self) -> Q_[float] | None:
        """The maximum current loading of the line (in A)."""
        # Do not add a setter. The user must know that if they change the max_current, it changes
        # for all lines that share the parameters. It is better to set it on the parameters.
        return self._parameters.max_current

    @property
    def with_shunt(self) -> bool:
        return self._with_shunt

    def _res_series_values_getter(self, warning: bool) -> tuple[Complex, Complex]:
        pot1, pot2 = self._res_potentials_getter(warning)  # V
        du_line = pot1 - pot2
        i_line = self._z_line_inv * du_line  # Zₗ x Iₗ = ΔU -> I = Zₗ⁻¹ x ΔU
        return du_line, i_line

    def _res_series_currents_getter(self, warning: bool) -> Complex:
        _, i_line = self._res_series_values_getter(warning)
        return i_line

    @property
    @ureg_wraps("A", (None,))
    def res_series_currents(self) -> Q_[Complex]:
        """Get the current in the series elements of the line (in A)."""
        return self._res_series_currents_getter(warning=True)

    def _res_series_power_losses_getter(self, warning: bool) -> Complex:
        du_line, i_line = self._res_series_values_getter(warning)
        return du_line * i_line.conjugate()  # Sₗ = ΔU.Iₗ*

    @property
    @ureg_wraps("VA", (None,))
    def res_series_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the series elements of the line (in VA)."""
        return self._res_series_power_losses_getter(warning=True)

    def _res_shunt_values_getter(self, warning: bool) -> tuple[Complex, Complex, Complex, Complex]:
        assert self.with_shunt, "This method only works when there is a shunt"
        pot1, pot2 = self._res_potentials_getter(warning)
        vg = 0j
        ig = self._yg * vg
        i1_shunt = (self._y_shunt * pot1 - ig) / 2
        i2_shunt = (self._y_shunt * pot2 - ig) / 2
        return pot1, pot2, i1_shunt, i2_shunt

    def _res_shunt_currents_getter(self, warning: bool) -> tuple[Complex, Complex]:
        if not self.with_shunt:
            return 0j, 0j
        _, _, cur1, cur2 = self._res_shunt_values_getter(warning)
        return cur1, cur2

    @property
    @ureg_wraps(("A", "A"), (None,))
    def res_shunt_currents(self) -> tuple[Q_[Complex], Q_[Complex]]:
        """Get the currents in the shunt elements of the line (in A)."""
        cur1, cur2 = self._res_shunt_currents_getter(warning=True)
        return cur1, cur2

    def _res_shunt_power_losses_getter(self, warning: bool) -> Complex:
        if not self.with_shunt:
            return 0j
        pot1, pot2, cur1, cur2 = self._res_shunt_values_getter(warning)
        return pot1 * cur1.conjugate() + pot2 * cur2.conjugate()

    @property
    @ureg_wraps("VA", (None,))
    def res_shunt_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the shunt elements of the line (in VA)."""
        return self._res_shunt_power_losses_getter(warning=True)

    def _res_power_losses_getter(self, warning: bool) -> Complex:
        series_losses = self._res_series_power_losses_getter(warning)
        shunt_losses = self._res_shunt_power_losses_getter(warning=False)  # we warn on the previous line
        return series_losses + shunt_losses

    @property
    @ureg_wraps("VA", (None,))
    def res_power_losses(self) -> Q_[Complex]:
        """Get the power losses in the line (in VA)."""
        return self._res_power_losses_getter(warning=True)

    @property
    def res_violated(self) -> bool | None:
        """Whether the line current exceeds the maximum current (loading > 100%).

        Returns ``None`` if the maximum current is not set.
        """
        i_max = self._parameters._max_current
        if i_max is None:
            return None
        current1, current2 = self._res_currents_getter(warning=True)
        return float(np.max([abs(current1), abs(current2)])) > i_max

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
            current1, current2 = self._res_currents_getter(warning=True)
            res["results"] = {
                "current1": [current1.real, current1.imag],
                "current2": [current2.real, current2.imag],
            }
        return res

    def _results_to_dict(self, warning: bool, full: bool) -> JsonDict:
        current1, current2 = self._res_currents_getter(warning)
        results = {
            "id": self.id,
            "current1": [current1.real, current1.imag],
            "current2": [current2.real, current2.imag],
        }
        if full:
            potential1, potential2 = self._res_potentials_getter(warning=False)
            results["potential1"] = [potential1.real, potential1.imag]
            results["potential2"] = [potential2.real, potential2.imag]
            power1, power2 = self._res_powers_getter(
                warning=False,
                potential1=potential1,
                potential2=potential2,
                current1=current1,
                current2=current2,
            )
            results["power1"] = [power1.real, power1.imag]
            results["power2"] = [power2.real, power2.imag]
            voltage1, voltage2 = self._res_voltages_getter(warning=False, potential1=potential1, potential2=potential2)
            results["voltage1"] = [voltage1.real, voltage1.imag]
            results["voltage2"] = [voltage2.real, voltage2.imag]
            s = self._res_power_losses_getter(warning=False)
            results["power_losses"] = [s.real, s.imag]
            i = self._res_series_currents_getter(warning=False)
            results["series_current"] = [i.real, i.imag]
            s = self._res_series_power_losses_getter(warning=False)
            results["series_power_loss"] = [s.real, s.imag]
            shunt_current1, shunt_current2 = self._res_shunt_currents_getter(warning=False)
            results["shunt_current1"] = [shunt_current1.real, shunt_current1.imag]
            results["shunt_current2"] = [shunt_current2.real, shunt_current2.imag]
            s = self._res_shunt_power_losses_getter(warning=False)
            results["shunt_power_losses"] = [s.real, s.imag]
        return results
