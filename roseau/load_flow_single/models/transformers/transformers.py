import logging

from shapely.geometry.base import BaseGeometry

from roseau.load_flow import TransformerParameters
from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.typing import Id, JsonDict
from roseau.load_flow.units import Q_, ureg_wraps
from roseau.load_flow_engine.cy_engine import CySingleTransformer
from roseau.load_flow_single.models.branches import AbstractBranch
from roseau.load_flow_single.models.buses import Bus

logger = logging.getLogger(__name__)


class Transformer(AbstractBranch):
    """A generic transformer model.

    The model parameters are defined using the ``parameters`` argument.
    """

    def __init__(
        self,
        id: Id,
        bus1: Bus,
        bus2: Bus,
        *,
        parameters: TransformerParameters,
        tap: float = 1.0,
        geometry: BaseGeometry | None = None,
    ) -> None:
        """Transformer constructor.

        Args:
            id:
                A unique ID of the transformer in the network branches.

            bus1:
                Bus to connect the first extremity of the transformer.

            bus2:
                Bus to connect the first extremity of the transformer.

            tap:
                The tap of the transformer, for example 1.02.

            parameters:
                Parameters defining the electrical model of the transformer. This is an instance of
                the :class:`TransformerParameters` class and can be used by multiple transformers.

            geometry:
                The geometry of the transformer.
        """
        assert parameters.type == "single"  # TODO error
        super().__init__(id=id, bus1=bus1, bus2=bus2, geometry=geometry)
        self.tap = tap
        self._parameters = parameters

        z2, ym, k = parameters._z2, parameters._ym, parameters._k
        self._cy_element = CySingleTransformer(z2=z2, ym=ym, k=k * tap)
        self._cy_connect()

    @property
    def tap(self) -> float:
        """The tap of the transformer, for example 1.02."""
        return self._tap

    @tap.setter
    def tap(self, value: float) -> None:
        if value > 1.1:
            logger.warning(f"The provided tap {value:.2f} is higher than 1.1. A good value is between 0.9 and 1.1.")
        if value < 0.9:
            logger.warning(f"The provided tap {value:.2f} is lower than 0.9. A good value is between 0.9 and 1.1.")
        self._tap = value
        self._invalidate_network_results()
        if self._cy_element is not None:
            z2, ym, k = self.parameters._z2, self.parameters._ym, self.parameters._k
            self._cy_element.update_transformer_parameters(z2, ym, k * value)

    @property
    def parameters(self) -> TransformerParameters:
        """The parameters of the transformer."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: TransformerParameters) -> None:
        type1 = self._parameters.type
        type2 = value.type
        if type1 != type2:
            msg = f"The updated type changed for transformer {self.id!r}: {type1} to {type2}."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_TRANSFORMER_TYPE)
        self._parameters = value
        self._invalidate_network_results()
        if self._cy_element is not None:
            z2, ym, k = value._z2, value._ym, value._k
            self._cy_element.update_transformer_parameters(z2, ym, k * self.tap)

    @property
    def max_power(self) -> Q_[float] | None:
        """The maximum power loading of the transformer (in VA)."""
        # Do not add a setter. The user must know that if they change the max_power, it changes
        # for all transformers that share the parameters. It is better to set it on the parameters.
        return self.parameters.max_power

    @property
    @ureg_wraps("VA", (None,))
    def res_power_losses(self) -> Q_[complex]:
        """Get the total power losses in the transformer (in VA)."""
        powers1, powers2 = self._res_powers_getter(warning=True)
        return powers1 + powers2

    @property
    def res_violated(self) -> bool | None:
        """Whether the transformer power exceeds the maximum power (loading > 100%).

        Returns ``None`` if the maximum power is not set.
        """
        s_max = self.parameters._max_power
        if s_max is None:
            return None
        powers1, powers2 = self._res_powers_getter(warning=True)
        # True if either the primary or secondary is overloaded
        return bool((abs(powers1.sum()) > s_max) or (abs(powers2.sum()) > s_max))

    #
    # Json Mixin interface
    #
    def _to_dict(self, include_results: bool) -> JsonDict:
        res = super()._to_dict(include_results=include_results)
        res["tap"] = self.tap
        res["params_id"] = self.parameters.id

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

            power_losses = power1 + power2
            results["power_losses"] = [power_losses.real, power_losses.imag]

        return results
