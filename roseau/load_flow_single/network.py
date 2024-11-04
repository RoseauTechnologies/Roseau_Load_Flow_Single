"""
This module defines the electrical network class.
"""

import logging
import textwrap
import time
import warnings
from collections.abc import Mapping, Sized
from itertools import chain
from typing import TYPE_CHECKING, NoReturn, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from typing_extensions import Self

from roseau.load_flow._solvers import AbstractSolver
from roseau.load_flow.exceptions import RoseauLoadFlowException, RoseauLoadFlowExceptionCode
from roseau.load_flow.io import network_from_dict, network_to_dict
from roseau.load_flow.typing import Id, JsonDict, MapOrSeq, Solver
from roseau.load_flow.utils import JsonMixin, _optional_deps
from roseau.load_flow.utils.types import _DTYPES, LoadTypeDtype
from roseau.load_flow_engine.cy_engine import CyElectricalNetwork, CyGround, CyPotentialRef
from roseau.load_flow_single.models import Transformer
from roseau.load_flow_single.models.branches import AbstractBranch
from roseau.load_flow_single.models.buses import Bus
from roseau.load_flow_single.models.core import Element
from roseau.load_flow_single.models.lines import Line
from roseau.load_flow_single.models.loads import AbstractLoad, PowerLoad
from roseau.load_flow_single.models.loads.loads import CurrentLoad, ImpedanceLoad
from roseau.load_flow_single.models.sources import VoltageSource
from roseau.load_flow_single.models.switches import Switch

if TYPE_CHECKING:
    from networkx import Graph

logger = logging.getLogger(__name__)

_E = TypeVar("_E", bound=Element)


class ElectricalNetwork(JsonMixin):
    """Electrical network class.

    This class represents an electrical network, its elements, and their connections. After
    creating the network, the load flow solver can be run on it using the
    :meth:`solve_load_flow` method.

    Args:
        buses:
            The buses of the network. Either a list of buses or a dictionary of buses with
            their IDs as keys. Buses are the nodes of the network. They connect other elements
            such as loads and sources. Buses can be connected together with branches.

        lines:
            The lines of the network. Either a list of lines or a dictionary of lines with their IDs as keys.

        transformers:
            The transformers of the network. Either a list of transformers or a dictionary of transformers with their IDs as keys.

        switches:
            The switches of the network. Either a list of switches or a dictionary of switches with their IDs as keys.

        loads:
            The loads of the network. Either a list of loads or a dictionary of loads with their
            IDs as keys. There are three types of loads: constant power, constant current, and
            constant impedance.

        sources:
            The sources of the network. Either a list of sources or a dictionary of sources with
            their IDs as keys. A network must have at least one source. Note that two sources
            cannot be connected with a switch.

        crs:
            An optional Coordinate Reference System to use with geo data frames. If not provided,
            the ``EPSG:4326`` CRS will be used.

    Attributes:
        buses (dict[Id, roseau.load_flow.Bus]):
            Dictionary of buses of the network indexed by their IDs. Also available as a
            :attr:`GeoDataFrame<buses_frame>`.

        lines (dict[Id, roseau.load_flow.Line]):
            Dictionary of lines of the network indexed by their IDs. Also available as a
            :attr:`GeoDataFrame<lines_frame>`.

        transformers (dict[Id, roseau.load_flow.Transformer]):
            Dictionary of transformers of the network indexed by their IDs. Also available as a
            :attr:`GeoDataFrame<transformers_frame>`.

        switches (dict[Id, roseau.load_flow.Switch]):
            Dictionary of switches of the network indexed by their IDs. Also available as a
            :attr:`GeoDataFrame<switches_frame>`.

        loads (dict[Id, roseau.load_flow.AbstractLoad]):
            Dictionary of loads of the network indexed by their IDs. Also available as a
            :attr:`DataFrame<loads_frame>`.

        sources (dict[Id, roseau.load_flow.VoltageSource]):
            Dictionary of voltage sources of the network indexed by their IDs. Also available as a
            :attr:`DataFrame<sources_frame>`.
    """

    _DEFAULT_SOLVER: Solver = "newton_goldstein"

    #
    # Methods to build an electrical network
    #
    def __init__(
        self,
        *,
        buses: MapOrSeq[Bus],
        lines: MapOrSeq[Line],
        transformers: MapOrSeq[Transformer],
        switches: MapOrSeq[Switch],
        loads: MapOrSeq[PowerLoad | CurrentLoad | ImpedanceLoad],
        sources: MapOrSeq[VoltageSource],
        crs: str | CRS | None = None,
    ) -> None:
        self.buses: dict[Id, Bus] = self._elements_as_dict(buses, RoseauLoadFlowExceptionCode.BAD_BUS_ID)
        self.lines: dict[Id, Line] = self._elements_as_dict(lines, RoseauLoadFlowExceptionCode.BAD_LINE_ID)
        self.transformers: dict[Id, Transformer] = self._elements_as_dict(
            transformers, RoseauLoadFlowExceptionCode.BAD_TRANSFORMER_ID
        )
        self.switches: dict[Id, Switch] = self._elements_as_dict(switches, RoseauLoadFlowExceptionCode.BAD_SWITCH_ID)
        self.loads: dict[Id, PowerLoad | CurrentLoad | ImpedanceLoad] = self._elements_as_dict(
            loads, RoseauLoadFlowExceptionCode.BAD_LOAD_ID
        )
        self.sources: dict[Id, VoltageSource] = self._elements_as_dict(
            sources, RoseauLoadFlowExceptionCode.BAD_SOURCE_ID
        )

        # Add ground and pref
        self._ground = CyGround()
        self._potential_ref = CyPotentialRef()
        self._ground.connect(self._potential_ref, [(0, 0)])
        for bus in self.buses.values():
            bus._cy_element.connect(self._ground, [(1, 0)])
        for line in self.lines.values():
            if line.with_shunt:
                self._ground.connect(line._cy_element, [(0, 2)])

        self._elements: list[Element] = []
        self._has_loop = False
        self._has_floating_neutral = False
        self._check_validity(constructed=False)
        self._create_network()
        self._valid = True
        self._solver = AbstractSolver.from_dict(data={"name": "newton", "params": {}}, network=self)
        if crs is None:
            crs = "EPSG:4326"
        self.crs: CRS = CRS(crs)

    def __repr__(self) -> str:
        def count_repr(__o: Sized, /, singular: str, plural: str | None = None) -> str:
            """Singular/plural count representation: `1 bus` or `2 buses`."""
            n = len(__o)
            if n == 1:
                return f"{n} {singular}"
            return f"{n} {plural if plural is not None else singular + 's'}"

        return (
            f"<{type(self).__name__}:"
            f" {count_repr(self.buses, 'bus', 'buses')},"
            f" {count_repr(self.lines, 'line', 'lines')},"
            f" {count_repr(self.transformers, 'transformer', 'transformers')},"
            f" {count_repr(self.switches, 'switch', 'switches')},"
            f" {count_repr(self.loads, 'load')},"
            f" {count_repr(self.sources, 'source')},"
            f">"
        )

    @staticmethod
    def _elements_as_dict(elements: MapOrSeq[_E], error_code: RoseauLoadFlowExceptionCode) -> dict[Id, _E]:
        """Convert a sequence or a mapping of elements to a dictionary of elements with their IDs as keys."""
        typ = error_code.name.removeprefix("BAD_").removesuffix("_ID").replace("_", " ")
        elements_dict: dict[Id, _E] = {}
        if isinstance(elements, Mapping):
            for element_id, element in elements.items():
                if element.id != element_id:
                    msg = (
                        f"{typ.capitalize()} ID {element.id!r} does not match its key in the dictionary {element_id!r}."
                    )
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg, code=error_code)
                elements_dict[element_id] = element
        else:
            for element in elements:
                if element.id in elements_dict:
                    msg = f"Duplicate {typ.lower()} ID {element.id!r} in the network."
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg, code=error_code)
                elements_dict[element.id] = element
        return elements_dict

    @classmethod
    def from_element(cls, initial_bus: Bus) -> Self:
        """Construct the network from only one element (bus) and add the others automatically.

        Args:
            initial_bus:
                Any bus of the network. The network is constructed from this bus and all the
                elements connected to it. This is usually the main source bus of the network.

        Returns:
            The network constructed from the given bus and all the elements connected to it.
        """
        buses: list[Bus] = []
        lines: list[Line] = []
        transformers: list[Transformer] = []
        switches: list[Switch] = []
        loads: list[PowerLoad | CurrentLoad | ImpedanceLoad] = []
        sources: list[VoltageSource] = []

        elements: list[Element] = [initial_bus]
        visited_elements: set[Element] = set()
        while elements:
            e = elements.pop(-1)
            visited_elements.add(e)
            if isinstance(e, Bus):
                buses.append(e)
            elif isinstance(e, Line):
                lines.append(e)
            elif isinstance(e, Transformer):
                transformers.append(e)
            elif isinstance(e, Switch):
                switches.append(e)
            elif isinstance(e, AbstractLoad):
                loads.append(e)
            elif isinstance(e, VoltageSource):
                sources.append(e)
            for connected_element in e._connected_elements:
                if connected_element not in visited_elements and connected_element not in elements:
                    elements.append(connected_element)
        return cls(
            buses=buses,
            lines=lines,
            transformers=transformers,
            switches=switches,
            loads=loads,
            sources=sources,
        )

    #
    # Properties to access the data as dataframes
    #
    @property
    def buses_frame(self) -> gpd.GeoDataFrame:
        """The :attr:`buses` of the network as a geo dataframe."""
        data = []
        for bus in self.buses.values():
            min_voltage = bus.min_voltage.magnitude if bus.min_voltage is not None else float("nan")
            max_voltage = bus.max_voltage.magnitude if bus.max_voltage is not None else float("nan")
            data.append((bus.id, min_voltage, max_voltage, bus.geometry))
        return gpd.GeoDataFrame(
            data=pd.DataFrame.from_records(
                data=data,
                columns=["id", "min_voltage", "max_voltage", "geometry"],
                index="id",
            ),
            geometry="geometry",
            crs=self.crs,
        )

    @property
    def lines_frame(self) -> gpd.GeoDataFrame:
        """The :attr:`lines` of the network as a geo dataframe."""
        data = []
        for line in self.lines.values():
            max_current = line.max_current.magnitude if line.max_current is not None else float("nan")
            data.append(
                (
                    line.id,
                    line.bus1.id,
                    line.bus2.id,
                    line.parameters.id,
                    line.length.m,
                    max_current,
                    line.geometry,
                )
            )
        return gpd.GeoDataFrame(
            data=pd.DataFrame.from_records(
                data=data,
                columns=["id", "bus1_id", "bus2_id", "parameters_id", "length", "max_current", "geometry"],
                index="id",
            ),
            geometry="geometry",
            crs=self.crs,
        )

    @property
    def transformers_frame(self) -> gpd.GeoDataFrame:
        """The :attr:`transformers` of the network as a geo dataframe."""
        data = []
        for transformer in self.transformers.values():
            max_power = transformer.max_power.magnitude if transformer.max_power is not None else float("nan")
            data.append(
                (
                    transformer.id,
                    transformer.bus1.id,
                    transformer.bus2.id,
                    transformer.parameters.id,
                    max_power,
                    transformer.geometry,
                )
            )
        return gpd.GeoDataFrame(
            data=pd.DataFrame.from_records(
                data=data,
                columns=["id", "bus1_id", "bus2_id", "parameters_id", "max_power", "geometry"],
                index="id",
            ),
            geometry="geometry",
            crs=self.crs,
        )

    @property
    def switches_frame(self) -> gpd.GeoDataFrame:
        """The :attr:`switches` of the network as a geo dataframe."""
        data = []
        for switch in self.switches.values():
            data.append((switch.id, switch.bus1.id, switch.bus2.id, switch.geometry))
        return gpd.GeoDataFrame(
            data=pd.DataFrame.from_records(
                data=data,
                columns=["id", "bus1_id", "bus2_id", "geometry"],
                index="id",
            ),
            geometry="geometry",
            crs=self.crs,
        )

    @property
    def loads_frame(self) -> pd.DataFrame:
        """The :attr:`loads` of the network as a dataframe."""
        return pd.DataFrame.from_records(
            data=[(load_id, load.type, load.bus.id, load.is_flexible) for load_id, load in self.loads.items()],
            columns=["id", "type", "bus_id", "flexible"],
            index="id",
        )

    @property
    def sources_frame(self) -> pd.DataFrame:
        """The :attr:`sources` of the network as a dataframe."""
        return pd.DataFrame.from_records(
            data=[(source_id, source.bus.id) for source_id, source in self.sources.items()],
            columns=["id", "bus_id"],
            index="id",
        )

    #
    # Helpers to analyze the network
    #
    @property
    def buses_clusters(self) -> list[set[Id]]:
        """Sets of galvanically connected buses, i.e buses connected by lines or a switches.

        This can be useful to isolate parts of the network for localized analysis. For example, to
        study a LV subnetwork of a MV feeder.

        See Also:
            :meth:`Bus.get_connected_buses() <roseau.load_flow.models.Bus.get_connected_buses>`: Get
            the buses in the same galvanically isolated section as a certain bus.
        """
        visited: set[Id] = set()
        result: list[set[Id]] = []
        for bus in self.buses.values():
            if bus.id in visited:
                continue
            bus_cluster = set(bus.get_connected_buses())
            visited |= bus_cluster
            result.append(bus_cluster)
        return result

    def to_graph(self) -> "Graph":
        """Create a networkx graph from this electrical network.

        The graph contains the geometries of the buses in the nodes data and the geometries and
        branch types in the edges data.

        Note:
            This method requires *networkx* to be installed. You can install it with the ``"graph"``
            extra if you are using pip: ``pip install "roseau-load-flow[graph]"``.
        """
        nx = _optional_deps.networkx
        graph = nx.Graph()
        for bus in self.buses.values():
            graph.add_node(bus.id, geom=bus.geometry)
        for line in self.lines.values():
            max_current = line.max_current.magnitude if line.max_current is not None else None
            graph.add_edge(
                line.bus1.id,
                line.bus2.id,
                id=line.id,
                type="line",
                parameters_id=line.parameters.id,
                max_current=max_current,
                geom=line.geometry,
            )
        for transformer in self.transformers.values():
            max_power = transformer.max_power.magnitude if transformer.max_power is not None else None
            graph.add_edge(
                transformer.bus1.id,
                transformer.bus2.id,
                id=transformer.id,
                type="transformer",
                parameters_id=transformer.parameters.id,
                max_power=max_power,
                geom=transformer.geometry,
            )
        for switch in self.switches.values():
            graph.add_edge(switch.bus1.id, switch.bus2.id, id=switch.id, type="switch", geom=switch.geometry)
        return graph

    #
    # Method to solve a load flow
    #
    def solve_load_flow(
        self,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        warm_start: bool = True,
        solver: Solver = _DEFAULT_SOLVER,
        solver_params: JsonDict | None = None,
    ) -> tuple[int, float]:
        """Solve the load flow for this network.

        To get the results of the load flow for the whole network, use the `res_` properties on the
        network (e.g. ``print(net.res_buses``). To get the results for a specific element, use the
        `res_` properties on the element (e.g. ``print(net.buses["bus1"].res_potentials)``.

        You need to activate the license before calling this method. Alternatively you may set the
        environment variable ``ROSEAU_LOAD_FLOW_LICENSE_KEY`` to your license key and it will be
        picked automatically when calling this method. See the :ref:`license` page for more
        information.

        Args:
            max_iterations:
                The maximum number of allowed iterations.

            tolerance:
                Tolerance needed for the convergence.

            warm_start:
                If true (the default), the solver is initialized with the potentials of the last
                successful load flow result (if any). Otherwise, the potentials are reset to their
                initial values.

            solver:
                The name of the solver to use for the load flow. The options are:
                    - ``'newton'``: the classical Newton-Raphson solver.
                    - ``'newton_goldstein'``: the Newton-Raphson solver with the Goldstein and
                      Price linear search.

            solver_params:
                A dictionary of parameters used by the solver. Available parameters depend on the
                solver chosen. For more information, see the :ref:`solvers` page.

        Returns:
            The number of iterations performed and the residual error at the last iteration.
        """
        if not self._valid:
            self._check_validity(constructed=False)
            self._create_network()  # <-- calls _propagate_potentials, no warm start
            self._solver.update_network(self)

        # Update solver
        if solver != self._solver.name:
            solver_params = solver_params if solver_params is not None else {}
            self._solver = AbstractSolver.from_dict(data={"name": solver, "params": solver_params}, network=self)
        elif solver_params is not None:
            self._solver.update_params(solver_params)

        if not warm_start:
            self._reset_inputs()

        start = time.perf_counter()
        try:
            iterations, residual = self._solver.solve_load_flow(max_iterations=max_iterations, tolerance=tolerance)
        except RuntimeError as e:
            self._handle_error(e)

        end = time.perf_counter()

        if iterations == max_iterations:
            msg = (
                f"The load flow did not converge after {iterations} iterations. The norm of the residuals is "
                f"{residual:5n}"
            )
            logger.error(msg=msg)
            raise RoseauLoadFlowException(
                msg, RoseauLoadFlowExceptionCode.NO_LOAD_FLOW_CONVERGENCE, iterations, residual
            )

        logger.debug(f"The load flow converged after {iterations} iterations and {end - start:.3n} s.")
        self._no_results = False

        # Lazily update the results of the elements
        for element in self._elements:
            element._fetch_results = True
            element._no_results = False

        # The results are now valid
        self._results_valid = True

        return iterations, residual

    def _handle_error(self, e: RuntimeError) -> NoReturn:
        msg = e.args[0]
        if msg.startswith("0 "):
            msg = f"The license cannot be validated. The detailed error message is {msg[2:]!r}"
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.LICENSE_ERROR) from e
        elif msg.startswith("1 "):
            msg = msg[2:]
            zero_elements_index, inf_elements_index = self._solver._cy_solver.analyse_jacobian()
            if zero_elements_index:
                zero_elements = [self._elements[i] for i in zero_elements_index]
                printable_elements = ", ".join(f"{type(e).__name__}({e.id!r})" for e in zero_elements)
                msg += (
                    f"The problem seems to come from the elements [{printable_elements}] that have at least one "
                    f"disconnected phase. "
                )
            if inf_elements_index:
                inf_elements = [self._elements[i] for i in inf_elements_index]
                printable_elements = ", ".join(f"{type(e).__name__}({e.id!r})" for e in inf_elements)
                msg += (
                    f"The problem seems to come from the elements [{printable_elements}] that induce infinite "
                    f"values. This might be caused by flexible loads with very high alpha."
                )
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_JACOBIAN) from e
        else:
            assert msg.startswith("2 ")
            msg = msg[2:]
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.NO_BACKWARD_FORWARD) from e

    #
    # Properties to access the load flow results as dataframes
    #
    def _check_valid_results(self) -> None:
        """Check that the results exist and warn if they are invalid."""
        if self._no_results:
            msg = "The load flow results are not available because the load flow has not been run yet."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.LOAD_FLOW_NOT_RUN)

        if not self._results_valid:
            warnings.warn(
                message=(
                    "The results of this network may be outdated. Please re-run a load flow to "
                    "ensure the validity of results."
                ),
                category=UserWarning,
                stacklevel=2,
            )

    @property
    def res_buses(self) -> pd.DataFrame:
        """The load flow results of the network buses.

        The results are returned as a dataframe with the following index:
            - `bus_id`: The id of the bus.

        and the following columns:
            - `potential`: The complex potential of the bus (in Volts).
        """
        self._check_valid_results()
        res_dict = {"bus_id": [], "potential": []}
        dtypes = {c: _DTYPES[c] for c in res_dict}
        for bus_id, bus in self.buses.items():
            potential = bus._res_potential_getter(warning=False)
            res_dict["bus_id"].append(bus_id)
            res_dict["potential"].append(potential)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("bus_id")

    @property
    def res_buses_voltages(self) -> pd.DataFrame:
        """The load flow results of the complex voltages of the buses (V).

        The voltages are computed from the potentials of the buses.

        The results are returned as a dataframe with the following index:
            - `bus_id`: The id of the bus.

        and the following columns:
            - `voltage`: The complex voltage of the bus (in Volts).
            - `min_voltage`: The minimum voltage of the bus (in Volts).
            - `max_voltage`: The maximum voltage of the bus (in Volts).
        """
        self._check_valid_results()
        voltages_dict = {
            "bus_id": [],
            "voltage": [],
            "min_voltage": [],
            "max_voltage": [],
            "violated": [],
        }
        dtypes = {c: _DTYPES[c] for c in voltages_dict}
        for bus_id, bus in self.buses.items():
            min_voltage = bus._min_voltage
            max_voltage = bus._max_voltage
            voltage_limits_set = False

            if min_voltage is None:
                min_voltage = float("nan")
            else:
                voltage_limits_set = True
            if max_voltage is None:
                max_voltage = float("nan")
            else:
                voltage_limits_set = True
            voltage = bus._res_voltage_getter(warning=False)
            voltage_abs = abs(voltage)
            violated = (voltage_abs < min_voltage or voltage_abs > max_voltage) if voltage_limits_set else None
            voltages_dict["bus_id"].append(bus_id)
            voltages_dict["voltage"].append(voltage)
            voltages_dict["min_voltage"].append(min_voltage)
            voltages_dict["max_voltage"].append(max_voltage)
            voltages_dict["violated"].append(violated)
        return pd.DataFrame(voltages_dict).astype(dtypes).set_index("bus_id")

    @property
    def res_lines(self) -> pd.DataFrame:
        """The load flow results of the network lines.

        The results are returned as a dataframe with the following index:
            - `line_id`: The id of the line.

        and the following columns:
            - `current1`: The complex current of the line (in Amps) at the
                first bus.
            - `current2`: The complex current of the line (in Amps) the
                second bus.
            - `power1`: The complex power of the line (in VoltAmps) the
                first bus.
            - `power2`: The complex power of the line (in VoltAmps) the
                second bus.
            - `potential1`: The complex potential of the first bus (in Volts).
            - `potential2`: The complex potential of the second bus (in Volts).
            - `series_loss`: The complex power losses of the line (in VoltAmps)
                due to the series and mutual impedances.
            - `series_current`: The complex current in the series impedance of the line (in Amps).

        Additional information can be easily computed from this dataframe. For example:

        * To get the active power losses, use the real part of the complex power losses
        * To get the total power losses, add the columns ``power1 + power2``
        * To get the power losses in the shunt components of the line, subtract the series losses
          from the total power losses computed in the previous step:
          ``(power1 + power2) - series_loss``
        * To get the currents in the shunt components of the line:
          - For the first bus, subtract the columns ``current1 - series_current``
          - For the second bus, add the columns ``series_current + current2``
        """
        self._check_valid_results()
        res_dict = {
            "line_id": [],
            "current1": [],
            "current2": [],
            "power1": [],
            "power2": [],
            "potential1": [],
            "potential2": [],
            "series_losses": [],
            "series_current": [],
            "max_current": [],
            "violated": [],
        }
        dtypes = {c: _DTYPES[c] for c in res_dict}
        for line in self.lines.values():
            current1, current2 = line._res_currents_getter(warning=False)
            potential1, potential2 = line._res_potentials_getter(warning=False)
            du_line, series_current = line._res_series_values_getter(warning=False)
            power1 = potential1 * current1.conj()
            power2 = potential2 * current2.conj()
            series_loss = du_line * series_current.conj()
            i_max = line.parameters._max_current
            violated = None if i_max is None else (abs(current1) > i_max or abs(current2) > i_max)
            res_dict["line_id"].append(line.id)
            res_dict["current1"].append(current1)
            res_dict["current2"].append(current2)
            res_dict["power1"].append(power1)
            res_dict["power2"].append(power2)
            res_dict["potential1"].append(potential1)
            res_dict["potential2"].append(potential2)
            res_dict["series_losses"].append(series_loss)
            res_dict["series_current"].append(series_current)
            res_dict["max_current"].append(i_max)
            res_dict["violated"].append(violated)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("line_id")

    @property
    def res_transformers(self) -> pd.DataFrame:
        """The load flow results of the network transformers.

        The results are returned as a dataframe with the following index:
            - `transformer_id`: The id of the transformer.

        and the following columns:
            - `current1`: The complex current of the transformer (in Amps) at the first bus.
            - `current2`: The complex current of the transformer (in Amps) at the second bus.
            - `power1`: The complex power of the transformer (in VoltAmps) at the first bus.
            - `power2`: The complex power of the transformer (in VoltAmps) at the second bus.
            - `potential1`: The complex potential of the first bus (in Volts).
            - `potential2`: The complex potential of the second bus (in Volts).
            - `max_power`: The maximum power loading (in VoltAmps) of the transformer.
        """
        self._check_valid_results()
        res_dict = {
            "transformer_id": [],
            "current1": [],
            "current2": [],
            "power1": [],
            "power2": [],
            "potential1": [],
            "potential2": [],
            "max_power": [],
            "violated": [],
        }
        dtypes = {c: _DTYPES[c] for c in res_dict}
        for transformer in self.transformers.values():
            current1, current2 = transformer._res_currents_getter(warning=False)
            potential1, potential2 = transformer._res_potentials_getter(warning=False)
            power1 = potential1 * current1.conj()
            power2 = potential2 * current2.conj()
            s_max = transformer.parameters._max_power
            violated = (abs(power1) > s_max or abs(power2) > s_max) if s_max is not None else None
            res_dict["transformer_id"].append(transformer.id)
            res_dict["current1"].append(current1)
            res_dict["current2"].append(current2)
            res_dict["power1"].append(power1)
            res_dict["power2"].append(power2)
            res_dict["potential1"].append(potential1)
            res_dict["potential2"].append(potential2)
            res_dict["max_power"].append(s_max)
            res_dict["violated"].append(violated)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("transformer_id")

    @property
    def res_switches(self) -> pd.DataFrame:
        """The load flow results of the network switches.

        The results are returned as a dataframe with the following index:
            - `switch_id`: The id of the switch.

        and the following columns:
            - `current1`: The complex current of the switch (in Amps) at the first bus.
            - `current2`: The complex current of the switch (in Amps) at the second bus.
            - `power1`: The complex power of the switch (in VoltAmps) at the first bus.
            - `power2`: The complex power of the switch (in VoltAmps) at the second bus.
            - `potential1`: The complex potential of the first bus (in Volts).
            - `potential2`: The complex potential of the second bus (in Volts).
        """
        self._check_valid_results()
        res_dict = {
            "switch_id": [],
            "current1": [],
            "current2": [],
            "power1": [],
            "power2": [],
            "potential1": [],
            "potential2": [],
        }
        dtypes = {c: _DTYPES[c] for c in res_dict}
        for switch in self.switches.values():
            if not isinstance(switch, Switch):
                continue
            current1, current2 = switch._res_currents_getter(warning=False)
            potential1, potential2 = switch._res_potentials_getter(warning=False)
            power1 = potential1 * current1.conj()
            power2 = potential2 * current2.conj()
            res_dict["switch_id"].append(switch.id)
            res_dict["current1"].append(current1)
            res_dict["current2"].append(current2)
            res_dict["power1"].append(power1)
            res_dict["power2"].append(power2)
            res_dict["potential1"].append(potential1)
            res_dict["potential2"].append(potential2)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("switch_id")

    @property
    def res_loads(self) -> pd.DataFrame:
        """The load flow results of the network loads.

        The results are returned as a dataframe with the following index:
            - `load_id`: The id of the load.

        and the following columns:
            - `type`: The type of the load, can be ``{'power', 'current', 'impedance'}``.
            - `current`: The complex current of the load (in Amps).
            - `power`: The complex power of the load (in VoltAmps).
            - `potential`: The complex potential of the load (in Volts).
        """
        self._check_valid_results()
        res_dict = {"load_id": [], "type": [], "current": [], "power": [], "potential": []}
        dtypes = {c: _DTYPES[c] for c in res_dict} | {"type": LoadTypeDtype}
        for load_id, load in self.loads.items():
            current = load._res_current_getter(warning=False)
            potential = load._res_potential_getter(warning=False)
            power = potential * current.conj()
            res_dict["load_id"].append(load_id)
            res_dict["type"].append(load.type)
            res_dict["current"].append(current)
            res_dict["power"].append(power)
            res_dict["potential"].append(potential)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("load_id")

    @property
    def res_loads_voltages(self) -> pd.DataFrame:
        """The load flow results of the complex voltages of the loads (V).

        The results are returned as a dataframe with the following index:
            - `load_id`: The id of the load.

        and the following columns:
            - `type`: The type of the load, can be ``{'power', 'current', 'impedance'}``.s
            - `voltage`: The complex voltage of the load (in Volts).
        """
        self._check_valid_results()
        voltages_dict = {"load_id": [], "type": [], "voltage": []}
        dtypes = {c: _DTYPES[c] for c in voltages_dict} | {"type": LoadTypeDtype}
        for load_id, load in self.loads.items():
            voltage = load._res_voltage_getter(warning=False)
            voltages_dict["load_id"].append(load_id)
            voltages_dict["type"].append(load.type)
            voltages_dict["voltage"].append(voltage)
        return pd.DataFrame(voltages_dict).astype(dtypes).set_index("load_id")

    @property
    def res_loads_flexible_powers(self) -> pd.DataFrame:
        """The load flow results of the flexible powers of the "flexible" loads.

        The results are returned as a dataframe with the following index:
            - `load_id`: The id of the load.

        and the following columns:
            - `power`: The complex flexible power of the load (in VoltAmps).

        Note that the flexible powers are the powers that flow in the load elements and not in the
        lines. These are only different in case of delta loads. To access the powers that flow in
        the lines, use the ``power`` column from the :attr:`res_loads` property instead.
        """
        self._check_valid_results()
        loads_dict = {"load_id": [], "flexible_power": []}
        dtypes = {c: _DTYPES[c] for c in loads_dict}
        for load_id, load in self.loads.items():
            if not (isinstance(load, PowerLoad) and load.is_flexible):
                continue
            flexible_power = load._res_flexible_power_getter(warning=False)
            loads_dict["load_id"].append(load_id)
            loads_dict["flexible_power"].append(flexible_power)
        return pd.DataFrame(loads_dict).astype(dtypes).set_index("load_id")

    @property
    def res_sources(self) -> pd.DataFrame:
        """The load flow results of the network sources.

        The results are returned as a dataframe with the following index:
            - `source_id`: The id of the source.

        and the following columns:
            - `current`: The complex current of the source (in Amps).
            - `power`: The complex power of the source (in VoltAmps).
            - `potential`: The complex potential of the source (in Volts).
        """
        self._check_valid_results()
        res_dict = {"source_id": [], "current": [], "power": [], "potential": []}
        dtypes = {c: _DTYPES[c] for c in res_dict}
        for source_id, source in self.sources.items():
            current = source._res_current_getter(warning=False)
            potential = source._res_potential_getter(warning=False)
            power = potential * current.conj()
            res_dict["source_id"].append(source_id)
            res_dict["current"].append(current)
            res_dict["power"].append(power)
            res_dict["potential"].append(potential)
        return pd.DataFrame(res_dict).astype(dtypes).set_index("source_id")

    #
    # Internal methods, please do not use
    #
    def _connect_element(self, element: Element) -> None:
        """Connect an element to the network.

        When an element is added to the network, extra processing is done to keep the network valid. This method is
        used in the by the `network` setter of `Element` instances to add the element to the internal dictionary of
        `self`.

        Args:
            element:
                The element to add. Only lines, loads, buses and sources can be added.
        """
        # The C++ electrical network and the tape will be recomputed
        container: dict[Id, Element]
        can_disconnect = False
        if isinstance(element, Bus):
            container, element_type = self.buses, "bus"
        elif isinstance(element, AbstractLoad):
            container, element_type = self.loads, "load"
            can_disconnect = True
        elif isinstance(element, Line):
            container, element_type = self.lines, "line"
        elif isinstance(element, Transformer):
            container, element_type = self.transformers, "transformer"
        elif isinstance(element, Switch):
            container, element_type = self.switches, "switch"
        elif isinstance(element, VoltageSource):
            container, element_type = self.sources, "source"
            can_disconnect = True
        else:
            msg = f"Unknown element {element} can not be added to the network."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT)
        if element.id in container and container[element.id] is not element:
            element._disconnect()  # Don't leave it lingering in other elemnets _connected_elements
            msg = f"A {element_type} of ID {element.id!r} is already connected to the network."
            if can_disconnect:
                msg += f" Disconnect the old {element_type} first if you meant to replace it."
            logger.error(msg)
            raise RoseauLoadFlowException(msg, RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT)
        container[element.id] = element
        element._network = self
        if isinstance(element, Bus):
            element._cy_element.connect(self._ground, [(1, 0)])
        elif isinstance(element, Line) and element.with_shunt:
            self._ground.connect(element._cy_element, [(0, 2)])
        self._valid = False
        self._results_valid = False

    def _disconnect_element(self, element: Element) -> None:
        """Remove an element of the network.

        When an element is removed from the network, extra processing is needed to keep the network valid. This method
        is used in the by the `network` setter of `Element` instances (when the provided network is `None`) to remove
        the element to the internal dictionary of `self`.

        Args:
            element:
                The element to remove.
        """
        # The C++ electrical network and the tape will be recomputed
        if isinstance(element, Bus | AbstractBranch):
            msg = f"{element!r} is a {type(element).__name__} and it cannot be disconnected from a network."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT)
        elif isinstance(element, AbstractLoad):
            self.loads.pop(element.id)
        elif isinstance(element, VoltageSource):
            self.sources.pop(element.id)
        else:
            msg = f"{element!r} is not a valid load or source."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.BAD_ELEMENT_OBJECT)
        element._network = None
        self._valid = False
        self._results_valid = False

    def _create_network(self) -> None:
        """Create the Cython and C++ electrical network of all the passed elements."""
        self._valid = True
        self._propagate_potentials()
        cy_elements = []
        for element in self._elements:
            cy_elements.append(element._cy_element)
        self._cy_electrical_network = CyElectricalNetwork(elements=np.array(cy_elements), nb_elements=len(cy_elements))

    def _check_validity(self, constructed: bool) -> None:
        """Check the validity of the network to avoid having a singular jacobian matrix. It also assigns the `self`
        to the network field of elements.

        Args:
            constructed:
                True if the network is already constructed, and we have added an element, False
                otherwise.
        """
        elements: set[Element] = set()
        elements.update(self.buses.values())
        elements.update(self.lines.values())
        elements.update(self.transformers.values())
        elements.update(self.switches.values())
        elements.update(self.loads.values())
        elements.update(self.sources.values())

        if not elements:
            msg = "Cannot create a network without elements."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.EMPTY_NETWORK)

        found_source = False
        for element in elements:
            # Check connected elements and check network assignment
            for adj_element in element._connected_elements:
                if adj_element not in elements:
                    msg = (
                        f"{type(adj_element).__name__} element ({adj_element.id!r}) is connected "
                        f"to {type(element).__name__} element ({element.id!r}) but "
                    )
                    if constructed:
                        msg += "was not passed to the ElectricalNetwork constructor."
                    else:
                        msg += "has not been added to the network. It must be added with 'connect'."
                    logger.error(msg)
                    raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.UNKNOWN_ELEMENT)

            # Check that there is at least a `VoltageSource` element in the network
            if isinstance(element, VoltageSource):
                found_source = True

        # Raises an error if no voltage sources
        if not found_source:
            msg = "There is no voltage source provided in the network, you must provide at least one."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.NO_VOLTAGE_SOURCE)

        # Assign the network
        for element in elements:
            if element.network is None:
                element._network = self
            elif element.network != self:
                element._raise_several_network()

    def _reset_inputs(self) -> None:
        """Reset the input vector used for the first step of the newton algorithm to its initial value."""
        if self._solver is not None:
            self._solver.reset_inputs()

    def _propagate_potentials(self) -> None:
        """Set the bus potentials that have not been initialized yet and compute self._elements order."""
        starting_potential, starting_source = self._get_potential()
        elements = [(starting_source, starting_potential, None)]
        self._elements = []
        self._has_loop = False
        visited = {starting_source}
        while elements:
            element, potential, parent = elements.pop(-1)
            self._elements.append(element)
            if isinstance(element, Bus) and not element._initialized:
                element.potential = potential
                element._initialized_by_the_user = False  # only used for serialization
            for e in element._connected_elements:
                if e not in visited:
                    if isinstance(element, Transformer):
                        k = element.parameters._ulv / element.parameters._uhv
                        new_potential = potential.copy()
                        for key, p in new_potential.items():
                            new_potential[key] = p * k
                        elements.append((e, new_potential, element))
                        visited.add(e)
                    else:
                        elements.append((e, potential, element))
                        visited.add(e)
                elif parent != e:
                    self._has_loop = True

        if len(visited) < len(self.buses) + len(self.lines) + len(self.transformers) + len(self.switches) + len(
            self.loads
        ) + len(self.sources):
            unconnected_elements = [
                element
                for element in chain(
                    self.buses.values(),
                    self.lines.values(),
                    self.transformers.values(),
                    self.switches.values(),
                    self.loads.values(),
                    self.sources.values(),
                )
                if element not in visited
            ]
            printable_elements = textwrap.wrap(
                ", ".join(f"{type(e).__name__}({e.id!r})" for e in unconnected_elements), 500
            )
            msg = f"The elements {printable_elements} are not electrically connected to a voltage source."
            logger.error(msg)
            raise RoseauLoadFlowException(msg=msg, code=RoseauLoadFlowExceptionCode.POORLY_CONNECTED_ELEMENT)

    def _get_potential(self) -> tuple[complex, VoltageSource]:
        """Compute initial potentials from the voltages sources of the network, get also the starting source"""
        starting_source = None
        potential = None
        # if there are multiple voltage sources, start from the higher one (the last one in the sorted below)
        for source in sorted(self.sources.values(), key=lambda x: np.average(np.abs(x._voltage))):
            source_voltage = source._voltage
            starting_source = source
            potential = source_voltage

        return potential, starting_source

    #
    # Network saving/loading
    #
    @classmethod
    def from_dict(cls, data: JsonDict, *, include_results: bool = True) -> Self:
        """Construct an electrical network from a dict created with :meth:`to_dict`.

        Args:
            data:
                The dictionary containing the network data.

            include_results:
                If True (default) and the results of the load flow are included in the dictionary,
                the results are also loaded into the element.

        Returns:
            The constructed network.
        """
        buses, lines, transformers, switches, loads, sources, grounds, p_refs, has_results = network_from_dict(
            data=data, include_results=include_results
        )
        network = cls(
            buses=buses,
            lines=lines,
            transformers=transformers,
            switches=switches,
            loads=loads,
            sources=sources,
        )
        network._no_results = not has_results
        network._results_valid = has_results
        return network

    def _to_dict(self, include_results: bool) -> JsonDict:
        return network_to_dict(en=self, include_results=include_results)

    #
    # Results saving
    #
    def _results_to_dict(self, warning: bool, full: bool) -> JsonDict:
        """Get the voltages and currents computed by the load flow and return them as a dict."""
        if warning:
            self._check_valid_results()  # Warn only once if asked
        return {
            "buses": [bus._results_to_dict(warning=False, full=full) for bus in self.buses.values()],
            "lines": [line._results_to_dict(warning=False, full=full) for line in self.lines.values()],
            "transformers": [
                transformer._results_to_dict(warning=False, full=full) for transformer in self.transformers.values()
            ],
            "switches": [switch._results_to_dict(warning=False, full=full) for switch in self.switches.values()],
            "loads": [load._results_to_dict(warning=False, full=full) for load in self.loads.values()],
            "sources": [source._results_to_dict(warning=False, full=full) for source in self.sources.values()],
        }
