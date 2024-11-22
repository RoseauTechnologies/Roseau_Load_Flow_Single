import logging

import numpy as np

from roseau.load_flow.models.lines.parameters import LineParameters as TriLineParameters
from roseau.load_flow.typing import ComplexArrayLike2D, Id
from roseau.load_flow.units import Q_, ureg_wraps
from roseau.load_flow.utils import Insulator, LineType, Material

logger = logging.getLogger(__name__)


class LineParameters(TriLineParameters):
    """Parameters that define electrical models of lines."""

    @ureg_wraps(None, (None, None, "ohm/km", "S/km", "A", None, None, None, "mm²"))
    def __init__(
        self,
        id: Id,
        z_line: complex | ComplexArrayLike2D,
        y_shunt: complex | ComplexArrayLike2D | None = None,
        ampacities: float | None = None,
        line_type: LineType | None = None,
        materials: Material | None = None,
        insulators: Insulator | None = None,
        sections: float | Q_[float] | None = None,
    ) -> None:
        """LineParameters constructor.

        Args:
            id:
                A unique ID of the line parameters, typically its canonical name.

            z_line:
                 The Z matrix of the line (Ohm/km).

            y_shunt:
                The Y matrix of the line (Siemens/km). This field is optional if the line has no shunt part.

            ampacities:
                The ampacities of the line (A). The ampacities are optional, they are
                not used in the load flow but can be used to check for overloading.
                See also :meth:`Line.res_violated <roseau.load_flow.Line.res_violated>`.

            line_type:
                The type of the line (overhead, underground, twisted). The line type is optional,
                it is informative only and is not used in the load flow. This field gets
                automatically filled when the line parameters are created from a geometric model or
                from the catalogue.

            materials:
                The type of the conductor material (Aluminum, Copper, ...). The material is
                optional, it is informative only and is not used in the load flow. This field gets
                automatically filled when the line parameters are created from a geometric model or
                from the catalogue.

            insulators:
                The type of the cable insulator (PVC, XLPE, ...). The insulator is optional,
                it is informative only and is not used in the load flow. This field gets
                automatically filled when the line parameters are created from a geometric model or
                from the catalogue.
        """
        z_line_tri = [[z_line]] if np.isscalar(z_line) else z_line
        y_shunt_tri = [[y_shunt]] if y_shunt is not None and np.isscalar(y_shunt) else y_shunt
        super().__init__(
            id,
            z_line=z_line_tri,
            y_shunt=y_shunt_tri,
            ampacities=ampacities,
            line_type=line_type,
            materials=materials,
            insulators=insulators,
            sections=sections,
        )
