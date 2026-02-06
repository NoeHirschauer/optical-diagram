"""
``optical_diagram``
===================
"""


__version__ = "0.4.2"


from ._base import (
    RIGHT,
    LEFT,
    UP,
    DOWN,
    UR,
    UL,
    DR,
    DL,
    ORIGIN,
    Group,
    get_axis_direction,
    get_normal_direction,
    get_normal_vector,
)
from ._mirrors import PlaneMirror, ConcaveMirror, ConvexMirror, BeamSplitter
from ._lenses import ConvergingLens, DivergingLens
from ._beams import SimpleBeam, DivergingBeam, RayTracedBeam
from ._annotations import (
    Plane,
    Point,
    OpticalAxis,
    Label,
    Arrow,
    Rectangle,
    SurroundingRectangle,
)
from ._fiber import Fiber, FiberSplitter, FiberCoupler
from ._table import OpticalTable

__all__ = [
    # Directions
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "UR",
    "UL",
    "DR",
    "DL",
    "ORIGIN",

    # Mirrors
    "PlaneMirror",
    "ConcaveMirror",
    "ConvexMirror",
    "BeamSplitter",
    
    # Lenses
    "ConvergingLens",
    "DivergingLens",

    # Annotations
    "Plane",
    "Point",
    "OpticalAxis",
    "Label",
    "Arrow",
    "Rectangle",
    "SurroundingRectangle",
    
    # Fiber components
    "Fiber",
    "FiberSplitter",
    "FiberCoupler",

    # Beams
    "SimpleBeam",
    "DivergingBeam",
    "RayTracedBeam",
    
    # Table
    "OpticalTable",
    
    # Utilities
    "Group",
    "get_axis_direction",
    "get_normal_direction",
    "get_normal_vector",
]