"""
``optical_diagram``
===================


This package provides a set of classes and functions to create optical diagrams.
They are all available in the ``optical_diagram`` namespace, but this documentation
is organized into sections for clarity.


Base Classes
------------

To create a new optical element, you can inherit from the base classes. They are
documented here for reference as they implement a lot of the core functionality of the
package::

    OpticalElement
    OpticalSystem

Directions and coordinate system
--------------------------------

Helper constants for directions in the 2D plane::

    RIGHT
    LEFT
    UP
    DOWN
    UR
    UL
    DR
    DL
    ORIGIN


Lenses & Refractive Elements
----------------------------

Classes implementing lenses and other refractive elements. They are all subclasses of
a common base class ``Lens`` which implements the basic functionality of a lens::

    ConvergingLens
    DivergingLens


Mirrors & Reflective Elements
-----------------------------

Classes implementing mirrors and other reflective elements. They are all subclasses of
a common base class ``Mirror`` which implements the basic functionality of a mirror::

    PlaneMirror
    ConcaveMirror
    ConvexMirror
    BeamSplitter


Fibers & Waveguides
--------------------

Classes implementing optical fibers, waveguides and related integrated optics components::

    Fiber
    FiberSplitter
    FiberCoupler


Beams & Ray Tracing
-------------------

Classes implementing beam-tracing utilities::

    SimpleBeam
    DivergingBeam
    RayTracedBeam


Annotations & Labels
--------------------

Tools to annotate the optical diagram with labels, arrows, and other visual elements::

    OpticalAxis
    Plane
    Circle
    Point
    Label
    Rectangle
    SurroundingRectangle


Optical Table (figure container)
--------------------------------

Base class to create a figure with an optical table and associated elements::

    OpticalTable


Helpers & Utilities
-------------------

Other utility classes and functions to manage multiple optical elements or perform common
operations::

    Group
    get_axis_direction
    get_normal_direction
    get_normal_vector

"""

import importlib.metadata as _importlib_metadata

__version__ = _importlib_metadata.version("optical_diagram")


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
    OpticalElement,
    OpticalSystem,
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
    Circle,
    Point,
    OpticalAxis,
    Label,
    Rectangle,
    SurroundingRectangle,
)
from ._fiber import Fiber, FiberSplitter, FiberCoupler
from ._table import OpticalTable

__all__ = [  # noqa: RUF022
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
    # Base classes
    "OpticalElement",
    "OpticalSystem",
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
    "Circle",
    "Point",
    "OpticalAxis",
    "Label",
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
