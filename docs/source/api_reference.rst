``optical_diagram`` reference
=============================

.. currentmodule:: optical_diagram._base

Base Classes
------------

.. Document the base classes
.. autosummary::
   :toctree: _autosummary/_base
   :caption: Base Classes
   :template: autosummary/class.rst
   :nosignatures:

   OpticalElement
   OpticalSystem
   Group

.. currentmodule:: optical_diagram

Lenses & Refractive Elements
----------------------------

.. autosummary::
   :toctree: _autosummary/_lenses
   :caption: Lenses & Refractive Elements
   :template: autosummary/class.rst
   :nosignatures:

   ConvergingLens
   DivergingLens

Mirrors & Reflective Elements
-----------------------------

.. autosummary::
   :toctree: _autosummary/_mirrors
   :caption: Mirrors & Reflective Elements
   :template: autosummary/class.rst
   :nosignatures:

   PlaneMirror
   ConcaveMirror
   ConvexMirror
   BeamSplitter

Fibers & Waveguides
--------------------
.. autosummary::
   :toctree: _autosummary/_fibers
   :caption: Fibers & Waveguides
   :template: autosummary/class.rst
   :nosignatures:

   Fiber
   FiberSplitter
   FiberCoupler

Beams & Ray Tracing
-------------------

.. autosummary::
   :toctree: _autosummary/_beams
   :caption: Beams & "Ray Tracing"
   :template: autosummary/class.rst
   :nosignatures:

   SimpleBeam
   DivergingBeam
   RayTracedBeam

Annotations & Labels
--------------------

.. autosummary::
   :toctree: _autosummary/_annotations
   :caption: Annotations & Labels
   :template: autosummary/class.rst
   :nosignatures:

   OpticalAxis
   Plane
   Point
   Label
   Rectangle
   SurroundingRectangle

Optical Table (figure container)
--------------------------------

.. autosummary::
   :toctree: _autosummary/_table
   :caption: Optical Table (figure container)
   :template: autosummary/class.rst
   :nosignatures:

   OpticalTable

Helpers & Utilities
-------------------

.. autosummary::
   :toctree: _autosummary/_base
   :caption: Helpers & Utilities
   :template: autosummary/function.rst
   :nosignatures:

   get_axis_direction
   get_normal_direction
   get_normal_vector