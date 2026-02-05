API reference
=============


.. We need to change the current module to the base module as these classes are not imported into the main module. Otherwise, they won't be included in the API reference.

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


.. On to the rest of the API reference, we change back to the main module so that the classes are documented under the main module instead of the submodules.

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

   Group
   get_axis_direction
   get_normal_direction
   get_normal_vector