========
Releases
========

.. currentmodule:: optical_diagram

.. contents:: All Releases
   :depth: 2
   :local:


Version 0.4.2
-------------

- fixed the color property of `Point` and changed the color logic of fibers to ensure
  that the color of the `Fiber` is properly reflected in the end `Points`

Previous Releases
=================

Version 0.4.1
--------------

- Fixed the anchor point of `FiberSplitter` for move_to such that it uses its input point.
- Changed the coordinate system of `OpticalTable` such that it is centered. This breaks backwards compatibility, but I am alone on this project, so I can do that. 
- Updated the examples to reflect the changes to `OpticalTable`.
- Added display tests for `FiberSplitter`.
- Added show_labels method for debugging in `FiberSplitter` and `FiberCoupler`.

Version 0.4.0
-------------

- Added `FiberCoupler` class for modeling 2x2 fiber couplers.
- Improved documentation and examples for `FiberSplitter`.
- Better logic of `FiberSplitter` properties and methods to avoid code duplication and improve maintainability.

Version 0.3.0
-------------

- Added `Group` class for grouping multiple optical elements together. Transforms applied to the group affect all contained elements.
- Added `FiberSplitter` modelling a 1x2 fiber splitter.

Version 0.2.0
-------------

- Split the module `optical_diagram` into a package with submodules for different types of optical elements (lenses, mirrors, fibers, beams).
- Updated documentation to reflect the new structure and added an example of how to use the `BeamSplitter` & `Fiber`.
- Added tests for base classes, lenses & mirrors to make sure that the ray tracing logic is correct and that the properties of the optical elements are properly calculated.

Version 0.1.0
-------------
- Initial release with basic classes for optical elements (lenses, mirrors) and ray tracing functionality.