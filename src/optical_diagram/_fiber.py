from typing import Literal, Self

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy.typing import ArrayLike

from ._annotations import Point
from ._base import RIGHT, UP, Group, OpticalElement, get_normal_vector

__all__ = ["Fiber", "FiberSplitter", "FiberCoupler"]  # noqa: RUF022


class Fiber(OpticalElement):
    """
    Connects two points or elements with a smooth cubic spline.
    Endpoints are full Point objects, supporting raytracing and alignment.
    """

    def __init__(
        self,
        start: ArrayLike | OpticalElement,
        end: ArrayLike | OpticalElement,
        angle_start: float = 0.0,
        angle_end: float = 0.0,
        stiffness: float = 0.5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        start : ArrayLike or OpticalElement
            Starting coordinates or an element with .center used as start position.
        end : ArrayLike or OpticalElement
            Ending coordinates or an element with .center used as end position.
        angle_start : float, optional
            Tangent angle at the start point in degrees. Default is 0.0.
        angle_end : float, optional
            Tangent angle at the end point in degrees. Default is 0.0.
        stiffness : float, optional
            Stiffness of the fiber. Default is 0.5.
        **kwargs : dict, optional
            styling forwarded to the base OpticalElement class.

        """

        # 1. Resolve initial coordinates
        p0_pos = start.center if hasattr(start, "center") else np.asarray(start)
        p3_pos = end.center if hasattr(end, "center") else np.asarray(end)

        # 2. Set default styles
        kwargs.setdefault("color", "C0")
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("zorder", 4)

        # 3. Create endpoints as Point objects
        # We pass the fiber's color to the dots so they match by default
        dot_color = kwargs.get("color")
        self.start_point: Point = Point(p0_pos, color=dot_color)
        """The starting point of the fiber."""
        self.end_point: Point = Point(p3_pos, color=dot_color)
        """The ending point of the fiber."""

        self.angle_start = float(angle_start)
        self.angle_end = float(angle_end)
        self.stiffness = stiffness
        self._show_start = False
        self._show_end = False

        # 4. Initialize base at the midpoint
        midpoint = (self.start_point.center + self.end_point.center) / 2.0
        super().__init__(midpoint, size=1.0, angle=0.0, **kwargs)

        # override the color property to ensure it is applied to everything
        if "color" in kwargs:
            self.color = kwargs.pop("color")

    # --- Geometry & Bounding Box ---

    def _get_bezier_points(self):
        """Internal helper to calculate the 4 Cubic Bezier points using Point.center."""
        p0 = self.start_point.center
        p3 = self.end_point.center
        dist = np.linalg.norm(p3 - p0)
        L = dist * self.stiffness

        rad_s = np.deg2rad(self.angle_start)
        p1 = p0 + L * np.array([np.cos(rad_s), np.sin(rad_s)])

        rad_e = np.deg2rad(self.angle_end)
        p2 = p3 - L * np.array([np.cos(rad_e), np.sin(rad_e)])

        return p0, p1, p2, p3

    def _get_local_points(self):
        """Returns the 4 Bezier control points relative to the fiber center."""
        pts = np.array(self._get_bezier_points())
        return pts - self.center

    # --- Transformation Overrides ---
    # We delegate the math to the Point objects, then update Fiber.center

    def shift(self, vector):
        self.start_point.shift(vector)
        self.end_point.shift(vector)
        return super().shift(vector)

    def rotate(self, angle_degrees, about_point=None):
        # Determine pivot
        pivot = about_point.center if hasattr(about_point, "center") else about_point
        if pivot is None:
            pivot = self.center

        # Rotate endpoints as individual OpticalElements
        self.start_point.rotate(angle_degrees, about_point=pivot)
        self.end_point.rotate(angle_degrees, about_point=pivot)

        # Update fiber-specific tangent angles
        self.angle_start += angle_degrees
        self.angle_end += angle_degrees

        return super().rotate(angle_degrees, about_point)

    def flip(self, axis=UP, about_point=None):
        # Determine pivot
        pivot = about_point.center if hasattr(about_point, "center") else about_point
        if pivot is None:
            pivot = self.center

        # Flip endpoints as individual OpticalElements
        self.start_point.flip(axis=axis, about_point=pivot)
        self.end_point.flip(axis=axis, about_point=pivot)

        # Reflect tangent angles: 2*phi - theta
        axis_unit = np.asarray(axis) / np.linalg.norm(axis)
        axis_angle = np.degrees(np.arctan2(axis_unit[1], axis_unit[0]))
        self.angle_start = 2 * axis_angle - self.angle_start
        self.angle_end = 2 * axis_angle - self.angle_end

        return super().flip(axis, about_point)

    # --- Drawing ---

    def show_connections(self, start: bool = True, end: bool = True):
        """
        Toggle visibility of the endpoint markers.

        Parameters
        ----------
        start : bool, optional
            Whether to show the starting point marker. Default is True.
        end : bool, optional
            Whether to show the ending point marker. Default is True.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self._show_start = start
        self._show_end = end
        return self

    # Override facecolor so fibers cannot be filled (avoids closed, filled bezier patch)
    @property
    def facecolor(self):
        """Fiber is not fillable; this always returns 'none'."""
        return "none"

    @facecolor.setter
    def facecolor(self, value):
        # ignore attempts to set a fill on Fiber; ensure it stays 'none'
        self._style["facecolor"] = "none"

    @property
    def color(self):
        """Shorthand color. Proxies to edgecolor for the fiber line"""
        return self._style.get("edgecolor")

    @color.setter
    def color(self, value):
        """Set shorthand color for the fiber line (proxies to edgecolor).

        This also updates the `color` of the endpoint markers to match the fiber line
        color for visual consistency.
        """
        if value is None:
            self._style.pop("edgecolor", None)
        else:
            self._style["edgecolor"] = value
            # ensure facecolor is set to none
            self._style["facecolor"] = "none"

            # Update endpoint colors to match the fiber line color
            self.start_point.color = value
            self.end_point.color = value

    def _get_mpl_artist(self):
        p0, p1, p2, p3 = self._get_bezier_points()
        verts = [p0, p1, p2, p3]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        return PathPatch(Path(verts, codes, closed=False), **self._style)

    def draw(self, ax):
        # 1. Draw the fiber path
        ax.add_patch(self._get_mpl_artist())

        # 2. Draw the Points only if toggled
        if self._show_start:
            ax.add_patch(self.start_point._get_mpl_artist())
        if self._show_end:
            ax.add_patch(self.end_point._get_mpl_artist())


class FiberSplitter(Group):
    """Composite element: `Group` of two `Fibers` sharing a common input `Point`.

    Notes
    -----

    The two `Fiber` objects are the only elements in the `Group`, and they share a common
    input `Point` at the start of the splitter. Transformations applied to the
    `FiberSplitter` are propagated to the internal `Fiber` objects, ensuring that the
    splitter behaves as a single cohesive element in the optical diagram.

    """

    def __init__(
        self,
        input_pos: ArrayLike | OpticalElement,
        axis: ArrayLike = RIGHT,
        length: float = 1.0,
        height: float = 1.0,
        alignment: Literal["center", "top", "bottom"] = "center",
        **kwargs,
    ):
        """
        Parameters
        ----------
        input_pos : ArrayLike or OpticalElement
            Input coordinates or an element with .center used as input position.
        axis : ArrayLike, optional
            Axial direction from outputs toward the input. Defaults to RIGHT
        length : float, optional
            Axial distance from outputs to input. Default is 1.0.
        height : float, optional
            Transverse separation between the two outputs. Default is 1.0.
        alignment : {"center", "top", "bottom"}, optional
            determines which transverse location the input is aligned to. Default is
            ``'center'``. The definition of the different options is given below for an
            axis pointing to the right:

            - ``'top'`` ::

                input_point ──┬── out_a_point
                              │
                              └── out_b_point

            - ``'center'`` ::

                              ┌── out_a_point
                input_point ──┤
                              └── out_b_point

            - ``'bottom'`` ::

                              ┌── out_a_point
                              │
                input_point ──┴── out_b_point

        **kwargs : dict, optional
            styling forwarded to the internal Fiber objects
        """
        # styling defaults
        kwargs.setdefault("color", "C0")

        # Resolve input position
        input_pos = (
            input_pos.center if hasattr(input_pos, "center") else np.asarray(input_pos)
        )

        # Normalize axis and compute transverse (perp) direction
        axis = np.asarray(axis, dtype=float)
        if np.allclose(axis, 0):
            raise ValueError("axis must be non-zero")
        axis = axis / np.linalg.norm(axis)
        axis_transverse = get_normal_vector(axis)

        # Compute output axial positions
        out_a_pos = input_pos + length * axis
        out_b_pos = input_pos + length * axis

        # Compute output transverse position based on alignment
        if alignment == "center":
            # shift both outputs equally
            out_a_pos += (height / 2) * axis_transverse
            out_b_pos -= (height / 2) * axis_transverse
        elif alignment == "top":
            # shift output B only
            out_b_pos -= height * axis_transverse
        elif alignment == "bottom":
            # shift output A only
            out_a_pos += height * axis_transverse
        else:
            raise ValueError("alignment must be 'center', 'top' or 'bottom'")

        # Create internal Fibers and and map their Points to this object's Points
        fa = Fiber(input_pos, out_a_pos, **kwargs)
        fb = Fiber(input_pos, out_b_pos, **kwargs)

        self.input_point: Point = fa.start_point  # shared input Point
        """The input point of the splitter."""
        self.out_a_point: Point = fa.end_point  # output A Point
        """The output point A of the splitter."""
        self.out_b_point: Point = fb.end_point  # output B Point
        """The output point B of the splitter."""
        self.fiber_a: Fiber = fa
        """The internal Fiber connecting the input to output A."""
        self.fiber_b: Fiber = fb
        """The internal Fiber connecting the input to output B."""

        # Initialize as a Group containing only the Fibers for correct transforms
        elements = [self.fiber_a, self.fiber_b]
        super().__init__(elements, **kwargs)

        if "color" in kwargs:
            self.color = kwargs.pop("color")

        # We need to update the points if any changes to length/height/alignment occur
        # For this, we save the properties internally.
        self._length = length
        self._height = height
        self._alignment = alignment
        self._axis = axis
        self._axis_transverse = get_normal_vector(self._axis)

        # show flags (used to set fiber marker visibility)
        self._show_input = False
        self._show_out_a = False
        self._show_out_b = False
        self._show_labels = (
            False  # controls optional debug/annotation labels on splitter points
        )

    @property
    def color(self):
        """Shorthand color. Proxies to the internal fibers' color."""
        return self.fiber_a.color  # both fibers should have the same color

    @color.setter
    def color(self, value):
        self.fiber_a.color = value
        self.fiber_b.color = value

    def _update_geometry(self):
        """Internal helper to update output positions based on new geometry."""

        A = self._axis
        AT = self._axis_transverse
        in_pos = self.input_point.center
        height = self.height
        length = self.length
        alignment = self.alignment

        # re-compute the output positions based on the new geometry
        if alignment == "center":
            # if centered, both outputs are equidistant from input
            new_out_a_pos = in_pos + (height / 2) * AT + length * A
            new_out_b_pos = in_pos - (height / 2) * AT + length * A
        elif alignment == "top":
            # if top-aligned, A is aligned with input and B is 'down'
            new_out_a_pos = in_pos + length * A
            new_out_b_pos = in_pos - height * AT + length * A
        elif alignment == "bottom":
            # if bottom-aligned, B is aligned with input and A is 'up'
            new_out_a_pos = in_pos + height * AT + length * A
            new_out_b_pos = in_pos + length * A

        self.out_a_point.move_to(new_out_a_pos)
        self.out_b_point.move_to(new_out_b_pos)

    # Properties to update internal point positions when changed
    @property
    def length(self) -> float:
        """Axial distance from input to outputs."""
        return self._length

    @length.setter
    def length(self, value: float):
        if value < 0:
            raise ValueError("length must be non-negative")

        if value == self._length:
            return  # no change

        # store new length
        self._length = float(value)

        # update the positions
        self._update_geometry()

    @property
    def height(self) -> float:
        """Transverse separation between the two outputs."""
        return self._height

    @height.setter
    def height(self, value: float):

        if value < 0:
            raise ValueError("height must be non-negative")

        if value == self._height:
            return  # no change

        # store new height
        self._height = float(value)

        # update the positions
        self._update_geometry()

    @property
    def alignment(self) -> Literal["center", "top", "bottom"]:
        """Alignment of input relative to outputs.

        Notes
        -----

        The input point is the anchor of the `FiberSplitter`. Changing the alignment
        will move the output points accordingly, while keeping the input point fixed.
        """
        return self._alignment

    @alignment.setter
    def alignment(self, value: Literal["center", "top", "bottom"]):
        # check validity
        if value not in {"center", "top", "bottom"}:
            raise ValueError("alignment must be 'center', 'top' or 'bottom'")

        if value == self._alignment:
            return  # no change

        # store new alignment
        self._alignment = value

        # Update the positions if there was a change
        self._update_geometry()

    # convenience wrappers for chaining these changes
    def set_length(self, value: float) -> Self:
        """Set the axial length from input to outputs. Chainable.

        This is equivalent to setting the `length` property, but allows chaining.

        Parameters
        ----------
        value : float
            New length value.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self.length = value
        return self

    def set_height(self, value: float) -> Self:
        """Set the transverse height between the two outputs. Chainable.

        This is equivalent to setting the `height` property, but allows chaining.

        Parameters
        ----------
        value : float
            New height value.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self.height = value
        return self

    def set_alignment(self, value: Literal["center", "top", "bottom"]) -> Self:
        """Set the alignment of the input relative to the outputs. Chainable.

        This is equivalent to setting the `alignment` property, but allows chaining.

        Parameters
        ----------
        value : {"center", "top", "bottom"}
            New alignment value.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self.alignment = value
        return self

    # Show connections: delegate to internal fibers so drawing is handled by Group.draw()
    def show_connections(
        self, input: bool = True, out_a: bool = True, out_b: bool = True
    ) -> Self:
        """
        Toggle visibility of the endpoint markers. Chainable.

        Parameters
        ----------
        input : bool, optional
            Whether to show the input point marker. Default is True.
        out_a : bool, optional
            Whether to show the output A point marker. Default is True.
        out_b : bool, optional
            Whether to show the output B point marker. Default is True.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self._show_input = input
        self._show_out_a = out_a
        self._show_out_b = out_b

        self.fiber_a.show_connections(start=input, end=out_a)
        self.fiber_b.show_connections(start=input, end=out_b)

        return self

    def show_labels(self) -> Self:
        """Display labels to the corner points for debugging.

        These labels are only rendered when the element is drawn, and are not part of the
        element's geometry or layout. They are intended for debugging and visualization
        purposes.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self._show_labels = True

        return self

    # transformations that involve any kind of rotation should update the internal axis
    def rotate(self, angle_degrees, about_point=None):
        # Rotate internal axis
        rad = np.deg2rad(angle_degrees)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        self._axis = R @ self._axis

        self._axis_transverse = get_normal_vector(self._axis)

        return super().rotate(angle_degrees, about_point)

    def flip(self, axis=UP, about_point=None):
        # Flip internal axis (symmetry on the axis / reflection)
        axis_unit = np.asarray(axis) / np.linalg.norm(axis)

        # reflect self._axis (v) across the line defined by axis_unit (u):
        # v' = 2*(v·u)*u - v
        reflected = 2 * np.dot(self._axis, axis_unit) * axis_unit - self._axis

        # normalize and store
        self._axis = reflected / np.linalg.norm(reflected)

        # update transverse accordingly
        self._axis_transverse = get_normal_vector(self._axis)
        return super().flip(axis, about_point)

    # override the move_to method such that the anchor is the input and not center point
    def move_to(self, position: ArrayLike | OpticalElement):
        """Move the splitter so that its input point is placed at ``position``.

        This overrides the base `Group.move_to` behavior, which typically uses
        the element's center as the anchor. Here, the splitter is translated such
        that `input_point` is moved to ``position``, and all other points are
        shifted accordingly.

        Parameters
        ----------
        position : ArrayLike or OpticalElement
            Target location for the input point. May be an object with a
            ``center`` attribute or an array-like ``(x, y)`` coordinate.
        """
        # resolve position
        pos = position.center if hasattr(position, "center") else np.asarray(position)

        # compute shift vector from current input position to new position
        shift_vec = pos - self.input_point.center

        # apply shift to the whole splitter
        # (delegates to Fiber.shift, which updates the points)
        return self.shift(shift_vec)

    # Drawing: draw both fibers and optionally the markers
    def _get_mpl_artist(self):
        # composite element doesn't produce a single PathPatch; return None
        return None

    def draw(self, ax):
        # draw internal fiber paths
        ax.add_patch(self.fiber_a._get_mpl_artist())
        ax.add_patch(self.fiber_b._get_mpl_artist())

        # draw connection markers if requested
        if self._show_input:
            ax.add_patch(self.input_point._get_mpl_artist())
        if self._show_out_a:
            ax.add_patch(self.out_a_point._get_mpl_artist())
        if self._show_out_b:
            ax.add_patch(self.out_b_point._get_mpl_artist())
        if self._show_labels:
            from ._annotations import Label

            A = self._axis

            # add labels to the corner points
            point_labels = Group(
                [
                    Label(self.input_point, -A, "input", c=self.input_point.color),
                    Label(self.out_a_point, A, "out_a", c=self.out_a_point.color),
                    Label(self.out_b_point, A, "out_b", c=self.out_b_point.color),
                ]
            )
            point_labels.draw(ax)


class FiberCoupler(Group):
    """Coupler between 2 channels (2 inputs, 2 outputs)

    Implemented as a Group of 4 Fibers to be able to independently control the center
    separation (distance between the midpoints of the two channels) without affecting the
    overall geometry defined by the input/output positions.
    """

    def __init__(
        self,
        position: ArrayLike | OpticalElement,
        axis: ArrayLike = RIGHT,
        height: float = 1.0,
        length: float = 1.0,
        min_separation: float = 0.1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        position : ArrayLike or OpticalElement
            Center position of the coupler.
        axis : ArrayLike, optional
            Axial direction from inputs to outputs. Default is RIGHT.
        height : float, optional
            Transverse separation between the two channels. Default is 1.0.
        length : float, optional
            Axial length of each channel. Default is 1.0.
        min_separation : float, optional
            Minimum separation between the two channels at the center. Defaults to
            ``0.1``.
        """
        # style defaults
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("color", "C0")

        # resolve center
        center = (
            position.center
            if hasattr(position, "center")
            else np.asarray(position, dtype=float)
        )

        # axis / transverse
        axis = np.asarray(axis, dtype=float)
        if np.allclose(axis, 0):
            raise ValueError("axis must be non-zero")
        axis = axis / np.linalg.norm(axis)
        axis_transverse = get_normal_vector(axis)

        half_len = float(length) / 2.0
        half_h = float(height) / 2.0

        # corner points (rectangle)
        in_a = center - half_len * axis + half_h * axis_transverse
        in_b = center - half_len * axis - half_h * axis_transverse
        out_a = center + half_len * axis + half_h * axis_transverse
        out_b = center + half_len * axis - half_h * axis_transverse

        # midpoints along the centerline between input and output (shared per channel)
        mid_a = center + (min_separation / 2) * axis_transverse
        mid_b = center - (min_separation / 2) * axis_transverse

        # create fibers (segments). Use two segments per channel: one from the input
        # to the shared centerline midpoint and one from that midpoint to the output.
        # This keeps the input/output endpoints fixed while allowing the center
        # separation (distance between mid_a and mid_b) to be adjusted independently
        # of the overall coupler geometry.
        fa1 = Fiber(in_a, mid_a, **kwargs)
        fa2 = Fiber(mid_a, out_a, **kwargs)

        fb1 = Fiber(in_b, mid_b, **kwargs)
        fb2 = Fiber(mid_b, out_b, **kwargs)

        # store point references (corners & mids)
        self.in_a_point: Point = fa1.start_point
        """Input point of fiber channel A."""
        self.mid_a_point: Point = fa1.end_point
        """Midpoint of fiber channel A."""
        self.out_a_point: Point = fa2.end_point
        """Output point of fiber channel A."""

        self.in_b_point: Point = fb1.start_point
        """Input point of fiber channel B."""
        self.mid_b_point: Point = fb1.end_point
        """Midpoint of fiber channel B."""
        self.out_b_point: Point = fb2.end_point
        """Output point of fiber channel B."""

        # fibers list for the Group
        elements = [fa1, fa2, fb1, fb2]
        self._show_labels = False  # controls visibility of internal labels
        self.fiber_a1, self.fiber_a2, self.fiber_b1, self.fiber_b2 = elements

        # initialize Group with the four fiber segments
        super().__init__(elements, **kwargs)

        if "color" in kwargs:
            self.color = kwargs.pop("color")

        # store geometry state
        self._axis = axis
        self._axis_transverse = axis_transverse
        self._length = float(length)
        self._height = float(height)
        self._min_separation = float(min_separation)

        # show flags for corner markers (optional)
        self._show_in_a = False
        self._show_in_b = False
        self._show_out_a = False
        self._show_out_b = False
        self._show_labels = False  # for debugging

    @property
    def color(self):
        """Shorthand color. Proxies to the internal fibers' color."""
        return self.fiber_a1.color  # both fibers should have the same color

    @color.setter
    def color(self, value):
        self.fiber_a1.color = value
        self.fiber_a2.color = value
        self.fiber_b1.color = value
        self.fiber_b2.color = value

    def _update_geometry(self):
        """Internal helper to update corner and mid point positions based on new geometry."""

        A = self._axis
        AT = self._axis_transverse
        center = self.center
        height = self.height
        length = self.length
        min_separation = self.min_separation

        half_len = length / 2.0
        half_h = height / 2.0
        half_sep = min_separation / 2.0

        # compute new corner points
        new_in_a = center - half_len * A + half_h * AT
        new_in_b = center - half_len * A - half_h * AT
        new_out_a = center + half_len * A + half_h * AT
        new_out_b = center + half_len * A - half_h * AT

        # compute new mid points
        new_mid_a = center + half_sep * AT
        new_mid_b = center - half_sep * AT

        # move existing corner points to new positions (inputs/outputs)
        self.in_a_point.move_to(new_in_a)
        self.in_b_point.move_to(new_in_b)
        self.out_a_point.move_to(new_out_a)
        self.out_b_point.move_to(new_out_b)

        # move existing mid points to new positions
        self.mid_a_point.move_to(new_mid_a)
        self.mid_b_point.move_to(new_mid_b)

        # Manually move the start points of fiber_a2 and fiber_b2 to match the mid points.
        # These start_point attributes are separate Point instances created when the Fiber
        # objects were initialized; they do not share references with mid_a_point or
        # mid_b_point.
        # Because of this, they are not updated by the midpoint transforms and must be
        # synchronized here. We cannot simply reuse the same Point objects for both roles,
        # or the transforms applied by each Fiber would accumulate incorrectly.
        self.fiber_a2.start_point.move_to(new_mid_a)
        self.fiber_b2.start_point.move_to(new_mid_b)

    # --- Properties that update point positions --------------------------------
    @property
    def length(self) -> float:
        """Axial length of each channel (from input to output)."""
        return self._length

    @length.setter
    def length(self, value):
        if value < 0:
            raise ValueError("length must be non-negative")

        if value == self._length:
            return  # no change

        # store new length
        self._length = float(value)

        # update the positions
        self._update_geometry()

    @property
    def height(self) -> float:
        """Height (transverse separation) between the two channels."""
        return self._height

    @height.setter
    def height(self, value: float):
        if value < 0:
            raise ValueError("height must be non-negative")

        if value == self._height:
            return  # no change

        # store new height
        self._height = float(value)

        # update the positions
        self._update_geometry()

    @property
    def min_separation(self) -> float:
        """Minimum separation between the two channels at the center."""
        return self._min_separation

    @min_separation.setter
    def min_separation(self, value: float):
        if value < 0:
            raise ValueError("min_separation must be non-negative")

        if value == self._min_separation:
            return  # no change

        # store new min_separation
        self._min_separation = float(value)

        # update the positions
        self._update_geometry()

    # chaining helpers
    def set_length(self, value: float) -> Self:
        """Set the axial length of each channel. Chainable."""
        self.length = value
        return self

    def set_height(self, value: float) -> Self:
        """Set the height (transverse separation) of each channel. Chainable."""
        self.height = value
        return self

    def set_min_separation(self, value: float) -> Self:
        """Set the minimum separation between the two channels at the center. Chainable."""
        self.min_separation = value
        return self

    # --- show connections (delegates to internal fibers) -----------------------
    def show_connections(
        self, in_a: bool = True, in_b: bool = True, out_a: bool = True, out_b: bool = True
    ) -> Self:
        """Show or hide connection markers at the input/output points of each channel.

        Parameters
        ----------
        in_a : bool, optional
            Whether to show the input marker for channel A. Default is True.
        in_b : bool, optional
            Whether to show the input marker for channel B. Default is True.
        out_a : bool, optional
            Whether to show the output marker for channel A. Default is True.
        out_b : bool, optional
            Whether to show the output marker for channel B. Default is True.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self._show_in_a = in_a
        self._show_in_b = in_b
        self._show_out_a = out_a
        self._show_out_b = out_b

        # map to fiber segment visibility:
        # - in markers correspond to the start of the first segment for each channel
        # - out markers correspond to the end of the second segment for each channel
        self.fiber_a1.show_connections(start=in_a, end=False)
        self.fiber_a2.show_connections(start=False, end=out_a)
        self.fiber_b1.show_connections(start=in_b, end=False)
        self.fiber_b2.show_connections(start=False, end=out_b)

        return self

    # --- keep internal axis up-to-date on transforms ---------------------------
    def rotate(self, angle_degrees, about_point=None):
        rad = np.deg2rad(angle_degrees)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        self._axis = R @ self._axis
        self._axis = self._axis / np.linalg.norm(self._axis)
        self._axis_transverse = get_normal_vector(self._axis)

        return super().rotate(angle_degrees, about_point)

    def flip(self, axis=UP, about_point=None):
        axis_unit = np.asarray(axis) / np.linalg.norm(axis)
        reflected = 2 * np.dot(self._axis, axis_unit) * axis_unit - self._axis
        self._axis = reflected / np.linalg.norm(reflected)
        self._axis_transverse = get_normal_vector(self._axis)
        return super().flip(axis, about_point)

    def show_labels(self):
        """Display labels to the corner points for debugging.

        These labels are only rendered when the element is drawn, and are not part of the
        element's geometry or layout. They are intended for debugging and visualization
        purposes.

        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        self._show_labels = True

        return self

    # --- Drawing: draw fibers (Group.draw covers fibers) and optional corner dots
    def _get_mpl_artist(self):
        return None

    def draw(self, ax):
        # draw fibers via Group behavior
        super().draw(ax)
        # draw corner markers if requested
        if self._show_in_a:
            ax.add_patch(self.in_a_point._get_mpl_artist())
        if self._show_in_b:
            ax.add_patch(self.in_b_point._get_mpl_artist())
        if self._show_out_a:
            ax.add_patch(self.out_a_point._get_mpl_artist())
        if self._show_out_b:
            ax.add_patch(self.out_b_point._get_mpl_artist())
        if self._show_labels:
            from ._annotations import Label

            A = self._axis

            # add labels to the corner points
            point_labels = Group(
                [
                    Label(self.in_a_point, -A, "in_a", c=self.in_a_point.color),
                    Label(self.in_b_point, -A, "in_b", c=self.in_b_point.color),
                    Label(self.out_a_point, A, "out_a", c=self.out_a_point.color),
                    Label(self.out_b_point, A, "out_b", c=self.out_b_point.color),
                ]
            )
            point_labels.draw(ax)
