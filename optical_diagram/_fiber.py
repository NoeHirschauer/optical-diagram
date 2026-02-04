from typing import Literal

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from ._annotations import Point
from ._base import RIGHT, UP, Group, OpticalElement, get_normal_vector

__all__ = ["Fiber", "FiberSplitter"]


class Fiber(OpticalElement):
    """
    Connects two points or elements with a smooth cubic spline.
    Endpoints are full Point objects, supporting raytracing and alignment.
    """

    def __init__(
        self, start, end, angle_start=0.0, angle_end=0.0, stiffness=0.5, **kwargs
    ):
        # ensure fiber is not filled by default (closed PathPatch becomes filled otherwise)
        kwargs.setdefault("facecolor", "none")

        # 1. Resolve initial coordinates
        p0_pos = start.center if hasattr(start, "center") else np.asarray(start)
        p3_pos = end.center if hasattr(end, "center") else np.asarray(end)

        # 2. Set default styles
        kwargs.setdefault("edgecolor", "C0")
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("zorder", 4)
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")

        # 3. Create endpoints as Point objects
        # We pass the fiber's color to the dots so they match by default
        dot_color = kwargs.get("edgecolor")
        self.start_point = Point(p0_pos, color=dot_color)
        self.end_point = Point(p3_pos, color=dot_color)

        self.angle_start = float(angle_start)
        self.angle_end = float(angle_end)
        self.stiffness = stiffness
        self._show_start = False
        self._show_end = False

        # 4. Initialize base at the midpoint
        midpoint = (self.start_point.center + self.end_point.center) / 2.0
        super().__init__(midpoint, size=1.0, angle=0.0, **kwargs)

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

    def get_local_points(self):
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

    def show_connections(self, start=True, end=True):
        """
        Toggle visibility of the endpoint markers.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self._show_start = start
        self._show_end = end
        return self

    # Override facecolor so fibers cannot be filled (avoids closed, filled bezier patch)
    @property
    def facecolor(self):
        return "none"

    @facecolor.setter
    def facecolor(self, value):
        # ignore attempts to set a fill on Fiber; ensure it stays 'none'
        self._style["facecolor"] = "none"

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
    """
    Composite element implemented as a Group of two Fibers that share a common input Point.

    Notes
    -----

    Only the 3 endpoint Points are part of this Group; the internal Fibers are not.
    Transformations on the FiberSplitter are delegated to the Points, which in turn
    update the internal Fibers automatically.
    """

    def __init__(
        self,
        input_pos,
        axis=RIGHT,
        length=1.0,
        height=1.0,
        alignment: Literal["center", "top", "bottom"] = "center",
        **kwargs,
    ):
        """
        Parameters
        ----------
        input_pos : array-like or OpticalElement
            Input coordinates or an element with .center used as input position.
        axis : array-like, optional
            Axial direction from outputs toward the input. Defaults to RIGHT
        length : float, optional
            Axial distance from outputs to input. Default is 1.0.
        height : float, optional
            Transverse separation between the two outputs. Default is 1.0.
        alignment : {"center", "top", "bottom"}, optional
            determines which transverse location the input is aligned to. Default is
            ``'center'``.
        **kwargs : dict, optional
            styling forwarded to the internal Fiber objects
        """
        # styling defaults
        kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("edgecolor", "C0")

        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")

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

        self.input_point = fa.start_point  # shared input Point
        self.out_a_point = fa.end_point  # output A Point
        self.out_b_point = fb.end_point  # output B Point

        self.fiber_a = fa
        self.fiber_b = fb

        # Initialize as a Group containing only the Fibers for correct transforms
        elements = [self.fiber_a, self.fiber_b]
        super().__init__(elements, **kwargs)

        # We need to update these points if any changes to length/height/alignment occur
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

    # Properties to update internal point positions when changed
    @property
    def length(self):
        """float : axial distance from input to outputs."""
        return self._length

    @length.setter
    def length(self, value):
        # Update output point positions based on new length (we only need to shift
        # along the axis direction)
        self.out_a_point.shift((value - self._length) * self._axis)
        self.out_b_point.shift((value - self._length) * self._axis)

        # store new length
        self._length = float(value)

    @property
    def height(self):
        """float : transverse separation between the two outputs."""
        return self._height

    @height.setter
    def height(self, value):
        # Update output point positions based on new height and current alignment
        if self._alignment == "center":
            # if centered, shift both outputs equally in opposite directions
            self.out_a_point.shift(((value - self._height) / 2) * self._axis_transverse)
            self.out_b_point.shift(((value - self._height) / 2) * -self._axis_transverse)
        elif self._alignment == "top":
            # if top-aligned, only shift output B "downwards"
            self.out_b_point.shift((value - self._height) * -self._axis_transverse)
        elif self._alignment == "bottom":
            # if bottom-aligned, only shift output A "upwards"
            self.out_a_point.shift((value - self._height) * self._axis_transverse)

        # store new height
        self._height = float(value)

    @property
    def alignment(self):
        """{'center', 'top', 'bottom'} : Alignment of input relative to outputs.

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
        # Update the positions of the output points based on the new alignment.
        # For this, we need to compute the change in alignment and shift the output
        # points accordingly.

        new_alignment = value
        old_alignment = self._alignment

        if new_alignment == old_alignment:
            return  # no change

        half_shift_transverse = (self._height / 2) * self._axis_transverse

        if new_alignment == "center":
            if old_alignment == "top":
                # shift both outputs equally 1/2 height "upwards"
                self.out_a_point.shift(half_shift_transverse)
                self.out_b_point.shift(half_shift_transverse)
            elif old_alignment == "bottom":
                # shift both outputs equally 1/2 height "downwards"
                self.out_a_point.shift(-half_shift_transverse)
                self.out_b_point.shift(-half_shift_transverse)

        elif new_alignment == "top":
            if old_alignment == "center":
                # shift both outputs equally 1/2 height "downwards"
                self.out_a_point.shift(-half_shift_transverse)
                self.out_b_point.shift(-half_shift_transverse)
            elif old_alignment == "bottom":
                # shift both outputs "downwards" by full height
                self.out_a_point.shift(-2 * half_shift_transverse)
                self.out_b_point.shift(-2 * half_shift_transverse)

        elif new_alignment == "bottom":
            if old_alignment == "center":
                # shift both outputs equally 1/2 height "upwards"
                self.out_a_point.shift(half_shift_transverse)
                self.out_b_point.shift(half_shift_transverse)
            elif old_alignment == "top":
                # shift both outputs "upwards" by full height
                self.out_a_point.shift(2 * half_shift_transverse)
                self.out_b_point.shift(2 * half_shift_transverse)

        # store new alignment
        self._alignment = new_alignment

    # convenience wrappers for chaining these changes
    def set_length(self, value):
        """Set the axial length from input to outputs.

        This is equivalent to setting the `length` property, but allows chaining.

        Parameters
        ----------
        value : float
            New length value.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.length = value
        return self

    def set_height(self, value):
        """Set the transverse height between the two outputs.

        This is equivalent to setting the `height` property, but allows chaining.

        Parameters
        ----------
        value : float
            New height value.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.height = value
        return self

    def set_alignment(self, value: Literal["center", "top", "bottom"]):
        """Set the alignment of the input relative to the outputs.

        This is equivalent to setting the `alignment` property, but allows chaining.

        Parameters
        ----------
        value : {"center", "top", "bottom"}
            New alignment value.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.alignment = value
        return self

    # Show connections: delegate to internal fibers so drawing is handled by Group.draw()
    def show_connections(self, input=True, out_a=True, out_b=True):
        """
        Toggle visibility of the endpoint markers.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self._show_input = input
        self._show_out_a = out_a
        self._show_out_b = out_b

        self.fiber_a.show_connections(start=input, end=out_a)
        self.fiber_b.show_connections(start=input, end=out_b)

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
