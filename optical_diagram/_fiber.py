import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from ._annotations import Point
from ._base import UP, OpticalElement

__all__ = ["Fiber"]


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
