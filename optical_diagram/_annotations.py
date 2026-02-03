import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import Polygon as MplPolygon

from ._base import ORIGIN, OpticalElement

__all__ = [
    "Plane",
    "Point",
    "OpticalAxis",
    "Label",
    "Rectangle",
    "SurroundingRectangle",
]


class Plane(OpticalElement):
    """
    A plane represented by a dashed line.
    Used for visualization; rays pass through without interaction.
    """

    def __init__(self, position=ORIGIN, size=1.0, angle=0.0, **kwargs):
        # Default Plane is vertical at 0 deg to match Lens/Mirror convention
        kwargs.setdefault("linestyle", "--")
        kwargs.setdefault("color", "gray")
        super().__init__(position, size, angle, **kwargs)

    def get_local_points(self):
        # Defines a vertical line segment in local space
        h = self._size / 2.0
        return np.array([[0, -h], [0, h]])

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class Point(OpticalElement):
    """
    A reference point represented by a small dot.
    Rays pass through its plane without interaction.
    """

    def __init__(self, position=ORIGIN, size=0.1, **kwargs):
        # Style for the visual dot
        kwargs.setdefault("color", "black")
        kwargs.setdefault("zorder", 15)
        # size=0.1 defines the visual diameter of the dot.
        super().__init__(position, size=size, angle=0.0, **kwargs)

    def get_local_points(self):
        # A point has no extent, but provides the center for the artist
        return np.array([[0, 0]])

    def _get_mpl_artist(self):
        style = self._style.copy()
        # Ensure the dot is filled
        if style.get("facecolor") == "none":
            style["facecolor"] = style.get("edgecolor", "k")

        return Circle(self.center, radius=self._size / 2, **style)


class OpticalAxis(OpticalElement):
    """
    An optical axis connecting two points.

    Note: OpticalAxis behaves slightly differently as it is defined by
    two points rather than center+size.
    """

    def __init__(self, start_point, end_point, **kwargs):
        if isinstance(start_point, OpticalElement):
            self._start = start_point.center
        else:
            self._start = np.asarray(start_point)
        if isinstance(end_point, OpticalElement):
            self._end = end_point.center
        else:
            self._end = np.asarray(end_point)
        center = (self._start + self._end) / 2
        dist = np.linalg.norm(self._end - self._start)

        # Calculate angle
        delta = self._end - self._start
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        kwargs.setdefault("color", "gray")
        kwargs.setdefault("linestyle", "--")
        kwargs.setdefault("zorder", -1)

        super().__init__(center, dist, angle, **kwargs)

    def get_local_points(self):
        h = self._size / 2.0
        return np.array([[-h, 0], [h, 0]])

    def _get_mpl_artist(self):
        # We re-calculate based on current center/angle in case it was moved
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class Label(OpticalElement):
    """
    A text label positioned relative to an anchor point or element.
    """
    def __init__(self, anchor, direction=None, text="label", buffer=0.1, **kwargs):
        self.anchor = anchor
        self.direction = (
            np.asarray(direction) if direction is not None else np.array((0, 0))
        )
        self.text = text
        self.buffer = buffer
        kwargs.setdefault(
            "horizontalalignment",
            self._get_alignment(self.direction[0], "left", "right", "center"),
        )
        kwargs.setdefault(
            "verticalalignment",
            self._get_alignment(self.direction[1], "bottom", "top", "center"),
        )
        base_pos = (
            anchor.get_edge(self.direction)
            if hasattr(anchor, "get_edge")
            else np.asarray(anchor)
        )
        final_pos = base_pos + (self.direction * self.buffer)
        super().__init__(final_pos, size=0, angle=0.0, **kwargs)

    def _get_alignment(self, val, pos_res, neg_res, zero_res):
        if val > 0.05:
            return pos_res
        if val < -0.05:
            return neg_res
        return zero_res

    def _get_mpl_artist(self):
        return None

    def draw(self, ax):
        style = self._style.copy()
        color = style.pop("color", style.pop("edgecolor", "black"))
        style.pop("facecolor", None)
        style.pop("linewidth", None)
        style.pop("rotation", None)
        return ax.text(
            self.center[0],
            self.center[1],
            self.text,
            color=color,
            rotation=self._angle,
            **style,
        )


class Rectangle(OpticalElement):
    """
    A rectangle represented by a polygon, defined by width and height.
    """
    def __init__(self, position=ORIGIN, width=1.0, height=1.0, angle=0.0, **kwargs):
        self.width = float(width)
        self.height = float(height)
        super().__init__(position, size=max(width, height), angle=angle, **kwargs)
        defaults = {"edgecolor": "k", "facecolor": "none", "linewidth": 1.5, "zorder": 10}
        for k, v in defaults.items():
            self._style.setdefault(k, v)

    def get_local_points(self):
        w2, h2 = self.width / 2.0, self.height / 2.0
        return np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        return MplPolygon(pts, closed=True, **self._style)

    def scale(self, scale_factor, about_point=None):
        self.width *= scale_factor
        self.height *= scale_factor
        return super().scale(scale_factor, about_point)

    # Rectangle uses facecolor as its shorthand "color" (filled boxes)
    @property
    def color(self):
        return self._style.get("facecolor", None)

    @color.setter
    def color(self, value):
        if value is None:
            self._style.pop("facecolor", None)
            self._style.pop("edgecolor", None)
        else:
            self._style["facecolor"] = value
            self._style.pop("color", None)


class SurroundingRectangle(Rectangle):
    """
    A rectangle that surrounds a set of optical elements, with optional buffer.
    """
    def __init__(self, elements, buff=0.2, **kwargs):
        if not isinstance(elements, (list, tuple, np.ndarray)):
            elements = [elements]
        all_min_x, all_min_y, all_max_x, all_max_y = [], [], [], []
        for el in elements:
            x_min, y_min, x_max, y_max = el.get_boundary_box()
            all_min_x.append(x_min)
            all_min_y.append(y_min)
            all_max_x.append(x_max)
            all_max_y.append(y_max)
        g_min_x, g_max_x = min(all_min_x), max(all_max_x)
        g_min_y, g_max_y = min(all_min_y), max(all_max_y)
        width = (g_max_x - g_min_x) + 2 * buff
        height = (g_max_y - g_min_y) + 2 * buff
        center = np.array([(g_min_x + g_max_x) / 2, (g_min_y + g_max_y) / 2])
        kwargs.setdefault("edgecolor", "C3")
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("zorder", 5)
        super().__init__(position=center, width=width, height=height, angle=0.0, **kwargs)
