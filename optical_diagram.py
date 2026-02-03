"""
This is `optical_diagram`, a small module building on Matplotlib to create
optical diagrams with lenses, mirrors, fibers, and (minimal) ray tracing.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path

__all__ = [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "UR",
    "UL",
    "DR",
    "DL",
    "ORIGIN",
    "PlaneMirror",
    "ConcaveMirror",
    "ConvexMirror",
    "ConvergingLens",
    "DivergingLens",
    "BeamSplitter",
    "Plane",
    "Point",
    "OpticalAxis",
    "Label",
    "Rectangle",
    "SurroundingRectangle",
    "Fiber",
    "SimpleBeam",
    "DivergingBeam",
    "RayTracedBeam",
    "OpticalTable",
    "get_axis_direction",
    "get_normal_direction",
    "get_normal_vector",
]

__version__ = "0.1.0"

# -----------------------------------------------------------------------------
# Constants & Vectors
# -----------------------------------------------------------------------------
# %%
RIGHT  = np.array((1.0, 0.0))
LEFT   = np.array((-1.0, 0.0))
UP     = np.array((0.0, 1.0))
DOWN   = np.array((0.0, -1.0))
ORIGIN = np.array((0.0, 0.0))

UR = UP + RIGHT
UL = UP + LEFT
DR = DOWN + RIGHT
DL = DOWN + LEFT


# -----------------------------------------------------------------------------
# Base Classes
# -----------------------------------------------------------------------------
# %%


class OpticalElement(ABC):
    """
    Abstract base class for all optical elements.

    Implements a state-based approach (Manim-style). The geometric properties
    (center, angle, size) are stored as attributes and the Matplotlib patch
    is generated only when requested.

    Parameters
    ----------
    position : array_like, shape (2,)
        The (x, y) coordinates of the center. Defaults to ``(0,0)``
    size : float
        Characteristic size (diameter for lens, side for square). Defaults to ``1.0``
    angle : float
        Rotation angle in degrees. Defaults to ``0.0``
    **kwargs
        Style arguments passed to the matplotlib patch (color, alpha, etc).
    """

    def __init__(self, position=ORIGIN, size=1, angle=0.0, **kwargs):
        if isinstance(position, OpticalElement):
            _position = position.center
        else:
            _position = position

        self._center = np.asarray(_position, dtype=float).ravel()
        if self._center.size != 2:
            raise ValueError("Position must be a length-2 sequence.")

        self._size = float(size)
        self._angle = float(angle)

        # Default styles
        self._style = {
            # "edgecolor": "k",
            # "facecolor": "none",
            "linewidth": 1.5,
            "zorder": 10,
        }

        # Apply incoming kwargs into style first (but handle 'color' specially)
        # Extract shorthand color so we can map it to the correct category for this instance
        _col = kwargs.pop("color", None)

        # Apply any explicit kwargs (edgecolor/facecolor/etc) first
        self._style.update(kwargs)

        # Map shorthand color to the correct slot:
        # - For RayTracedBeam, SurroundingRectangle, Rectangle -> map to facecolor
        # - For everything else -> map to edgecolor
        # - Never override Fiber's explicit facecolor="none"
        if _col is not None:
            # runtime-safe type checks (names exist by the time __init__ runs)
            if isinstance(self, (RayTracedBeam, SurroundingRectangle, Rectangle)):
                # only set facecolor if not explicitly protected (Fiber's default 'none')
                if not (
                    isinstance(self, Fiber) and self._style.get("facecolor") == "none"
                ):
                    self._style["facecolor"] = _col
            else:
                self._style["edgecolor"] = _col

        # Ensure no leftover 'color' key remains in the style dict
        self._style.pop("color", None)

    # -------------------------------------------------------------------------
    # Geometric Properties
    # -------------------------------------------------------------------------
    @property
    def center(self):
        """np.ndarray: The current (x, y) center of the element."""
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.asarray(value, dtype=float)

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = float(value)

    @property
    def x(self):
        return self._center[0]

    @property
    def y(self):
        return self._center[1]

    

    @property
    def angle(self):
        return self._angle

    # -------------------------------------------------------------------------
    # Creation & Duplication
    # -------------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of this element.

        This is useful for creating multiple identical or similar elements
        without redefining all properties.

        Returns
        -------
        OpticalElement
            A new instance independent of the original.
        """
        return deepcopy(self)

    # -------------------------------------------------------------------------
    # Geometric Utilities (Anchors)
    # -------------------------------------------------------------------------
    def get_rotation_matrix(self):
        theta = np.deg2rad(self._angle)
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def get_local_points(self):
        """
        Return key points (like corners or ends) in unrotated, local coordinates (relative to 0,0).
        Must be implemented by subclasses to support get_boundary.
        """
        # Default: just top and bottom based on size (good for lines/lenses)
        half = self._size / 2.0
        return np.array([[0, half], [0, -half]])

    def get_critical_points(self):
        """Return the key geometric points transformed to world space."""
        local_pts = self.get_local_points()
        rot = self.get_rotation_matrix()
        return (rot @ local_pts.T).T + self.center

    def get_boundary_box(self):
        """Returns (min_x, min_y, max_x, max_y) of the element."""
        pts = self.get_critical_points()
        return np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])

    def get_edge(self, direction):
        """
        Get the point on the bounding box in a specific direction.

        Parameters
        ----------
        direction : array_like
            Vector indicating direction (e.g. RIGHT, UP).

        Returns
        -------
        np.array
            The (x, y) point on the edge.
        """
        min_x, min_y, max_x, max_y = self.get_boundary_box()
        center_box = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])

        # Simple box mapping
        direction = np.asarray(direction)
        # Normalize roughly to find corner or edge
        half_w = (max_x - min_x) / 2
        half_h = (max_y - min_y) / 2

        return center_box + direction * [half_w, half_h]

    # -------------------------------------------------------------------------
    # Physical Utilities (for ray tracing)
    # -------------------------------------------------------------------------
    def get_normal(self):
        """Returns the normal vector of the element plane."""
        theta = np.deg2rad(self._angle)
        # Assuming element lies along local Y, normal is along local X
        # Rotated by theta:
        return np.array([np.cos(theta), np.sin(theta)])

    def get_direction_vector(self):
        """Returns the vector pointing along the element's surface."""
        theta = np.deg2rad(self._angle)
        return np.array([-np.sin(theta), np.cos(theta)])

    def intersect(self, ray_origin, ray_direction):
        """
        Calculate intersection point of a ray with this element's plane.
        Returns None if parallel.
        """
        # Plane equation: (P - Center) . Normal = 0
        # Ray equation: P = Origin + t * Direction
        # Solve for t: t = (Center - Origin) . Normal / (Direction . Normal)

        normal = self.get_normal()
        denom = np.dot(ray_direction, normal)

        if np.isclose(denom, 0):
            return None  # Parallel

        t = np.dot(self.center - ray_origin, normal) / denom

        # Check if t is positive (element is in front)
        if t < 1e-9:
            return None

        intersection = ray_origin + t * ray_direction

        # Check if intersection is within the element's size (aperture check)
        dist_from_center = np.linalg.norm(intersection - self.center)
        if dist_from_center > self._size / 2:
            return None  # Missed the lens aperture

        return intersection

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        """
        Physics logic: How does this element alter an incoming ray?

        Parameters
        ----------
        ray_origin : array
            Point where ray hits the element.
        ray_direction : array
            Incoming direction vector.
        next_element_center : array, optional
            The position of the next element in the list (used for BeamSplitters
            to decide whether to transmit or reflect).

        Returns
        -------
        new_direction : array
            The normalized direction vector after interaction.
        """
        # Default behavior: Pass through unchanged (Glass slab / Window)
        return ray_direction

    # -------------------------------------------------------------------------
    # Transformations (Chainable)
    # -------------------------------------------------------------------------
    def move_to(self, target):
        """
        Move the center of the element to the specified point or another element's center.

        Parameters
        ----------
        target : array_like or `OpticalElement`
            The point (x, y) to move to, or an `OpticalElement` whose center coordinates
            will be used as target

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        if isinstance(target, OpticalElement):
            target = target.center
        else:
            target = np.asarray(target)
        self.center = target
        return self

    def shift(self, vector):
        """
        Translate the element by a vector.

        Parameters
        ----------
        vector : array_like
            (dx, dy) translation vector.

        Returns
        -------
        self
        """
        self.center = self.center + np.asarray(vector)
        return self

    def rotate(self, angle_degrees, about_point=None):
        """
        Rotate the element.

        Parameters
        ----------
        angle_degrees : float
            Angle to add to current rotation.
        about_point : array_like or `OpticalElement`, optional
            Point to rotate about. If an `OpticalElement` is provided, its center is used
            as the pivot point If None, rotates about self.center.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self._angle += angle_degrees

        # If rotating about a specific point, we must also update the center position
        if about_point is not None:
            if isinstance(about_point, OpticalElement):
                about_point = about_point.center
            else:
                about_point = np.asarray(about_point, float)

            pivot = np.asarray(about_point)
            rad = np.deg2rad(angle_degrees)
            rot_matrix = np.array(
                [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]
            )
            # v' = R(v - p) + p
            self.center = rot_matrix @ (self.center - pivot) + pivot

        return self

    def flip(self, axis=UP, about_point=None):
        """
        Mirror the element across an axis passing through a point.

        Parameters
        ----------
        axis : array_like
            Vector defining the line of reflection (e.g., UP for vertical flip,
            RIGHT for horizontal flip). Defaults to UP.
        about_point : array_like or `OpticalElement`, optional
            Point the reflection axis passes through. If an `OpticalElement`
            is provided, its center is used. If None, flips about self.center.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        # 1. Resolve the pivot (about_point)
        if about_point is None:
            pivot = self.center
        elif isinstance(about_point, OpticalElement):
            pivot = about_point.center
        else:
            pivot = np.asarray(about_point, float)

        # 2. Normalize the axis vector
        axis_vec = np.asarray(axis, float)
        axis_unit = axis_vec / np.linalg.norm(axis_vec)

        # 3. Update Position
        # Reflection formula: v' = 2 * proj_axis(v) - v
        # where v is the vector from pivot to center
        v = self.center - pivot
        v_reflected = 2 * np.dot(v, axis_unit) * axis_unit - v
        self.center = pivot + v_reflected

        # 4. Update Angle
        # Reflecting an orientation theta across a line at angle phi
        # results in a new angle: 2*phi - theta
        axis_angle = np.degrees(np.arctan2(axis_unit[1], axis_unit[0]))
        self._angle = 2 * axis_angle - self._angle

        return self

    def scale(self, scale_factor, about_point=None):
        """
        Scale the element's size and potentially its position.

        Parameters
        ----------
        scale_factor : float
            Factor to multiply the current size by.
        about_point : array_like or `OpticalElement`, optional
            The reference point for the scaling operation.
            If an `OpticalElement` is provided, its center is used.
            If None, the element scales in place (about its own center).

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        # 1. Update the internal size
        self._size *= scale_factor

        # 2. If scaling about a different point, update the center position
        if about_point is not None:
            if isinstance(about_point, OpticalElement):
                pivot = about_point.center
            else:
                pivot = np.asarray(about_point, float)

            # The vector from the pivot to the center scales by the factor
            # v' = s * (v - p) + p
            relative_vector = self.center - pivot
            self.center = (scale_factor * relative_vector) + pivot

        return self

    def next_to(self, other, direction=RIGHT, buff=0.0):
        """
        Move this element next to another element.

        Parameters
        ----------
        other : `OpticalElement`
            The reference element.
        direction : array_like
            Direction to place this element (e.g. RIGHT).
        buff : float
            Buffer distance. Defaults to 0
        """
        target_point = other.get_edge(direction)
        my_anchor = self.get_edge(-np.array(direction))

        shift_vec = target_point - my_anchor + np.array(direction) * buff
        self.shift(shift_vec)
        return self

    # -------------------------------------------------------------------------
    # Style Properties & Methods
    # -------------------------------------------------------------------------

    # color property that proxies to edgecolor or facecolor depending on class
    @property
    def color(self):
        """Shorthand color. Proxies to edgecolor except for RayTracedBeam/SurroundingRectangle/Rectangle where it proxies to facecolor."""
        if isinstance(self, (RayTracedBeam, SurroundingRectangle, Rectangle)):
            return self._style.get("facecolor", None)
        return self._style.get("edgecolor", None)

    @color.setter
    def color(self, value):
        if value is None:
            # remove both to be safe
            self._style.pop("edgecolor", None)
            self._style.pop("facecolor", None)
        else:
            if isinstance(self, (RayTracedBeam, SurroundingRectangle, Rectangle)):
                # respect Fiber protection (do not overwrite Fiber facecolor='none')
                if not (
                    isinstance(self, Fiber) and self._style.get("facecolor") == "none"
                ):
                    self._style["facecolor"] = value
                # ensure no stray 'color' entry
                self._style.pop("color", None)
            else:
                self._style["edgecolor"] = value
                self._style.pop("color", None)

    # small compatibility wrappers so existing call-sites keep working
    def set_color(self, color):
        self.color = color
        return self

    def set_facecolor(self, color):
        self.facecolor = color
        return self

    def set_edgecolor(self, color):
        self.edgecolor = color
        return self

    def set_style(self, **kwargs):
        """Update matplotlib style parameters (linewidth, alpha, etc)."""
        self._style.update(kwargs)
        return self

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    @abstractmethod
    def _get_mpl_artist(self):
        """Generate and return the matplotlib artist for the current state."""
        pass

    def draw(self, ax):
        """Add the element to the provided Axes."""
        artist = self._get_mpl_artist()
        ax.add_patch(artist)
        return artist


class OpticalSystem(OpticalElement):
    """
    Abstract base class for optical elements that can have a focal length
    (Mirrors & Lenses).
    """

    def __init__(self, position=ORIGIN, size=1, angle=0.0, focal_length=None, **kwargs):
        self._focal_length = focal_length
        super().__init__(position, size, angle, **kwargs)

        # Default styles (use setdefault so we don't clobber mapping applied in base class)
        defaults = {
            "edgecolor": "k",
            "facecolor": "none",
            "linewidth": 1.5,
            "zorder": 10,
        }
        for k, v in defaults.items():
            self._style.setdefault(k, v)

        # Respect the shorthand 'color' here as well (same semantics as in base class)
        if "color" in kwargs:
            col = kwargs["color"]
            self._style["edgecolor"] = col
            if "facecolor" not in kwargs:
                self._style["facecolor"] = col

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = value


class Lens(OpticalSystem):
    """Base class for lenses."""

    def get_local_points(self):
        h = self._size / 2.0
        return np.array([[0, -h], [0, h]])

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        return FancyArrowPatch(pts[0], pts[1], **self._style)

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        if self.focal_length is None:
            raise ValueError(f"Lens at {self.center} has no focal_length set.")

        # Vectors
        normal = self.get_normal()
        # Ensure normal points in the direction of propagation
        if np.dot(normal, ray_direction) < 0:
            normal = -normal
        lens_vec = self.get_direction_vector()

        # Ray components
        v_n = np.dot(ray_direction, normal)
        v_t = np.dot(ray_direction, lens_vec)

        # Current slope (tan theta)
        slope_in = v_t / v_n

        # Height from center
        r_vec = ray_origin - self.center
        h = np.dot(r_vec, lens_vec)

        # Apply Thin Lens Equation in slope space:
        # 1/si = 1/f - 1/so  =>  slope_out = slope_in - h/f
        slope_out = slope_in - (h / self.focal_length)

        # Construct new direction
        new_dir = normal + slope_out * lens_vec
        return new_dir / np.linalg.norm(new_dir)


class Mirror(OpticalSystem):
    """Base class for all mirrors (Plane, Concave, Convex)."""

    def get_local_points(self):
        # Mirrors are vertical lines by default (0 deg)
        h = self._size / 2.0
        return np.array([[0, -h], [0, h]])

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        normal = self.get_normal()
        # Normal faces the incoming ray for reflection
        if np.dot(ray_direction, normal) > 0:
            normal = -normal
        mirror_vec = self.get_direction_vector()

        # Ray components
        v_n = np.dot(ray_direction, normal)
        v_t = np.dot(ray_direction, mirror_vec)

        # Height from center
        r_vec = ray_origin - self.center
        h = np.dot(r_vec, mirror_vec)

        # 1. Handle "Lens-like" power if f is defined
        if self.focal_length is not None and not np.isinf(self.focal_length):
            slope_in = v_t / v_n
            # Reflection reverses the axial direction,
            # so the bending sign is inverted compared to transmission
            slope_out = slope_in + (h / self.focal_length)
        else:
            # Pure Plane Mirror
            slope_out = v_t / v_n

        # 2. Construct reflected direction (flip the normal component)
        # Note: -normal because it's reflecting back
        new_dir = -v_n * normal + (v_n * slope_out) * mirror_vec
        return new_dir / np.linalg.norm(new_dir)


# %% HELPER FUNCTIONS
def get_axis_direction(e1: OpticalElement, e2: OpticalElement) -> np.ndarray:
    """
    Returns the unit vector from element 1 to element 2

    Parameters
    ----------
    e1 : OpticalElement
        Starting element to determine the axis direction

    e2 : OpticalElement
        Ending element to determine the axis direction

    Returns
    -------
    np.ndarray
        unit vector pointing from ``e1`` to ``e2``
    """
    vec = e2.center - e1.center
    return vec / np.linalg.norm(vec)


def get_normal_direction(e1: OpticalElement, e2: OpticalElement) -> np.ndarray:
    """
    Returns the unit vector perpendicular to the direction from ``e1`` to ``e2``.

    Parameters
    ----------
    e1 : OpticalElement
        Starting element to determine the normal direction.
    e2 : OpticalElement
        Ending element to determine the normal direction.

    Returns
    -------
    np.ndarray
        Unit vector obtained by rotating the axis direction by +90°
        (counter clockwise) in the 2D plane.
    """
    # axis direction (unit vector) from e1 to e2

    u = get_axis_direction(e1, e2)

    # rotate by +90°: (x, y) → (‑y, x)
    n = np.array([-u[1], u[0]])

    # ensure unit length (mostly redundant)
    return n / np.linalg.norm(n)


def get_normal_vector(vec: np.ndarray | tuple) -> np.ndarray:
    """
    Returns a unit vector perpendicular to ``vec``

    Parameters
    ----------
    vec : np.ndarray | tuple
        Input vector

    Returns
    -------
    np.ndarray
        Unit vector obtained by rotating ``vec`` direction by +90° (counter clockwise)
        in the 2D plane.
    """
    n = np.array([-vec[1], vec[0]])
    return n / np.linalg.norm(n)


# -----------------------------------------------------------------------------
# Concrete Elements
# -----------------------------------------------------------------------------


# %% MIRRORS
class PlaneMirror(Mirror):
    """A flat mirror represented by a line '|'."""

    # def __init__(self, position, size, angle=0.0, **kwargs):
    def __init__(self, position=ORIGIN, size=1, angle=0, **kwargs):
        super().__init__(position, size, angle, focal_length=np.inf, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class ConcaveMirror(Mirror):
    """A converging mirror (f > 0) represented by ']'."""

    def __init__(self, position=ORIGIN, size=1, focal_length=None, angle=0.0, **kwargs):
        if focal_length:
            if focal_length <= 0:
                raise ValueError("ConcaveMirror must have positive focal_length")
        super().__init__(position, size, angle, focal_length, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "]-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class ConvexMirror(Mirror):
    """A diverging mirror (f < 0) represented by '['."""

    def __init__(self, position=ORIGIN, size=1, focal_length=None, angle=0.0, **kwargs):
        if focal_length:
            if focal_length <= 0:
                raise ValueError("ConvexMirror must have negative focal_length")
        super().__init__(position, size, angle, focal_length, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "[-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class BeamSplitter(OpticalSystem):
    """A cubic BeamSplitter represented by a box with a diagonal."""

    # def __init__(self, position=ORIGIN, size=1, angle=0, **kwargs):
    #     super().__init__(position, size, angle, focal_length=np.inf, **kwargs)

    # Diagonal normal calculation (45 degrees)
    def get_normal(self):
        # Normal of the diagonal (45 deg offset)
        theta = np.deg2rad(self.angle + 135)
        return np.array([np.cos(theta), np.sin(theta)])

    def intersect(self, ray_origin, ray_direction):
        normal = self.get_normal()
        denom = np.dot(ray_direction, normal)
        if np.isclose(denom, 0):
            return None

        t = np.dot(self.center - ray_origin, normal) / denom
        if t < 1e-9:
            return None

        intersection = ray_origin + t * ray_direction
        # Check against diagonal length (size * sqrt(2))
        dist = np.linalg.norm(intersection - self.center)
        if dist > (self.size * np.sqrt(2)) / 2:
            return None
        return intersection

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        if next_element_center is None:
            return ray_direction  # No next element, assume straight

        # 1. Calculate Transmission path
        trans_dir = ray_direction

        # 2. Calculate Reflection path
        # R = I - 2(I.N)N
        normal = self.get_normal()
        # Ensure normal faces incoming ray
        if np.dot(ray_direction, normal) > 0:
            normal = -normal

        reflect_dir = ray_direction - 2 * np.dot(ray_direction, normal) * normal

        # 3. Predict which one points to the next element
        to_next = next_element_center - ray_origin
        to_next = to_next / np.linalg.norm(to_next)

        # Dot product to check alignment
        score_trans = np.dot(trans_dir, to_next)
        score_refl = np.dot(reflect_dir, to_next)

        if score_refl > score_trans:
            return reflect_dir
        else:
            return trans_dir

    def get_local_points(self):
        # Square corners + diagonal definition
        h = self._size / 2.0
        return np.array(
            [
                [-h, -h],
                [h, -h],
                [h, h],
                [-h, h],  # 4 corners
            ]
        )

    def _get_mpl_artist(self):
        # 1. Square
        pts = self.get_critical_points()

        # 2. Diagonal (connects bottom-left to top-right in local space)
        diagonal = np.array([pts[0], pts[2]])

        # 3. Stack them
        all_verts = np.vstack([pts, diagonal])

        style = self._style.copy()
        style.setdefault("closed", False)  # so it doesn't close back from diagonal end
        style.setdefault("joinstyle", "round")  # avoid wierd stuff at the diagonal
        return MplPolygon(all_verts, **style)


# %% LENSES


class ConvergingLens(Lens):
    """A thin converging lens represented by an outwards pointing double-headed arrow."""

    # def __init__(self, *args, **kwargs):
    def __init__(self, position=ORIGIN, size=1, angle=0, focal_length=None, **kwargs):
        kwargs.setdefault("arrowstyle", "<->,head_width=5,head_length=5")
        super().__init__(position, size, angle, focal_length, **kwargs)


class DivergingLens(Lens):
    """A thin diverging lens represented by an inwards pointing double-headed arrow."""

    # def __init__(self, *args, **kwargs):
    def __init__(self, position=ORIGIN, size=1, angle=0, focal_length=None, **kwargs):
        kwargs.setdefault("arrowstyle", "<->,head_width=5,head_length=5")
        kwargs.setdefault("mutation_scale", -1)
        super().__init__(position, size, angle, focal_length, **kwargs)


# %% MARKERS
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
    Adds a text label to the diagram at a specified direction from an anchor.
    Supports .rotate(), .shift(), and .flip().
    """

    def __init__(
        self,
        anchor: tuple | np.ndarray | OpticalElement,
        direction: tuple | np.ndarray | None = None,
        text: str = "label",
        buffer=0.1,
        **kwargs,
    ):
        self.anchor = anchor
        self.direction = (
            np.asarray(direction) if direction is not None else np.array((0, 0))
        )
        self.text = text
        self.buffer = buffer

        # Calculate alignment based on direction vector
        kwargs.setdefault(
            "horizontalalignment",
            self._get_alignment(self.direction[0], "left", "right", "center"),
        )
        kwargs.setdefault(
            "verticalalignment",
            self._get_alignment(self.direction[1], "bottom", "top", "center"),
        )

        # Resolve coordinates from element or array
        base_pos = (
            anchor.get_edge(self.direction)
            if hasattr(anchor, "get_edge")
            else np.asarray(anchor)
        )
        final_pos = base_pos + (self.direction * self.buffer)

        # Initialize base; angle defaults to 0
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
        """Overrides the base draw to apply rotation and filter styles."""
        style = self._style.copy()

        # 1. Map 'edgecolor' to 'color' for the font
        color = style.pop("color", style.pop("edgecolor", "black"))

        # 2. Remove Patch-specific keys
        style.pop("facecolor", None)
        style.pop("linewidth", None)

        # 3. Handle rotation
        # We use self._angle (updated by .rotate()) as the text rotation
        # If 'rotation' was in style, self._angle takes precedence for consistency with rotate()
        style.pop("rotation", None)

        return ax.text(
            self.center[0],
            self.center[1],
            self.text,
            color=color,
            rotation=self._angle,  # This enables .rotate() support
            **style,
        )


class Rectangle(OpticalElement):
    """
    A basic rectangle defined by center, width, and height.
    """

    def __init__(self, position=ORIGIN, width=1.0, height=1.0, angle=0.0, **kwargs):
        self.width = float(width)
        self.height = float(height)
        # Initializing with the default styles defined in the base class
        super().__init__(position, size=max(width, height), angle=angle, **kwargs)

        # Update the default styles without overwriting previously-set values
        defaults = {
            "edgecolor": "k",
            "facecolor": "none",
            "linewidth": 1.5,
            "zorder": 10,
        }
        for k, v in defaults.items():
            self._style.setdefault(k, v)
        # Note: explicit kwargs were already applied by OpticalElement.__init__

    def get_local_points(self):
        """Returns the four corners of the rectangle in local coordinates."""
        w2, h2 = self.width / 2.0, self.height / 2.0
        return np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])

    def _get_mpl_artist(self):
        """Generates a Matplotlib Polygon artist based on current geometry."""
        pts = self.get_critical_points()
        return MplPolygon(pts, closed=True, **self._style)

    def scale(self, scale_factor, about_point=None):
        """Overridden to ensure width and height scale along with the base size."""
        self.width *= scale_factor
        self.height *= scale_factor
        return super().scale(scale_factor, about_point)


class SurroundingRectangle(Rectangle):
    """
    A rectangle that automatically sizes itself to surround one or more OpticalElements.
    """

    def __init__(self, elements, buff=0.2, **kwargs):
        # Handle single element or list of elements
        if not isinstance(elements, (list, tuple, np.ndarray)):
            elements = [elements]

        # Extract global bounding box for all provided elements
        all_min_x, all_min_y = [], []
        all_max_x, all_max_y = [], []

        for el in elements:
            # Use the library's built-in boundary calculation
            x_min, y_min, x_max, y_max = el.get_boundary_box()
            all_min_x.append(x_min)
            all_min_y.append(y_min)
            all_max_x.append(x_max)
            all_max_y.append(y_max)

        g_min_x, g_max_x = min(all_min_x), max(all_max_x)
        g_min_y, g_max_y = min(all_min_y), max(all_max_y)

        # Calculate final center and dimensions including the buffer
        width = (g_max_x - g_min_x) + 2 * buff
        height = (g_max_y - g_min_y) + 2 * buff
        center = np.array([(g_min_x + g_max_x) / 2, (g_min_y + g_max_y) / 2])

        # Use a default highlight style if none provided
        kwargs.setdefault("edgecolor", "C3")
        kwargs.setdefault("linewidth", 2.0)
        kwargs.setdefault("zorder", 5)  # Draw behind elements but above axes

        super().__init__(position=center, width=width, height=height, angle=0.0, **kwargs)


# %% BEAMS
class SimpleBeam(OpticalElement):
    """
    A simple line beam that passes through the centers of provided elements.
    """

    def __init__(self, elements, **kwargs):
        self.elements = elements
        # Default style for beams
        kwargs.setdefault("edgecolor", "C0")
        kwargs.setdefault("linewidth", 1.0)
        kwargs.setdefault("facecolor", "none")
        # Use first element center as dummy position for base class
        super().__init__(elements[0].center, 0, **kwargs)

    def get_path_points(self):
        return np.array([el.center for el in self.elements])

    def _get_mpl_artist(self):
        pts = self.get_path_points()
        # We use Polygon with closed=False to draw a polyline
        return MplPolygon(pts, closed=False, **self._style)


class DivergingBeam(OpticalElement):
    """
    A beam with variable width defined by offsets at each optical element.
    """

    def __init__(self, elements, offsets, **kwargs):
        self.elements = elements
        self.offsets = np.asarray(offsets)
        kwargs.setdefault("color", "C0")
        kwargs.setdefault("alpha", 0.2)
        super().__init__(elements[0].center, 0, **kwargs)

    def _get_mpl_artist(self):
        centers = np.array([el.center for el in self.elements])

        # Calculate the two sides of the beam
        # Top path: centers + offsets
        # Bottom path: centers - offsets (reversed to create a single closed loop)
        top_side = centers + self.offsets
        bottom_side = (centers - self.offsets)[-1:0:-1]

        verts = np.vstack([top_side, bottom_side])
        return MplPolygon(verts, closed=True, **self._style)


class RayTracedBeam(OpticalElement):
    def __init__(self, elements, initial_width=0, divergence=0.0, **kwargs):
        self.elements: list[OpticalElement] = elements
        self.initial_width = initial_width
        self.divergence = np.deg2rad(2 * divergence)

        # Ensure the beam has a default color and alpha
        kwargs.setdefault("color", "C0")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)  # usually cleaner when there are overlaps
        kwargs.setdefault("zorder", 5)
        super().__init__(elements[0].center, 0, **kwargs)

    def _get_line_intersection(self, p1, v1, p2, v2):
        """Helper to find the corner where incident and reflected rays 'meet'."""
        # Solving p1 + t*v1 = p2 + u*v2
        # det = v2_y * v1_x - v2_x * v1_y
        det = v2[1] * v1[0] - v2[0] * v1[1]
        if abs(det) < 1e-9:
            return p1  # Parallel fallback
        t = (v2[0] * (p2[1] - p1[1]) - v2[1] * (p2[0] - p1[0])) / det
        return p1 + t * v1

    def _get_mpl_artist(self):
        if len(self.elements) < 2:
            return MplPolygon(np.array([[0, 0]]), closed=False)

        # 1. Initial Setup: Calculate starting points and directions
        axis = self.elements[1].center - self.elements[0].center
        axis /= np.linalg.norm(axis)
        perp = np.array([-axis[1], axis[0]])

        # Current top and bottom marginal ray positions
        t_pos = self.elements[0].center + perp * (self.initial_width / 2)
        b_pos = self.elements[0].center - perp * (self.initial_width / 2)

        # Initial directions of the marginal rays
        div = self.divergence / 2
        rot_t = np.array([[np.cos(div), -np.sin(div)], [np.sin(div), np.cos(div)]])
        rot_b = np.array([[np.cos(-div), -np.sin(-div)], [np.sin(-div), np.cos(-div)]])
        t_dir = rot_t @ axis
        b_dir = rot_b @ axis

        polys = []

        # 2. Iterate through gaps between elements
        for i in range(len(self.elements) - 1):
            next_el = self.elements[i + 1]

            # Find intersections for both rays with the next element
            t_hit = next_el.intersect(t_pos, t_dir)
            b_hit = next_el.intersect(b_pos, b_dir)

            # Fallback projection if rays miss the physical aperture (for visualization)
            if t_hit is None or b_hit is None:
                normal = next_el.get_normal()
                for ray_p, ray_d in [(t_pos, t_dir), (b_pos, b_dir)]:
                    denom = np.dot(ray_d, normal)
                    if abs(denom) > 1e-9:
                        hit = (
                            ray_p
                            + (np.dot(next_el.center - ray_p, normal) / denom) * ray_d
                        )
                        if ray_p is t_pos:
                            t_hit = hit
                        else:
                            b_hit = hit

                if t_hit is None or b_hit is None:
                    break

            # Create a quadrilateral for this gap: [Top_Start, Bot_Start, Bot_end, Top_end]
            polys.append(np.array([t_pos, b_pos, b_hit, t_hit]))

            # 3. Update for the next segment
            # Calculate new directions after interacting with the element
            future_pos = (
                self.elements[i + 2].center if i + 2 < len(self.elements) else None
            )
            t_dir = next_el.interact(t_hit, t_dir, future_pos)
            b_dir = next_el.interact(b_hit, b_dir, future_pos)

            # The hit points on this surface become the start points for the next gap
            t_pos, b_pos = t_hit, b_hit

        # 4. Create the PolyCollection with the established style
        style = self._style.copy()
        # Map OpticalElement style keys to PolyCollection keywords
        if "facecolor" in style:
            style["facecolors"] = style.pop("facecolor")
        if "edgecolor" in style:
            style["edgecolors"] = style.pop("edgecolor")
        if "linewidth" in style:
            style["linewidths"] = style.pop("linewidth")

        return PolyCollection(polys, **style)

    def draw(self, ax):
        """Override to handle PolyCollection instead of Patch."""
        
        artist = self._get_mpl_artist()
        ax.add_collection(artist)
        return artist

    def get_intersection_with(self, element):
        """
        Return the center point between the two marginal-ray intersections of this beam
        with the provided OpticalElement.

        - element must be one of self.elements (ValueError otherwise).
        - If the element is the first element (beam source), returns its center.
        - If the two marginal rays hit at different points returns their midpoint.
        - If rays miss the element (after fallback), raises RuntimeError.
        """
        if element not in self.elements:
            raise ValueError("Provided element is not part of this beam's element list.")

        idx = self.elements.index(element)
        # If target is the source element, return its center
        if idx == 0:
            return np.asarray(element.center, dtype=float)

        # Initialize marginal ray positions and directions (same as _get_mpl_artist)
        axis = self.elements[1].center - self.elements[0].center
        axis = axis / np.linalg.norm(axis)
        perp = np.array([-axis[1], axis[0]])

        t_pos = self.elements[0].center + perp * (self.initial_width / 2)
        b_pos = self.elements[0].center - perp * (self.initial_width / 2)

        div = self.divergence / 2
        rot_t = np.array([[np.cos(div), -np.sin(div)], [np.sin(div), np.cos(div)]])
        rot_b = np.array([[np.cos(-div), -np.sin(-div)], [np.sin(-div), np.cos(-div)]])
        t_dir = rot_t @ axis
        b_dir = rot_b @ axis

        # Propagate until we compute hits on the target element
        for i in range(len(self.elements) - 1):
            next_el = self.elements[i + 1]

            # Primary intersection test
            t_hit = next_el.intersect(t_pos, t_dir)
            b_hit = next_el.intersect(b_pos, b_dir)

            # Fallback projection if rays miss the aperture (mirrors the visualization fallback)
            if t_hit is None or b_hit is None:
                normal = next_el.get_normal()
                for ray_p, ray_d, assign_to in ((t_pos, t_dir, "t"), (b_pos, b_dir, "b")):
                    denom = np.dot(ray_d, normal)
                    if abs(denom) > 1e-9:
                        hit = (
                            ray_p
                            + (np.dot(next_el.center - ray_p, normal) / denom) * ray_d
                        )
                    else:
                        hit = None
                    if assign_to == "t" and t_hit is None:
                        t_hit = hit
                    if assign_to == "b" and b_hit is None:
                        b_hit = hit

            # If we reached the target element, return the midpoint (or point)
            if i + 1 == idx:
                if t_hit is None or b_hit is None:
                    raise RuntimeError(
                        "Beam does not intersect the requested element (rays missed)."
                    )
                t_hit = np.asarray(t_hit, dtype=float)
                b_hit = np.asarray(b_hit, dtype=float)
                return (t_hit + b_hit) / 2.0

            # Prepare for next gap: update directions and start points
            future_pos = (
                self.elements[i + 2].center if i + 2 < len(self.elements) else None
            )
            t_dir = next_el.interact(t_hit, t_dir, future_pos)
            b_dir = next_el.interact(b_hit, b_dir, future_pos)
            t_pos, b_pos = t_hit, b_hit


# %% FIBERS


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


# -----------------------------------------------------------------------------
# Optical Table (Manager)
# -----------------------------------------------------------------------------
# %%


class OpticalTable:
    """
    The main container for the optical setup.

    Parameters
    ----------
    size : tuple
        (width, height) of the plotting area.
    units: {'inches', 'cm'}
        Units to use for the grid
    title : str, optional
        Title of the plot.
    mode : {'axial', 'top_down'}, optional
        Defines the labeling convention.
        'axial': x-axis is Z (propagation), y-axis is Y.
        'top_down': x-axis is X, y-axis is Y. (default)
    scale_factor : float
        Scaling for the figure size (defaults to 1 unit = 1 inch, roughly).
    dpi : int
        DPI of the output figure. Defaults to 100.
    """

    def __init__(
        self,
        size=(10, 10),
        units: Literal["inches", "cm"] = "cm",
        title="Optical Setup",
        mode: Literal["axial", "top-down"] = "top-down",
        scale_factor=1,
        dpi=100,
    ):
        self.w, self.h = size
        self.mode = mode
        self.elements = []

        # Init figure
        units_scaling = 1 / 2.54 if units == "cm" else 1

        self.fig, self.ax = plt.subplots(
            figsize=(
                self.w * scale_factor * units_scaling,
                self.h * scale_factor * units_scaling,
            ),
            dpi=dpi,
        )
        self.ax.set_title(title)

        # Set limits
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_aspect("equal")

        # Config axes labels
        if mode == "axial":
            self.ax.set_xlabel("Z (Propagation)")
            self.ax.set_ylabel("Y (Height)")
        elif mode == "top-down":
            self.ax.set_xlabel(f"X [{units}]")
            self.ax.set_ylabel(f"Y [{units}]")
        else:
            raise ValueError("``mode`` should be one of 'axial' or 'top-down'")

    def add(self, *elements):
        """Add one or multiple OpticalElements to the table."""
        for el in elements:
            if isinstance(el, OpticalElement):
                self.elements.append(el)
            else:
                raise TypeError(f"Expected OpticalElement, got {type(el)}")
        return self

    def auto_scale(self):
        """Resize all elements on the table to fit raytraced beams."""
        # Find all RayTracedBeams
        beams: list[RayTracedBeam]
        beams = [el for el in self.elements if isinstance(el, RayTracedBeam)]
        if not beams:
            return self  # nothing to do

        for beam in beams:
            for el in beam.elements:
                beam_position = beam.get_intersection_with(el)  # ensure intersections are computed
                top = el.center[1] + el.size / 2
                bot = el.center[1] - el.size / 2
                if beam_position[1] > top:
                    el.size = 2*(beam_position[1]+el.center[1]) * 1.5
                if beam_position[1] < bot:
                    el.size = 2*(el.center[1]-beam_position[1]) * 1.5
                    
        return self




    def show_grid(self, visible=True, alpha=0.2):
        """Toggle the background grid.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        if visible:
            self.ax.grid(visible, alpha=alpha)
            self.ax.set_xticks(np.arange(0, self.w + 1, 1))
            self.ax.set_yticks(np.arange(0, self.h + 1, 1))
        return self

    def hide_ticks(self):
        """
        Remove tick numbers and tick marks from both axes.

        This keeps the grid visible (if enabled) but removes the
        numerical labels and the small marks on the spine.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.ax.tick_params(
            axis="both",  # Apply to both x and y
            which="both",  # Apply to both major and minor ticks
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the left edge are off
            right=False,  # ticks along the right edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False,  # labels along the left edge are off
        )
        return self

    def hide_axis_labels(self):
        """
        Removes the axis labels (X, Y)

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)
        return self

    def render(self):
        """Draws all elements to the axes. Called automatically by show()."""
        # Clear existing patches to avoid duplicates if called multiple times
        try:
            self.ax.patches.clear()
        except AttributeError:
            pass

        for el in self.elements:
            el.draw(self.ax)

    def show(self):
        """Render and display the plot."""
        self.render()
        plt.tight_layout()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.show()

    def save(self, filename, dpi=400):
        """Render and save the plot."""
        self.render()
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")


# %%

if __name__ == "__main__":
    table = OpticalTable(size=(15, 10), dpi=300, title="Example setup").show_grid()

    starting_position = (1, 7)

    # Object
    obj_plane = Plane(starting_position, color="gray")

    p0 = Point(starting_position, size=0.05, color="C0")
    p1 = p0.copy().shift(0.1 * UP).set_color("C1")
    p2 = p1.copy().flip(RIGHT, about_point=p0).set_color("C2")

    table.add(obj_plane, p0, p1, p2)

    diameter = 2

    # Optical elements on the first axis
    f1 = 2
    l1 = ConvergingLens(starting_position, diameter, focal_length=f1).shift(f1 * RIGHT)

    f2 = 2 * f1
    l2 = ConvergingLens(l1, diameter, focal_length=f2).shift((f1 + f2) * RIGHT)

    cube = BeamSplitter(size=diameter, angle=90).move_to((l1.center + l2.center) / 2)

    f3 = -f1
    l3 = DivergingLens(l2, diameter / 2, focal_length=f3).shift((f2 + f3) * RIGHT)

    table.add(l1, l2, l3, cube)

    # Image plane on the first axis
    img_plane = obj_plane.copy().move_to(l3).shift(-f3 * RIGHT).scale(2)
    table.add(img_plane)

    # Optical elements on the second axis
    l4 = l2.copy().rotate(-90, about_point=cube).shift(UP)
    l4.focal_length = 2

    img_plane2 = obj_plane.copy().rotate(-90).next_to(l4, DOWN, buff=l4.focal_length)

    # raytrace the beams
    b0 = RayTracedBeam((p0, l1, cube, l2, l3, img_plane), divergence=10, color=p0.color)
    b1 = RayTracedBeam((p1, l1, cube, l2, l3, img_plane), divergence=10, color=p1.color)
    b2 = RayTracedBeam((p2, l1, cube, l4, img_plane2), divergence=10, color=p2.color)

    table.add(b0, b1, b2)

    fiber_start = b2.get_intersection_with(img_plane2)
    fiber_end = (11, 4)
    cable = Fiber(
        fiber_start, fiber_end, angle_start=-90, color=b2.color
    ).show_connections(end=False)

    box = Rectangle(width=1, height=0.5, facecolor="gray", edgecolor="k").next_to(
        cable.end_point, RIGHT
    )

    table.add(l4, img_plane2, cable, box)

    # add labels
    table.add(
        Label(box, UP, "Spectrometer", fontsize="small"),
        Label(l1, DOWN, "$L_1$"),
        Label(l2, DOWN, "$L_2$"),
        Label(l3, DOWN, "$L_3$").shift(DOWN * diameter / 4),
        Label(l4, LEFT, "$L_4$"),
        Label(img_plane, RIGHT, "Camera", fontweight="bold", buffer=0.2).rotate(-90),
    )

    # hide the numbered ticks on the diagram
    table.hide_ticks()

    # actually plot stuff
    table.show()


# %%
