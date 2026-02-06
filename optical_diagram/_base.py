from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

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
    "OpticalElement",
    "OpticalSystem",
    "Group",
    "get_axis_direction",
    "get_normal_direction",
    "get_normal_vector",
]

# Constants
RIGHT = np.array((1.0, 0.0))
LEFT = np.array((-1.0, 0.0))
UP = np.array((0.0, 1.0))
DOWN = np.array((0.0, -1.0))
ORIGIN = np.array((0.0, 0.0))

UR = UP + RIGHT
UL = UP + LEFT
DR = DOWN + RIGHT
DL = DOWN + LEFT


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

        # Map shorthand color to the correct slot. For this to work, an override
        # of the color property is needed in subclasses that want facecolor semantics.
        # By default, we map to edgecolor (as implemented in @color.setter).
        if _col is not None:
            self.color = _col

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
        """float: The current x-coordinate of the element's center."""
        return self._center[0]

    @property
    def y(self):
        """float: The current y-coordinate of the element's center."""
        return self._center[1]

    @property
    def angle(self):
        """float: The current rotation angle in degrees."""
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
        """
        Return the 2D rotation matrix based on the element's angle.
        """
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
        r"""
        Calculate intersection point of a ray with this element's plane.

        Parameters
        ----------
        ray_origin : array
            The (x, y) starting point of the ray.
        ray_direction : array
            The (x, y) direction vector of the ray (should be normalized).

        Returns
        -------
        intersection : np.array or None
            The (x, y) intersection point, or None if no intersection occurs.

        Notes
        -----
        The intersection is calculated using the plane equation and line equation.

        Plane: :math:`(P - Center) \cdot Normal = 0`
        Line: :math:`P = Origin + t \cdot Direction`
        Solving for t gives the intersection point:

        .. math::

            t = \frac{(Center - Origin) \cdot Normal}{Direction \cdot Normal}
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
        """Physics logic: How does this element alter an incoming ray?

        This method should be overridden by subclasses representing lenses, mirrors, etc.
        By default, it returns the ray direction unchanged (i.e., transparent element).

        Parameters
        ----------
        ray_origin : array
            Incoming ray origin point.
        ray_direction : array
            Incoming direction vector.
        next_element_center : array, optional
            Center of the next optical element in the ray path (if applicable).

        Returns
        -------
        new_direction : array
            The normalized direction vector after interaction.

        Notes
        -----

        The default implementation assumes no interaction (e.g., glass slab). However,
        this call signature must be maintained for compatibility with ray tracing logic.
        """
        # Default behavior: Pass through unchanged (Glass slab / Window / Free space)
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

    # Simplified color proxy: default maps to edgecolor.
    # Concrete subclasses can override this behavior when they want facecolor semantics.
    @property
    def color(self):
        """Shorthand color proxy.


        Defaults to edgecolor; subclasses may override to map to facecolor instead.
        If edgecolor is not set, falls back to facecolor.
        If neither is set, returns None.
        """
        return self._style.get("edgecolor", self._style.get("facecolor", None))

    @color.setter
    def color(self, value):
        """Set shorthand color -> edgecolor by default. Subclasses may override."""
        if value is None:
            self._style.pop("edgecolor", None)
            self._style.pop("facecolor", None)
        else:
            self._style["edgecolor"] = value
            self._style.pop("color", None)

    # small compatibility wrappers so existing call-sites keep working
    def set_color(self, color):
        """Set the shorthand color.

        Note
        ----
        This is equivalent to setting the `color` property directly.
        """
        self.color = color
        return self

    def set_facecolor(self, color):
        """Set the facecolor of the element."""
        self.facecolor = color
        return self

    def set_edgecolor(self, color):
        """Set the edgecolor of the element."""
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
    Abstract base class for optical elements that can have a focal length.
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
        """float or None: The focal length of the optical system.

        If None, the element has no focusing power.
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = value


class Group(OpticalElement):
    """A collection of OpticalElements treated as a single unit."""

    def __init__(self, elements, **kwargs):
        # Compute center as the centroid of all elements
        centers = np.array([el.center for el in elements])
        group_center = np.mean(centers, axis=0)

        # Compute size as the max distance from center to any element's edge
        max_dist = 0.0
        for el in elements:
            min_x, min_y, max_x, max_y = el.get_boundary_box()
            corners = np.array(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]
            )
            dists = np.linalg.norm(corners - group_center, axis=1)
            max_dist = max(max_dist, np.max(dists))

        super().__init__(position=group_center, size=2 * max_dist, angle=0.0, **kwargs)
        self.elements = elements

    def add(self, *elements):
        """Add one or more elements to the group and update its geometry.

        This method appends the given elements to ``self.elements`` and then
        recomputes the group's center and size based on all contained elements.
        The new center is taken as the centroid of the element centers, and the
        size is updated to twice the maximum distance from the center to any
        corner of any element's boundary box.

        Parameters
        ----------
        *elements
            One or more :class:`OpticalElement` instances to be added to the
            group.
        """
        self.elements.extend(elements)
        
        # Recompute center and size
        centers = np.array([el.center for el in self.elements])
        self.center = np.mean(centers, axis=0)

        max_dist = 0.0
        for el in self.elements:
            min_x, min_y, max_x, max_y = el.get_boundary_box()
            corners = np.array(
                [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]
            )
            dists = np.linalg.norm(corners - self.center, axis=1)
            max_dist = max(max_dist, np.max(dists))
        self._size = 2 * max_dist

    def _get_mpl_artist(self):
        # Groups do not have a direct artist; they render their elements instead.
        raise NotImplementedError("Group does not have a direct matplotlib artist.")

    def draw(self, ax):
        """Draw all elements in the group."""
        artists = []
        for el in self.elements:
            artist = el.draw(ax)
            artists.append(artist)
        return artists

    # ----- Delegated transforms (update group props AND elements) -----
    def move_to(self, target):
        if isinstance(target, OpticalElement):
            target = target.center
        else:
            target = np.asarray(target, float)
        delta = target - self.center
        # Update group center
        self.center = target
        # Apply to elements
        for el in self.elements:
            el.shift(delta)
        return self

    def shift(self, vector):
        vec = np.asarray(vector, float)
        # Update group center
        self.center = self.center + vec
        # Apply to elements
        for el in self.elements:
            el.shift(vec)
        return self

    def rotate(self, angle_degrees, about_point=None):
        # Resolve pivot
        if about_point is None:
            pivot = self.center
        elif isinstance(about_point, OpticalElement):
            pivot = about_point.center
        else:
            pivot = np.asarray(about_point, float)

        # Update group angle
        self._angle += angle_degrees

        # If rotating about a pivot not equal to group's center, update group's center
        if pivot is not None:
            rad = np.deg2rad(angle_degrees)
            rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            self.center = rot_matrix @ (self.center - pivot) + pivot

        # Delegate to elements
        for el in self.elements:
            el.rotate(angle_degrees, about_point=pivot)
        return self

    def flip(self, axis=UP, about_point=None):
        # Resolve pivot
        if about_point is None:
            pivot = self.center
        elif isinstance(about_point, OpticalElement):
            pivot = about_point.center
        else:
            pivot = np.asarray(about_point, float)

        # Normalize axis
        axis_vec = np.asarray(axis, float)
        axis_unit = axis_vec / np.linalg.norm(axis_vec)

        # Update group's center by reflection
        v = self.center - pivot
        v_reflected = 2 * np.dot(v, axis_unit) * axis_unit - v
        self.center = pivot + v_reflected

        # Update group's angle
        axis_angle = np.degrees(np.arctan2(axis_unit[1], axis_unit[0]))
        self._angle = 2 * axis_angle - self._angle

        # Delegate to elements
        for el in self.elements:
            el.flip(axis=axis, about_point=pivot)
        return self

    def scale(self, scale_factor, about_point=None):
        # Resolve pivot
        if about_point is None:
            pivot = self.center
        elif isinstance(about_point, OpticalElement):
            pivot = about_point.center
        else:
            pivot = np.asarray(about_point, float)

        # Update group's size
        self._size *= scale_factor

        # Update group's center if scaling about another point
        if pivot is not None:
            relative_vector = self.center - pivot
            self.center = (scale_factor * relative_vector) + pivot

        # Delegate to elements
        for el in self.elements:
            el.scale(scale_factor, about_point=pivot)
        return self

    def next_to(self, other, direction=RIGHT, buff=0.0):
        target_point = other.get_edge(direction)
        my_anchor = self.get_edge(-np.array(direction))
        shift_vec = target_point - my_anchor + np.array(direction) * buff

        # Update group center
        self.center = self.center + shift_vec

        # Apply to elements
        for el in self.elements:
            el.shift(shift_vec)
        return self


# helper functions
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
