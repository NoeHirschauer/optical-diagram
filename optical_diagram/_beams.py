import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon as MplPolygon

from ._base import OpticalElement

__all__ = ["SimpleBeam", "DivergingBeam", "RayTracedBeam"]


class SimpleBeam(OpticalElement):
    """
    A simple line beam that passes through the centers of provided elements.
    """

    def __init__(self, elements, **kwargs):
        self.elements = elements
        kwargs.setdefault("edgecolor", "C0")
        kwargs.setdefault("linewidth", 1.0)
        kwargs.setdefault("facecolor", "none")
        super().__init__(elements[0].center, 0, **kwargs)

    def get_path_points(self):
        return np.array([el.center for el in self.elements])

    def _get_mpl_artist(self):
        pts = self.get_path_points()
        return MplPolygon(pts, closed=False, **self._style)

     # SimpleBeam is a simple line - treat `color` as edgecolor
    @property
    def color(self):
        return self._style.get("edgecolor", None)
    
    @color.setter
    def color(self, value):
        if value is None:
            self._style.pop("edgecolor", None)
            self._style.pop("color", None)
        else:
            self._style["edgecolor"] = value
            self._style.pop("color", None)

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

    # DivergingBeam is a filled polygon — treat `color` as facecolor
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

    def _get_mpl_artist(self):
        centers = np.array([el.center for el in self.elements])
        top_side = centers + self.offsets
        bottom_side = (centers - self.offsets)[-1:0:-1]
        verts = np.vstack([top_side, bottom_side])
        return MplPolygon(verts, closed=True, **self._style)


class RayTracedBeam(OpticalElement):
    """A beam represented by a filled polygon traced through multiple elements.

    Notes
    -----

    The beam is defined by an initial width and divergence angle. Rays are traced
    from the edges of the beam through each optical element in sequence, adjusting
    their directions based on the element's interaction rules.

    For this to work correctly, all elements must implement `intersect` and `interact`.
    Notably, for lenses and mirrors, these methods are only valid if they have a defined
    focal length.
    """

    def __init__(self, elements, initial_width=0, divergence=0.0, **kwargs):
        """
        Parameters
        ----------
        elements : iterable of OpticalElement
            The sequence of optical elements the beam passes through.
        initial_width : int, optional
            The initial width of the beam. Defaults to 0.
        divergence : float, optional
            The divergence (half-angle) of the beam in degrees. Defaults to 0.0.
        """
        self.elements = elements
        self.initial_width = initial_width
        self.divergence = np.deg2rad(2 * divergence)

        # Ensure the beam has a default color and alpha
        kwargs.setdefault("color", "C0")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)
        kwargs.setdefault("zorder", 5)
        super().__init__(elements[0].center, 0, **kwargs)

    # RayTracedBeam is a filled polygon — treat `color` as facecolor
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

    def _get_line_intersection(self, p1, v1, p2, v2):
        """Helper to find the corner where incident and reflected rays 'meet'."""
        # Solving p1 + t*v1 = p2 + u*v2
        # det = v2_y * v1_x - v2_x * v1_y
        det = v2[1] * v1[0] - v2[0] * v1[1]
        if abs(det) < 1e-9:
            return p1
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

        # Initial directions (with divergence)
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
        # Override draw to add PolyCollection instead of Patch
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

        This is useful for positionning other elements relative to the beam path, for
        example placing an annotation at the focal point of a lens in the beam if the
        beam has an angle.
        
        Parameters
        ----------
        element : OpticalElement
            The target element to find the beam intersection with.

        Returns
        -------
        np.ndarray
            The (x, y) coordinates of the beam intersection with the element.
        
        Raises
        ------
        ValueError
            If the provided element is not part of this beam's element list.
        RuntimeError
            If the beam does not intersect the requested element (rays missed).
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
