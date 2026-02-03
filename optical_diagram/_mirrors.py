import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon as MplPolygon

from ._base import OpticalSystem

__all__ = ["PlaneMirror", "ConcaveMirror", "ConvexMirror", "BeamSplitter"]


class _Mirror(OpticalSystem):
    """Base class for all mirrors (Plane, Concave, Convex)."""

    def get_local_points(self):
        h = self._size / 2.0
        return np.array([[0, -h], [0, h]])

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        r"""Calculate the new ray direction after reflecting off the mirror.

        Uses the same paraxial approximation as lenses, but adds the reflection
        behavior.

        Notes
        -----
        The math behind this uses the thin lens equation in slope space, adding a reflection:

        .. math::

            \frac{1}{s_i} = -\left(\frac{1}{f} - \frac{1}{s_o}\right) \implies
            slope_{out} = - slope_{in} + \frac{h}{f}

        where :math:`s_i` is the image distance, :math:`s_o` is the object distance,
        :math:`f` is the focal length, :math:`slope_{in}` is the incoming ray slope,
        :math:`slope_{out}` is the outgoing ray slope, and :math:`h` is the height from
        the mirror center.

        Note that the negative sign in front of the thin lens equation accounts for the
        reflection
        """

        normal = self.get_normal()
        if np.dot(ray_direction, normal) > 0:
            normal = -normal
        mirror_vec = self.get_direction_vector()
        v_n = np.dot(ray_direction, normal)
        v_t = np.dot(ray_direction, mirror_vec)
        r_vec = ray_origin - self.center
        h = np.dot(r_vec, mirror_vec)
        if self.focal_length is not None and not np.isinf(self.focal_length):
            slope_in = v_t / v_n
            slope_out = slope_in + (h / self.focal_length)
        else:
            slope_out = v_t / v_n
        new_dir = -v_n * normal + (v_n * slope_out) * mirror_vec
        return new_dir / np.linalg.norm(new_dir)


class PlaneMirror(_Mirror):
    """A flat mirror represented by a straight line."""

    def __init__(self, position=None, size=1, angle=0, **kwargs):
        if position is None:
            position = (0.0, 0.0)
        super().__init__(position, size, angle, focal_length=np.inf, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class ConcaveMirror(_Mirror):
    """A converging mirror (f > 0) represented by ``]`` (for light coming from the left)."""

    def __init__(self, position=None, size=1, focal_length=None, angle=0.0, **kwargs):
        if focal_length is not None and focal_length <= 0:
            raise ValueError("ConcaveMirror must have positive focal_length")
        super().__init__(position or (0, 0), size, angle, focal_length, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "]-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class ConvexMirror(_Mirror):
    """A diverging mirror (f < 0) represented by ``[`` (for light coming from the left)."""

    def __init__(self, position=None, size=1, focal_length=None, angle=0.0, **kwargs):
        if focal_length is not None and focal_length >= 0:
            raise ValueError("ConvexMirror must have negative focal_length")
        super().__init__(position or (0, 0), size, angle, focal_length, **kwargs)

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        style = self._style.copy()
        style.setdefault("arrowstyle", "[-")
        return FancyArrowPatch(pts[0], pts[1], **style)


class BeamSplitter(OpticalSystem):
    """A cubic BeamSplitter represented by a box with a diagonal.

    By default, the diagonal splits the cube from bottom-left to top-right.
    """

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
        r"""Calculate the new ray direction after interacting with the beam splitter.

        The beam splitter either reflects or transmits the ray, depending on which
        path leads to the next optical element.

        Notes
        -----
        
        The reflection is calculated assuming that the beamsplitter acts either like
        a plane mirror (reflection) or lets the ray pass straight through (transmission).
        The choice between reflection and transmission is made by checking which
        direction points more closely towards the next optical element.
        """

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
        # Square corners + diagonal
        h = self._size / 2.0
        return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

    def _get_mpl_artist(self):
        # 1. Square
        pts = self.get_critical_points()

        # 2. Diagonal (connects bottom-left to top-right in local space)
        diagonal = np.array([pts[0], pts[2]])

        # 3. Stack them
        all_verts = np.vstack([pts, diagonal])

        # 4. Define style
        style = self._style.copy()
        style.setdefault("closed", False)  # so it doesn't close back from diagonal end
        style.setdefault("joinstyle", "round")  # avoid weird angles at the diagonal
        
        return MplPolygon(all_verts, **style)
