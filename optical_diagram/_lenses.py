import numpy as np
from matplotlib.patches import FancyArrowPatch

from ._base import OpticalSystem

__all__ = ["ConvergingLens", "DivergingLens"]


class _Lens(OpticalSystem):
    """Base class for lenses."""

    def get_local_points(self):
        h = self._size / 2.0
        return np.array([[0, -h], [0, h]])

    def _get_mpl_artist(self):
        pts = self.get_critical_points()
        return FancyArrowPatch(pts[0], pts[1], **self._style)

    def interact(self, ray_origin, ray_direction, next_element_center=None):
        r"""Calculate the new ray direction after passing through the lens.

        Uses the thin lens approximation to adjust the ray direction based on
        the lens focal length and the ray's height from the lens center. This
        is a paraxial approximation of the lens behavior.

        Notes
        -----

        The math behind this uses the thin lens equation in slope space:

        .. math::

            \frac{1}{s_i} = \frac{1}{f} - \frac{1}{s_o} \implies
            slope_{out} = slope_{in} - \frac{h}{f}

        where :math:`s_i` is the image distance, :math:`s_o` is the object distance,
        :math:`f` is the focal length, :math:`slope_{in}` is the incoming ray slope,
        :math:`slope_{out}` is the outgoing ray slope, and :math:`h` is the height from
        the lens center.
        """
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


class ConvergingLens(_Lens):
    """A converging lens represented by an outwards pointing double-headed arrow."""

    def __init__(self, position=None, size=1, angle=0, focal_length=None, **kwargs):
        # make sure that the focal is positive
        if focal_length is not None and focal_length <= 0:
            raise ValueError("Focal length for a converging lens must be positive.")

        kwargs.setdefault("arrowstyle", "<->, head_width=5, head_length=5")
        super().__init__(position or (0, 0), size, angle, focal_length, **kwargs)


class DivergingLens(_Lens):
    """A diverging lens represented by an inwards pointing double-headed arrow."""

    def __init__(self, position=None, size=1, angle=0, focal_length=None, **kwargs):
        # make sure that the focal is negative
        if focal_length is not None and focal_length >= 0:
            raise ValueError("Focal length for a diverging lens must be negative.")

        kwargs.setdefault("arrowstyle", "<->,head_width=5,head_length=5")
        kwargs.setdefault("mutation_scale", -1)
        super().__init__(position or (0, 0), size, angle, focal_length, **kwargs)
