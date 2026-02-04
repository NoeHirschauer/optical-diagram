import numpy as np
import pytest

from optical_diagram._base import (
    OpticalElement,
    RIGHT,
    UP,
    ORIGIN,
    get_axis_direction,
    get_normal_direction,
    get_normal_vector,
)


# Minimal concrete subclass so we can instantiate OpticalElement
class DummyElement(OpticalElement):
    def __init__(self, position=ORIGIN, size=1.0, angle=0.0, **kwargs):
        super().__init__(position=position, size=size, angle=angle, **kwargs)

    def _get_mpl_artist(self):
        # simple patch that uses style values; tests don't render, just ensure method exists
        import matplotlib.patches as patches

        return patches.Circle(self.center, radius=self._size / 2.0, **self._style)


def test_geometric_properties_and_copy():
    e = DummyElement((1.0, 2.0), size=2.5, angle=15.0)
    assert np.allclose(e.center, np.array((1.0, 2.0)))
    assert pytest.approx(e.x) == 1.0
    assert pytest.approx(e.y) == 2.0
    assert pytest.approx(e.size) == 2.5
    assert pytest.approx(e.angle) == 15.0

    c = e.copy()
    assert c is not e
    assert np.allclose(c.center, e.center)
    assert pytest.approx(c.size) == e.size
    assert pytest.approx(c.angle) == e.angle


def test_rotation_matrix_and_local_critical_boundary_edge():
    e = DummyElement((0.0, 0.0), size=2.0, angle=90.0)
    R = e.get_rotation_matrix()
    expected_R = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(R, expected_R)

    local = e.get_local_points()
    assert np.allclose(local, np.array([[0.0, 1.0], [0.0, -1.0]]))

    crit = e.get_critical_points()
    # rotated local points should be [[-1,0],[1,0]] (for 90 deg) plus center (0,0)
    assert crit.shape == (2, 2)
    bx = e.get_boundary_box()
    assert len(bx) == 4

    # get_edge returns a point offset from center consistent with boundary box extents
    right_edge = e.get_edge(RIGHT)
    # boundary box format is (xmin, ymin, xmax, ymax)
    assert pytest.approx(right_edge[0]) == bx[2]


def test_normals_and_direction_vectors():
    e0 = DummyElement((0, 0), size=2.0, angle=0.0)
    assert np.allclose(e0.get_normal(), np.array([1.0, 0.0]))
    assert np.allclose(e0.get_direction_vector(), np.array([0.0, 1.0]))

    e90 = DummyElement((0, 0), size=2.0, angle=90.0)
    assert np.allclose(e90.get_normal(), np.array([0.0, 1.0]))
    assert np.allclose(e90.get_direction_vector(), np.array([-1.0, 0.0]))


def test_intersect_and_interact_behavior():
    e = DummyElement((0.0, 0.0), size=2.0, angle=0.0)
    ray_origin = np.array([-1.0, 0.0])
    ray_dir = RIGHT
    inter = e.intersect(ray_origin, ray_dir)
    assert inter is not None
    assert np.allclose(inter, np.array([0.0, 0.0]))

    # parallel ray -> None
    assert e.intersect(np.array((0.0, 0.0)), UP) is None

    # ray pointing away -> None
    assert e.intersect(np.array((1.0, 0.0)), RIGHT) is None

    # default interact returns unchanged direction
    out = e.interact(inter, ray_dir)
    assert np.allclose(out, ray_dir)


def test_transformations_move_shift_rotate_scale_flip_next_to():
    e = DummyElement((0.0, 0.0), size=2.0)
    e.move_to((1.0, 1.0))
    assert np.allclose(e.center, np.array((1.0, 1.0)))

    e.shift((1.0, 0.0))
    assert np.allclose(e.center, np.array((2.0, 1.0)))

    e.scale(2.0)
    assert pytest.approx(e.size) == 4.0

    # rotate about origin moves center accordingly
    e.rotate(90.0, about_point=(0.0, 0.0))
    assert isinstance(e.angle, float)
    # center should be rotated: (2,1) -> (-1,2)
    assert np.allclose(e.center, np.array([-1.0, 2.0]), atol=1e-7)

    # flip across UP (vertical) about origin: ( -1,2 ) -> (1,2)
    pre_angle = e.angle
    e.flip(axis=UP, about_point=(0.0, 0.0))
    assert np.allclose(e.center[0], 1.0)
    # angle updated by formula 2*phi - theta; phi for UP is 90 deg
    assert pytest.approx(e.angle) == 2 * 90.0 - pre_angle

    # next_to should place element to the right of another element
    a = DummyElement((0.0, 0.0), size=2.0)
    b = DummyElement((2.0, 2.0), size=3.0)
    b.next_to(a, direction=RIGHT, buff=0.1)
    assert b.center[0] > a.center[0]


def test_color_and_style_setters():
    e = DummyElement((0.0, 0.0), size=1.0)
    assert e.color is None

    # set via property
    e.color = "red"
    assert e._style.get("edgecolor") == "red"
    assert e.color == "red"

    # clear color
    e.color = None
    assert e._style.get("edgecolor", None) is None
    assert e._style.get("facecolor", None) is None

    # set_color convenience returns self and sets color
    ret = e.set_color("green")
    assert ret is e
    assert e._style.get("edgecolor") == "green"

    # set_style updates style dict
    e.set_style(alpha=0.5)
    assert pytest.approx(e._style.get("alpha"), rel=1e-6) == 0.5


def test_helper_functions_for_vectors():
    a = DummyElement((0.0, 0.0))
    b = DummyElement((1.0, 1.0))
    axis = get_axis_direction(a, b)
    assert pytest.approx(np.linalg.norm(axis), rel=1e-6) == 1.0

    n = get_normal_direction(a, b)
    assert pytest.approx(np.dot(axis, n), abs=1e-6) == 0.0

    v = get_normal_vector((1.0, 0.0))
    assert np.allclose(v, np.array([0.0, 1.0]))
