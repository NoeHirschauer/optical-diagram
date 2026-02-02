import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest


# since we are in tests/, import from parent directory. Note that pyproject.toml defines
# the rooot of
from optical_diagram import (
    RIGHT, LEFT, UP, DOWN, UR, UL, DR, DL, ORIGIN,
    PlaneMirror, ConcaveMirror, ConvexMirror, ConvergingLens, DivergingLens,
    BeamSplitter, Plane, Point, OpticalAxis, Label, Rectangle, SurroundingRectangle,
    Fiber, SimpleBeam, DivergingBeam, RayTracedBeam, OpticalTable,
    get_axis_direction, get_normal_direction, get_normal_vector,
)


def test_constants():
    assert np.allclose(RIGHT, np.array((1.0, 0.0)))
    assert np.allclose(LEFT, np.array((-1.0, 0.0)))
    assert np.allclose(UP, np.array((0.0, 1.0)))
    assert np.allclose(DOWN, np.array((0.0, -1.0)))
    assert np.allclose(ORIGIN, np.array((0.0, 0.0)))
    # diagonal checks
    assert np.allclose(UR, UP + RIGHT)
    assert np.allclose(UL, UP + LEFT)
    assert np.allclose(DR, DOWN + RIGHT)
    assert np.allclose(DL, DOWN + LEFT)


def test_helpers_direction_vectors():
    a = Point((0, 0))
    b = Point((1, 1))
    axis = get_axis_direction(a, b)
    assert pytest.approx(np.linalg.norm(axis), rel=1e-6) == 1.0
    n = get_normal_direction(a, b)
    assert pytest.approx(np.dot(axis, n), abs=1e-6) == 0.0
    v = get_normal_vector((1.0, 0.0))
    assert np.allclose(v, np.array([0.0, 1.0]))


def test_basic_elements_and_draw():
    fig, ax = plt.subplots()
    p = Point((0.1, 0.2))
    r = Rectangle(position=(1.0, 1.0), width=2.0, height=0.5)
    pl = Plane((2.0, 0.5), size=1.0)
    pm = PlaneMirror((0.5, 0.5), size=1.0)
    # draw calls should not raise
    ax.add_patch(p._get_mpl_artist())
    ax.add_patch(r._get_mpl_artist())
    ax.add_patch(pl._get_mpl_artist())
    ax.add_patch(pm._get_mpl_artist())


def test_transformations_chain():
    rect = Rectangle((0, 0), width=1, height=1)
    rect.move_to((1, 1))
    assert np.allclose(rect.center, np.array((1.0, 1.0)))
    rect.shift((1, 0))
    assert np.allclose(rect.center, np.array((2.0, 1.0)))
    rect.scale(2.0)
    assert pytest.approx(rect.width) == 2.0
    # rotate about origin moves center
    rect.rotate(90, about_point=(0, 0))
    assert isinstance(rect.angle, float)


def test_lens_and_mirror_interaction_smoke():
    lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)
    ray_origin = np.array((-1.0, 0.0))
    ray_direction = RIGHT.copy()
    inter = lens.intersect(ray_origin, ray_direction)
    assert inter is not None
    new_dir = lens.interact(inter, ray_direction)
    assert pytest.approx(np.linalg.norm(new_dir), rel=1e-6) == 1.0

    mirror = PlaneMirror(position=(2.0, 0.0), size=2.0)
    inter_m = mirror.intersect(np.array((3.0, 0.0)), LEFT.copy())
    assert inter_m is not None
    out = mirror.interact(inter_m, LEFT.copy())
    assert pytest.approx(np.linalg.norm(out), rel=1e-6) == 1.0


def test_beams_and_raytraced_smoke():
    p0 = Point((0.0, 0.0))
    p1 = Point((1.0, 0.0))
    sb = SimpleBeam((p0, p1))
    artist = sb._get_mpl_artist()
    assert artist is not None

    rb = RayTracedBeam((p0, p1), initial_width=0, divergence=0.0)
    mid = rb.get_intersection_with(p1)
    assert np.allclose(mid, p1.center)


def test_divergingbeam_and_surrounding_rectangle():
    p0 = Point((0.0, 0.0))
    p1 = Point((1.0, 0.5))
    db = DivergingBeam((p0, p1), offsets=[(0, 0.1), (0, 0.2)])
    artist = db._get_mpl_artist()
    assert artist is not None

    # SurroundingRectangle around p0 and p1
    sr = SurroundingRectangle([p0, p1], buff=0.1)
    assert sr.width > 0 and sr.height > 0


def test_fiber_and_table_smoke():
    p0 = Point((0.0, 0.0))
    p1 = Point((2.0, 0.0))
    fiber = Fiber(p0, p1, angle_start=0.0, angle_end=0.0)
    artist = fiber._get_mpl_artist()
    assert artist is not None

    table = OpticalTable(size=(4, 3), units="cm", title="Test")
    table.add(p0, p1, fiber)
    # render should not raise
    table.render()