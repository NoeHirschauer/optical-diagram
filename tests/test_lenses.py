import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from optical_diagram import (
    ConvergingLens,
    DivergingLens,
    Point,
    RayTracedBeam,
    OpticalTable,
    RIGHT,
)


def test_lens_focal_sign():
    # Converging lens must have positive focal length
    c_lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)
    assert pytest.approx(c_lens.focal_length) == 1.0

    with pytest.raises(ValueError):
        ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=-1.0)

    # Diverging lens must have negative focal length
    d_lens = DivergingLens(position=(0.0, 0.0), size=2.0, focal_length=-1.0)
    assert pytest.approx(d_lens.focal_length) == -1.0

    with pytest.raises(ValueError):
        DivergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)


def test_lens_on_axis_interaction_and_normalized_output():
    # incoming ray on axis

    # Converging lens must produce a unit vector, unchanged direction
    c_lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)

    ray_origin = np.array([-1.0, 0.0])
    ray_direction = RIGHT

    inter = c_lens.intersect(ray_origin, ray_direction)
    assert inter is not None

    new_dir = c_lens.interact(inter, ray_direction)
    assert pytest.approx(np.linalg.norm(new_dir), rel=1e-6) == 1.0

    # Diverging lens must also produce a unit vector, unchanged direction
    d_lens = DivergingLens(position=(0.0, 0.0), size=2.0, focal_length=-1.0)

    inter2 = d_lens.intersect(ray_origin, ray_direction)
    assert inter2 is not None

    new_dir2 = d_lens.interact(inter2, ray_direction)
    assert pytest.approx(np.linalg.norm(new_dir2), rel=1e-6) == 1.0


def test_lens_off_axis_interaction_and_normalized_output():
    # incoming ray above axis

    # Converging lens must produce a unit vector
    c_lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)

    ray_origin = np.array([-1.0, 0.5])
    ray_direction = RIGHT

    inter = c_lens.intersect(ray_origin, ray_direction)
    assert inter is not None

    # the theoretical new direction should be a vector from the intersection
    # to the focal point
    new_dir = c_lens.interact(inter, ray_direction)
    new_dir_th = np.array([1.0, -0.5])  # from (0,0.5) to (1,0)
    new_dir_th /= np.linalg.norm(new_dir_th)
    assert pytest.approx(np.linalg.norm(new_dir), rel=1e-6) == 1.0
    assert np.allclose(new_dir, new_dir_th)

    # Diverging lens must also produce a unit vector
    d_lens = DivergingLens(position=(0.0, 0.0), size=2.0, focal_length=-1.0)

    inter2 = d_lens.intersect(ray_origin, ray_direction)
    assert inter2 is not None

    # for a diverging lens, the new direction should point away from the focal point
    new_dir2 = d_lens.interact(inter2, ray_direction)
    new_dir2_th = np.array([1.0, 0.5])  # from (-1,0) to (0,0.5)
    new_dir2_th /= np.linalg.norm(new_dir2_th)
    assert pytest.approx(np.linalg.norm(new_dir2), rel=1e-6) == 1.0
    assert np.allclose(new_dir2, new_dir2_th)


def test_lens_angled_interaction_and_normalized_output():
    # incoming ray at 45 degrees

    # Converging lens must produce a unit vector
    c_lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)

    # ray from (-1,-2) at 45 degrees
    ray_origin = np.array([-1.0, -2.0])
    ray_direction = np.array([1.0, 1.0]) / np.sqrt(2)

    # this should intersect at (0,-1)
    inter = c_lens.intersect(ray_origin, ray_direction)
    assert inter is not None
    assert np.allclose(inter, np.array([0.0, -1.0]))

    new_dir = c_lens.interact(inter, ray_direction)
    assert pytest.approx(np.linalg.norm(new_dir), rel=1e-6) == 1.0

    # According to paraxial geometric optics, the ray should intersect the image focal
    # plane at y = focal_length * tan(input_angle). We can check that the output direction
    # is consistent with this
    focal_plane_y = c_lens.focal_length * np.tan(np.pi / 4)  # 45 degrees
    focal_plane_x = c_lens.x + c_lens.focal_length
    expected_dir = np.array([focal_plane_x - inter[0], focal_plane_y - inter[1]])
    expected_dir /= np.linalg.norm(expected_dir)
    assert np.allclose(new_dir, expected_dir)

    # Diverging lens must also produce a unit vector
    d_lens = DivergingLens(position=(0.0, 0.0), size=2.0, focal_length=-2.0)

    inter2 = d_lens.intersect(ray_origin, ray_direction)
    assert inter2 is not None
    assert np.allclose(inter2, np.array([0.0, -1.0]))  # similar intersection

    new_dir2 = d_lens.interact(inter2, ray_direction)
    assert pytest.approx(np.linalg.norm(new_dir2), rel=1e-6) == 1.0

    # According to paraxial geometric optics, this ray should intersect the image focal
    # plane at y = focal_length * tan(input_angle). We can check that the output direction
    # is consistent with this
    focal_plane_y2 = d_lens.focal_length * np.tan(np.pi / 4)  # 45 degrees
    focal_plane_x2 = d_lens.x + d_lens.focal_length
    expected_dir2 = np.array([inter2[0] - focal_plane_x2, inter2[1] - focal_plane_y2])
    expected_dir2 /= np.linalg.norm(expected_dir2)
    assert np.allclose(new_dir2, expected_dir2)


def test_lens_artist_and_draw_returns_artist():
    fig, ax = plt.subplots()
    for lens in (
        ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0),
        DivergingLens(position=(1.0, 0.0), size=2.0, focal_length=-1.0),
    ):
        artist = lens._get_mpl_artist()
        assert artist is not None
        artist2 = lens.draw(ax)
        assert artist2 is not None


def test_lens_with_raytracedbeam_and_table_renders():
    p0 = Point((-2.0, 0.0))
    p1 = Point((2.0, 0.0))
    rb = RayTracedBeam((p0, p1), initial_width=0, divergence=0.0)

    lens = ConvergingLens(position=(0.0, 0.0), size=2.0, focal_length=1.0)

    # The lens should intersect the beam's central ray and interact without error
    inter = lens.intersect(p0.center, RIGHT)
    if inter is not None:
        out = lens.interact(inter, RIGHT)
        assert pytest.approx(np.linalg.norm(out), rel=1e-6) == 1.0

    # Ensure artists exist and OpticalTable.render does not raise
    table = OpticalTable(size=(6, 4), units="cm", title="Lens Test")
    table.add(p0, p1, lens, rb)
    table.render()
