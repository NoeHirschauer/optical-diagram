import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from optical_diagram import (
    PlaneMirror,
    ConcaveMirror,
    ConvexMirror,
    ConvergingLens,
    DivergingLens,
    Point,
    RayTracedBeam,
    OpticalTable,
)


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_plane_mirror_on_axis_reflection_formula_and_normalization():
    m = PlaneMirror(position=(0.0, 0.0), size=4.0, angle=0.0)

    # on-axis, normal incidence
    ray_origin = np.array([-1.0, 0.0])
    ray_dir = _norm(np.array([1.0, 0.0]))

    inter = m.intersect(ray_origin, ray_dir)
    assert inter is not None

    # the intersection should be at the center of the mirror
    assert np.allclose(inter, m.center)

    out = m.interact(inter, ray_dir)
    # expected reflection: horizontal incidence -> horizontal reflection (-x direction)
    n = _norm(m.get_normal())
    expected = ray_dir - 2 * np.dot(ray_dir, n) * n
    assert pytest.approx(np.linalg.norm(out), rel=1e-6) == 1.0
    assert np.allclose(_norm(out), _norm(expected))


def test_plane_mirror_at_angle_reflection_formula_and_normalization():
    m = PlaneMirror(position=(0.0, 0.0), size=4.0, angle=45.0)

    # incoming ray at 0 deg (horizontal, at 45 deg of mirror normal)
    ray_origin = np.array([-1.0, 0.0])
    ray_dir = _norm(np.array([1.0, 0.0]))

    inter = m.intersect(ray_origin, ray_dir)
    assert inter is not None

    # the intersection should be at the center of the mirror
    assert np.allclose(inter, m.center)

    out = m.interact(inter, ray_dir)
    # expected reflection: 0 deg incidence on 45 deg mirror -> 90 deg reflection (down)
    n = _norm(m.get_normal())
    expected = _norm(ray_dir - 2 * np.dot(ray_dir, n) * n)
    assert pytest.approx(np.array([0.0, -1.0])) == expected
    assert pytest.approx(np.linalg.norm(out), rel=1e-6) == 1.0
    assert np.allclose(_norm(out), _norm(expected))


@pytest.mark.parametrize(
    "origin,direction",
    [
        (np.array([-1.0, 0.0]), _norm(np.array([1.0, 0.0]))),  # on-axis
        (np.array([-1.0, 0.5]), _norm(np.array([1.0, 0.0]))),  # off-axis
        (np.array([-1.0, -2.0]), _norm(np.array([1.0, 1.0]))),  # angled
    ],
)
def test_curved_mirrors_match_lenses_up_to_sign(origin, direction):
    # Use large apertures to avoid misses
    size = 4.0
    focal = 1.0

    conv_lens = ConvergingLens(position=(0.0, 0.0), size=size, focal_length=focal)
    conc_mirror = ConcaveMirror(position=(0.0, 0.0), size=size, focal_length=focal)

    div_lens = DivergingLens(position=(0.0, 0.0), size=size, focal_length=-focal)
    conv_mirror = ConvexMirror(position=(0.0, 0.0), size=size, focal_length=-focal)

    # mirror should produce same focusing but with propagation reversed (sign flip on
    # the propagation direction (x axis here))

    # converging lens vs concave mirror
    il = conv_lens.intersect(origin, direction)
    im = conc_mirror.intersect(origin, direction)
    assert il is not None and im is not None

    out_l = _norm(conv_lens.interact(il, direction))
    out_m = _norm(conc_mirror.interact(im, direction))
    assert np.allclose(out_m, out_l * np.array([-1, 1]), atol=1e-6)

    # diverging lens vs convex mirror
    il2 = div_lens.intersect(origin, direction)
    im2 = conv_mirror.intersect(origin, direction)
    assert il2 is not None and im2 is not None

    out_l2 = _norm(div_lens.interact(il2, direction))
    out_m2 = _norm(conv_mirror.interact(im2, direction))
    assert np.allclose(out_m2, out_l2 * np.array([-1, 1]), atol=1e-6)


def test_mirror_artist_and_draw_returns_artist():
    fig, ax = plt.subplots()
    for mirror in (
        PlaneMirror(position=(0.0, 0.0), size=2.0),
        ConcaveMirror(position=(1.0, 0.0), size=2.0, focal_length=1.0),
        ConvexMirror(position=(2.0, 0.0), size=2.0, focal_length=-1.0),
    ):
        artist = mirror._get_mpl_artist()
        assert artist is not None
        artist2 = mirror.draw(ax)
        assert artist2 is not None


def test_mirrors_with_table_render_smoke():
    # make sure that nothing raises during rendering & interaction with RayTracedBeam

    p0 = Point((-2.0, 0.0))
    p1 = Point((2.0, 0.0))
    rb = RayTracedBeam((p0, p1), initial_width=0, divergence=0.0)

    m = PlaneMirror(position=(0.0, 0.0), size=4.0)
    cm = ConcaveMirror(position=(1.0, 0.0), size=4.0, focal_length=1.0)

    table = OpticalTable(size=(6, 4), units="cm", title="Mirror Test")
    table.add(p0, p1, m, cm, rb)
    table.render()
