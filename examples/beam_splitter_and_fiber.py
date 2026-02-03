"""
Beam splitter and fiber example
===============================

.. currentmodule:: optical_diagram

This example shows how to create a setup with a beam splitter that splits an incoming
beam into two paths, one of which is coupled into an optical fiber leading to a box.

We demonstrate how to use the `BeamSplitter` and `Fiber` elements, as well as raytracing
to properly position the fiber at the output of the optical system. We also illustrate
how to use the `.color` property to color rays consistently.
"""

from optical_diagram import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    BeamSplitter,
    ConvergingLens,
    DivergingLens,
    Fiber,
    Label,
    OpticalTable,
    Plane,
    Point,
    RayTracedBeam,
    Rectangle,
)

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
cable = Fiber(fiber_start, fiber_end, angle_start=-90, color=b2.color).show_connections(
    end=False
)

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
