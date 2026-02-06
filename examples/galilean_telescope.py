"""
.. currentmodule:: optical_diagram

Galilean telescope (afocal) example
===================================

A minimal  example showing a Galilean telescope.

The setup consists of:

- a large converging objective (`ConvergingLens` element)
- a shorter negative-focal diverging eyepiece (`DivergingLens` element)
- a `RayTracedBeam` propagating through the system for on-axis and off-axis collimated sources

This script is intentionally verbose in comments to demonstrate how to declare
and initialize every OpticalElement used.
"""

# %%
# Import necessary classes from optical_diagram

from optical_diagram import (
    DOWN,
    RIGHT,
    ConvergingLens,
    DivergingLens,
    Label,
    OpticalTable,
    Plane,
    Point,
    RayTracedBeam,
)

# %%
# 1) Initialisation
# -----------------
#
# Define the dimensions of the plot and prepare collectors for elements that will be
# added later

diagram_size = (12, 6)  # width, height

# collectors to hold objects until the final cell
_elements = []
_beams = []
_labels = []

# %%
# 2) Define the lenses and planes
# -------------------------------
#
# We start by defining the objective lens and the object plane. Note how we use
# relative positioning to place all the elements in the diagram. This way, only editing
# the position of the ``object_plane`` updates the position of the other elements, and
# changing the focal lengths or sizes of elements automatically updates the layout

# Define the object plane (entrance pupil plane) and objective lens

diameter = diagram_size[1] * 0.5  # 50% of table height

object_plane = Plane(position=(-4, 0), size=diameter)

objective_f = 4.0  # focal length of objective (units consistent with table)
objective = ConvergingLens(size=diameter, focal_length=objective_f).next_to(
    object_plane, RIGHT, buff=objective_f
)

# store them for later addition to the table
_elements.extend([object_plane, objective])

# Define the eyepiece (diverging lens for Galilean telescope)

eyepiece_f = -1.0  # negative focal length (and of shorter magnitude)
separation = objective_f + eyepiece_f  # distance between lenses for an afocal system

eyepiece = DivergingLens(
    position=objective,  # setting another OpticalElement as reference uses its center
    size=diameter / 2,
    focal_length=eyepiece_f,
).shift(separation * RIGHT)

_elements.append(eyepiece)

# %%
# 3) Define the position of the output pupil plane
# ------------------------------------------------
#
# The output pupil is defined as the image of the input pupil (objective) through the
# optical system. This is not strictly necessary but helps to define intersection planes
# for the `RayTracedBeam`.
#
# Note that in our simple situation, we could set it at an arbitrary distance after the
# eyepiece, since the output rays are collimated. However, we use this opportunity to
# display other operations on OpticalElements.

dist_eyepiece_to_pupil = separation * eyepiece.size / objective.size  # similar triangles
pupil_plane = (
    object_plane.copy().move_to(eyepiece).shift(dist_eyepiece_to_pupil * RIGHT).scale(0.5)
)

_elements.append(pupil_plane)

# %%
# 4) Trace the beams through the system
# --------------------------------------
# We define a collimated source "at infinity" (on-axis and off-axis beams) using 2
# approaches to demonstrate different ways of defining the source & the beam.
#
# 1. On-axis source: use a Plane as the source element, and define a RayTracedBeam with
#    zero divergence to represent collimated light.
# 2. Off-axis source: use a Point source off-axis with zero size, and define a small
#    divergence angle to represent a slightly off-axis collimated beam.

on_axis_beam = RayTracedBeam(
    (object_plane, objective, eyepiece, pupil_plane),  # list of elements to trace through
    initial_width=diameter * 0.8,  # full diameter at the source
    divergence=0.0,  # perfectly collimated
    color="C0",
)

_beams.append(on_axis_beam)

off_axis_source = Point(position=(-1000, 50), size=0, color="C1")
off_axis_beam = RayTracedBeam(
    (off_axis_source, objective, eyepiece, pupil_plane),
    initial_width=0,
    divergence=diameter / 100,  # small angular spread
    color="C1",
)

_beams.append(off_axis_beam)

# %%
# 5) Add labels to elements
# -------------------------
#
# These labels will be shown in the diagram. You can use math mode and any text
# customization supported by matplotlib's `Text` class.

_label_obj = Label(
    objective,
    DOWN,
    f"Objective\n($f = {objective.focal_length:.1f}$)",
    fontsize="small",
)
_label_eye = Label(
    eyepiece,
    DOWN,
    f"Eyepiece\n($f = {eyepiece.focal_length:.1f}$)",
    fontsize="small",
    buffer=diameter / 4 + 0.1,
)

_labels.extend([_label_obj, _label_eye])

# %%
# 6) Create the diagram & render
# ------------------------------
#
# Finally, we create the OpticalTable, add all the elements, beams and labels, and
# render the diagram.
#
# Note that we do this at the end of the script because it is run in a Jupyter notebook,
# where exiting a cell with a variable displays it automatically.
#
# If running as a standalone script, you could create the table at the start and add
# elements as you go.
#
# In any case, the elements are only drawn/computed when `OpticalTable.show()` is called.

table = OpticalTable(
    size=diagram_size, title="Galilean Telescope (Afocal)", dpi=200, mode="axial"
)

# add all the collected items in a single call
table.add(*_elements, *_beams, *_labels)

# customize appearance: hide ticks, show grid
table.hide_ticks().show_grid(visible=True, alpha=0.15)

# render and display
table.show()

