"""
.. currentmodule:: optical_diagram

`FiberSplitter`
===============

Test of the `FiberSplitter` element, checking its geometry and display

This is both an example of how to use the `FiberSplitter` class, and a test to make sure
it's displayed correctly, especially when the splitter is transformed (rotated, flipped,
etc).
"""

from optical_diagram import DOWN, LEFT, RIGHT, UP, FiberSplitter, OpticalTable

# %%
# Test all alignments
# -------------------
#
# This test ensures that the `alignment` property works as expected.
# We test setting the alignment at initialization and after creation, via the
# `alignment` setter and the `set_alignment` method.
#
# - Left column: alignment set at initialization
# - Middle column: alignment set after creation via setter
# - Right column: alignment set after creation via method

table = OpticalTable(size=(15, 6), title="Test: All Alignments").show_grid()

default_splitter_params = {
    "axis": RIGHT,
    "length": 1,
    "height": 1,
    "color": "k",
}

# at initialization
s1 = FiberSplitter(
    input_pos=(-6, 2), alignment="top", **default_splitter_params
).show_labels()

s2 = FiberSplitter(
    input_pos=(-6, 0), alignment="center", **default_splitter_params
).show_labels()

s3 = FiberSplitter(
    input_pos=(-6, -2), alignment="bottom", **default_splitter_params
).show_labels()

table.add(s1, s2, s3)

# after creation via setter
s4 = FiberSplitter(
    input_pos=(-1, 2), alignment="bottom", **default_splitter_params
).show_labels()
s4.alignment = "top"

s5 = FiberSplitter(
    input_pos=(-1, 0), alignment="bottom", **default_splitter_params
).show_labels()
s5.alignment = "center"

s6 = FiberSplitter(
    input_pos=(-1, -2), alignment="top", **default_splitter_params
).show_labels()
s6.alignment = "bottom"

table.add(s4, s5, s6)


# after creation via method
s7 = (
    FiberSplitter(input_pos=(4, 2), alignment="bottom", **default_splitter_params)
    .show_labels()
    .set_alignment("top")
)

s8 = (
    FiberSplitter(input_pos=(4, 0), alignment="bottom", **default_splitter_params)
    .show_labels()
    .set_alignment("center")
)

s9 = (
    FiberSplitter(input_pos=(4, -2), alignment="top", **default_splitter_params)
    .show_labels()
    .set_alignment("bottom")
)

table.add(s7, s8, s9)

table.hide_axis_labels().hide_ticks()
table.show()

# %%
# Test changing height and length via property and method
# -------------------------------------------------------
#
# This test ensures that the `height` and `length` properties and their corresponding
# setter methods (`set_height` and `set_length`) update the geometry correctly.
#
# - Left column: height/length set at initialization
# - Middle column: height/length set after creation via setter
# - Right column: height/length set after creation via method

table = OpticalTable(size=(15, 10), title="Test: Height and Length Changes").show_grid()

# at initialization
s1 = FiberSplitter(
    input_pos=(-6, 4), axis=RIGHT, length=2, height=2, alignment="top", color="k"
).show_labels()

s2 = FiberSplitter(
    input_pos=(-6, 0), axis=RIGHT, length=2, height=2, alignment="center", color="k"
).show_labels()

s3 = FiberSplitter(
    input_pos=(-6, -4), axis=RIGHT, length=2, height=2, alignment="bottom", color="k"
).show_labels()


table.add(s1, s2, s3)

# after creation via setter
s4 = FiberSplitter(
    input_pos=(-1, 4), axis=RIGHT, length=1, height=1, alignment="top", color="k"
).show_labels()
s4.length = 2
s4.height = 2

s5 = FiberSplitter(
    input_pos=(-1, 0), axis=RIGHT, length=1, height=1, alignment="center", color="k"
).show_labels()
s5.length = 2
s5.height = 2

s6 = FiberSplitter(
    input_pos=(-1, -4), axis=RIGHT, length=1, height=1, alignment="bottom", color="k"
).show_labels()
s6.length = 2
s6.height = 2

table.add(s4, s5, s6)

# after creation via method
s7 = (
    FiberSplitter(
        input_pos=(4, 4), axis=RIGHT, length=1, height=1, alignment="top", color="k"
    )
    .show_labels()
    .set_length(2)
    .set_height(2)
)
s8 = (
    FiberSplitter(
        input_pos=(4, 0), axis=RIGHT, length=1, height=1, alignment="center", color="k"
    )
    .show_labels()
    .set_length(2)
    .set_height(2)
)
s9 = (
    FiberSplitter(
        input_pos=(4, -4), axis=RIGHT, length=1, height=1, alignment="bottom", color="k"
    )
    .show_labels()
    .set_length(2)
    .set_height(2)
)

table.add(s7, s8, s9)

table.hide_axis_labels().hide_ticks()
table.show()

# %%
# Test transforms: rotate, flip, move_to, shift
# ---------------------------------------------
#
# This test ensures that the `rotate`, `flip`, `move_to`, and `shift` methods update
# the geometry correctly. It also tests changing the length/height after a rotation.
#
# - 1st column: `shift` method
# - 2nd column: `move_to` method
# - 3rd column: `rotate` method
# - 4th column: `flip` method


table = OpticalTable(size=(18, 6), title="Test: Complex Transforms").show_grid()

# shift & test that geometry is correctly updated after shift
s1 = (
    FiberSplitter((-7, 0.5), axis=RIGHT, length=1, height=1, color="k")
    .show_labels()
    .shift(UP)
)

s2 = s1.copy().shift(DOWN * 1.5).set_height(2).set_alignment("top")
table.add(s1, s2)

# move_to & test that geometry is correctly updated after move_to
# Note: the anchor point of the splitter is the input position, not the center. We must
# check that this behaves as expected when we move the splitter to a new position.
s3 = s1.copy().move_to((-3, 1.5))
s4 = s3.copy().move_to((-3, -2)).set_height(2).set_alignment("bottom")

table.add(s3, s4)

# rotate & test that geometry is correctly updated after rotation.
s5 = (
    FiberSplitter((1, 1.5), axis=RIGHT, length=1, height=2, color="k")
    .show_labels()
    .rotate(90)
)
s6 = s5.copy().set_length(1.5).set_alignment("top").shift(DOWN * 3 + LEFT)

table.add(s5, s6)

# flip & test that geometry is correctly updated after flip
s7 = s3.copy().flip(axis=UP, about_point=s5)
s8 = (
    s7.copy()
    .set_height(1.5)
    .set_length(2)
    .flip(axis=RIGHT + 0.5 * UP, about_point=s7.out_b_point.center + DOWN)
    .set_alignment("top")
)

table.add(s7, s8)

table.hide_axis_labels().hide_ticks()
table.show()
# %%
