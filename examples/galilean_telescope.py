"""
Galilean telescope (afocal) example
-----------------------------------

A minimal  example showing a Galilean telescope:
- Large converging objective (long focal length)
- Shorter negative-focal diverging eyepiece
- A collimated ("from infinity") input beam approximated by a Point very far to the left
- A RayTracedBeam propagating through the system

This script is intentionally verbose in comments to demonstrate how to declare
and initialize every OpticalElement used.
"""

from optical_diagram import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ConvergingLens,
    DivergingLens,
    Label,
    OpticalTable,
    Plane,
    Point,
    RayTracedBeam,
)

# 1) Create the drawing canvas (OpticalTable)
table = OpticalTable(
    size=(16, 6), title="Galilean Telescope (Afocal)", dpi=150, mode="axial"
)

# 1) Define the objective lens (converging)
#    - position: place it roughly 1/4 from the left
#    - size: physical diameter for visualization
#    - focal_length: positive for converging lens (long focal)

objective_f = 4.0  # focal length of objective (units consistent with table)
objective = ConvergingLens(
    position=(4.0, 3.0), size=1.5, focal_length=objective_f, color="C0"
)


# 3) Define the eyepiece (diverging lens for Galilean telescope)
#    - focal_length: negative for a diverging eyepiece
#    - separation between lenses = f_objective + f_eyepiece (since f_eyepiece < 0)

eyepiece_f = -1.0  # negative focal length (shorter magnitude)
separation = objective_f + eyepiece_f  # < objective_f because eyepiece_f is negative
eyepiece_pos = (objective.center[0] + separation, objective.center[1])
eyepiece = DivergingLens(
    position=eyepiece_pos, size=1.0, focal_length=eyepiece_f, color="C1"
)


# 4) Define a collimated source "at infinity"
#    - We approximate a collimated beam from infinity by placing a Point far to the left.
#    - The RayTracedBeam will use this Point as its first element to compute the incoming axis.
#    - For an afocal system the output rays should be collimated again after the eyepiece.

infty_source = Point(
    position=(-200.0, 3.0), size=0.05, color="k"
)  # far-away point ≈ infinity



# 5) Add a screen/plane further to the right to visualize output directions (optional)

screen_x = eyepiece.center[0] + 8.0
screen = Point(position=(screen_x, 3.0), size=0.05, color="0.5")


# 6) Create a RayTracedBeam
#    - elements: (source, objective, eyepiece, screen)
#    - initial_width: beam diameter at the source (visual)
#    - divergence: small angular spread for a finite-width bundle (deg)

beam = RayTracedBeam(
    elements=(infty_source, objective, eyepiece, screen),
    initial_width=0.6,
    divergence=0.5,
    color="C2",
    alpha=0.25,
)




# Add elements to the table

table.add(infty_source, objective, eyepiece, screen, beam)


# Add labels to explain elements (Label(anchor, direction, text))

table.add(
    Label(objective, DOWN, "Objective\n(Converging, f = {:.1f})".format(objective_f)),
    Label(eyepiece, DOWN, "Eyepiece\n(Diverging, f = {:.1f})".format(eyepiece_f)),
    # Label(infty_source, LEFT, "Collimated source\n(approx. infinity)"),
)

# 9) Cosmetic: hide tick numbers and optionally show a light grid

table.hide_ticks().show_grid(visible=True, alpha=0.15)

# 10) Render and display the example (sphinx-gallery will capture this plot)
table.show()

