
import sys
import os

import matplotlib
matplotlib.use('Qt5Agg')  # ou 'TkAgg'
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from optical_diagram import (
    OpticalTable,
    ConvergingLens,
    Point,
    RayTracedBeam,
    Plane,
    Label,
    Arrow,
    UP, 
    DOWN, 
    LEFT, 
    RIGHT,
    DR,
    DL,
    UL,
    UR
)

# Créer une table optique
table = OpticalTable(size=(10, 6), title="Système Optique Simple", dpi=150)
table.show_grid(visible=True, alpha=0.3)

# Définir la position de départ
start_position = (1, 3)
dist_leds = 1



# Créer un plan matrice de leds
leds_plane = Plane(start_position, size=dist_leds*2, color="gray")
table.add(leds_plane)

# Créer des points sources

l0 = Point(start_position, size=0.05, color="black", facecolor="black")
l1 = Point(start_position, size=0.05, color="black", facecolor="black").shift(dist_leds * UP)
l2 = Point(start_position, size=0.05, color="black", facecolor="black").shift(dist_leds * DOWN)
table.add(l0, l1, l2)


# Créer une lentille convergente
lens1_center = 3 * RIGHT
lens1 = ConvergingLens(start_position, size=2, focal_length=1).shift(lens1_center)
table.add(lens1)


# Créer un plan objet
focal_plane_center = lens1.center + lens1.focal_length * LEFT
obj_plane = Plane(focal_plane_center, size=2, color="gray", alpha=0.3)
table.add(obj_plane)

# Créer des points sources
defocus = 0.5
p0 = Point(focal_plane_center, size=0.05, color="red", facecolor="red")
p2 = Point(focal_plane_center+defocus*LEFT, size=0.05, color="blue", facecolor="blue")

# Créer un point source virtuel sur le plan objet
delta_x = defocus / (focal_plane_center[0]-defocus-start_position[0]) * dist_leds
p1 = Point(focal_plane_center, size=0.05, color="blue", facecolor="blue", alpha=0.5).shift(delta_x * UP)
table.add(p0, p1, p2)


# Créer une lentille convergente
lens2 = ConvergingLens(start_position, size=2, focal_length=1).shift((6, 0))
table.add(lens2)


# Créer un plan image
image_plane = Plane(lens2.center + lens2.focal_length * RIGHT, size=2, color="black")
table.add(image_plane)

# Créer des faisceaux de rayons
beam2 = RayTracedBeam((l2, p0, lens1, lens2, image_plane), initial_width=0.03, divergence=0, color=p0.color)
beam3 = RayTracedBeam((l2, p2, lens1, lens2, image_plane), initial_width=0.03, divergence=0, color=p1.color)
table.add(beam2, beam3)

table.auto_scale()

# Ajouter des étiquettes
table.add(
    Label(leds_plane, DOWN, "Plan de LEDs", fontsize="small"),
    Label(obj_plane, DOWN, "Plan Objet", fontsize="small"),
    Label(lens1, DOWN, "$Obj$", fontsize="small"),
    Label(lens2, DOWN, "$L_2$", fontsize="small"),
    Label(image_plane, DOWN, "Plan Image", fontsize="small"),
    Label(p0, 0.5*DR, "P_{f}", color="red", fontsize="small"),
    Label(p1, 0.5*UR, "P_{v}", color="blue", fontsize="small"),
    Label(p2, 0.5*DL, "P_{d}", color="blue", fontsize="small")
)

# Ajouter des flèches annotées
arrow1 = Arrow(start=p0, end=p2, color="purple").shift(0.1*DOWN)
arrow2 = Arrow(start=p1, end=p0, color="purple").shift(0.1*RIGHT)
arrow3 = Arrow(start=l0, end=l2, color="purple").shift(0.1*LEFT)
arrow4 = Arrow(start=l0, end=p2, color="purple").shift(0.1*UP)

table.add(arrow1, arrow2, arrow3, arrow4)

# Ajouter des labels aux flèches
table.add(Label(arrow1, 0.5 * DOWN, "\Delta z", color="purple", fontsize="small"),
          Label(arrow2, 0.5 * RIGHT, "\Delta x_o", color="purple", fontsize="small"),
          Label(arrow3, 0.5 * LEFT, "d", color="purple", fontsize="small"),
          Label(arrow4, 0.5 * UP, "L", color="purple", fontsize="small"),
          )




# Masquer les graduations des axes
table.hide_ticks()

# Afficher le schéma
table.show()
