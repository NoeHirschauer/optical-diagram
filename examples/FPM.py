from optical_diagram import (
    OpticalTable,
    ConvergingLens,
    PlaneMirror,
    Point,
    RayTracedBeam,
    Plane,
    Label,
    UP, DOWN, LEFT, RIGHT,
)

# Créer une table optique
table = OpticalTable(size=(10, 6), title="Système Optique Simple", dpi=150)
table.show_grid(visible=True, alpha=0.3)

# Définir la position de départ
start_position = (1, 3)

# Créer un plan matrice de leds
leds_plane = Plane(start_position, size=2, color="gray")
table.add(leds_plane)

# Créer des points sources
l0 = Point(start_position, size=0.05, color="black")
l1 = Point(start_position, size=0.05, color="black").shift(1 * UP)
l2 = Point(start_position, size=0.05, color="black").shift(1 * DOWN)
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
p0 = Point(focal_plane_center, size=0.05, color="red")
p2 = Point(focal_plane_center+defocus*LEFT, size=0.05, color="blue")

# Créer un point source virtuel sur le plan objet

p1 = Point(focal_plane_center, size=0.05, color="blue", alpha=0.5).shift(0.4 * UP)
table.add(p0, p1, p2)



# Créer des faisceaux de rayons
beam0 = RayTracedBeam((l0, p0), initial_width=0.03, divergence=0, color=p0.color)
beam1 = RayTracedBeam((l2, p1), initial_width=0.03, divergence=0, color=p1.color)
table.add(beam0, beam1)




# Créer une lentille convergente
lens2 = ConvergingLens(start_position, size=2, focal_length=1).shift((6, 0))
table.add(lens2)


# Créer un plan image
image_plane = Plane(lens2.center + lens2.focal_length * RIGHT, size=2, color="black")
table.add(image_plane)

# Créer des faisceaux de rayons
beam2 = RayTracedBeam((p0, lens1, lens2, image_plane), initial_width=0.03, divergence=0, color=p0.color)
beam3 = RayTracedBeam((p1, lens1, lens2, image_plane), initial_width=0.03, divergence=0, color=p1.color)
table.add(beam2, beam3)

table.auto_scale()

# Ajouter des étiquettes
table.add(
    Label(leds_plane, (0, -1), "Plan de LEDs", fontsize="small"),
    Label(obj_plane, (0, -1), "Plan Objet", fontsize="small"),
    Label(lens1, (0, -1), "$L_1$", fontsize="small"),
    Label(lens2, (0, -1), "$L_2$", fontsize="small"),
    Label(image_plane, (0, -1), "Plan Image", fontsize="small"),
)

# Ajouter des flèches annotées
arrow1 = FancyArrowPatch(
    posA=leds_plane.center,
    posB=lens1.center,
    arrowstyle="->",
    color="green",
    mutation_scale=15,
    lw=1.5,
)
table.ax.add_patch(arrow1)

# Ajouter une annotation textuelle à la flèche
table.ax.annotate(
    "Direction de la lumière",
    xy=(lens1.center[0], lens1.center[1]),
    xytext=(leds_plane.center[0] + 0.5, leds_plane.center[1] + 0.5),
    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    fontsize="small",
    color="green",
)

# Masquer les graduations des axes
table.hide_ticks()

# Afficher le schéma
table.show()
