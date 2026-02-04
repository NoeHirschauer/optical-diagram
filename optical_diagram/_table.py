from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from ._base import OpticalElement
from ._beams import RayTracedBeam


class OpticalTable:
    """
    The main container for the optical setup.

    Parameters
    ----------
    size : tuple
        (width, height) of the plotting area.
    units: {'inches', 'cm'}
        Units to use for the grid
    title : str, optional
        Title of the plot.
    mode : {'axial', 'top_down'}, optional
        Defines the labeling convention.
        'axial': x-axis is Z (propagation), y-axis is Y.
        'top_down': x-axis is X, y-axis is Y. (default)
    scale_factor : float
        Scaling for the figure size (defaults to 1 unit = 1 inch, roughly).
    dpi : int
        DPI of the output figure. Defaults to 100.
    """

    def __init__(
        self,
        size=(10, 10),
        units: Literal["inches", "cm"] = "cm",
        title="Optical Setup",
        mode: Literal["axial", "top-down"] = "top-down",
        scale_factor=1,
        dpi=100,
    ):
        self.w, self.h = size
        self.mode = mode
        self.elements = []

        # Init figure
        units_scaling = 1 / 2.54 if units == "cm" else 1

        self.fig, self.ax = plt.subplots(
            figsize=(
                self.w * scale_factor * units_scaling,
                self.h * scale_factor * units_scaling,
            ),
            dpi=dpi,
        )
        self.ax.set_title(title)

        # Set limits
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_aspect("equal")

        # Config axes labels
        if mode == "axial":
            self.ax.set_xlabel("Z (Propagation)")
            self.ax.set_ylabel("Y (Height)")
        elif mode == "top-down":
            self.ax.set_xlabel(f"X [{units}]")
            self.ax.set_ylabel(f"Y [{units}]")
        else:
            raise ValueError("``mode`` should be one of 'axial' or 'top-down'")

    def add(self, *elements):
        """Add one or multiple OpticalElements to the table."""
        for el in elements:
            if isinstance(el, OpticalElement):
                self.elements.append(el)
            else:
                raise TypeError(f"Expected OpticalElement, got {type(el)}")
        return self
    
    def auto_scale(self):
        """Resize all elements on the table to fit raytraced beams."""
        # Find all RayTracedBeams
        beams: list[RayTracedBeam]
        beams = [el for el in self.elements if isinstance(el, RayTracedBeam)]
        if not beams:
            return self  # nothing to do

        for beam in beams:
            for el in beam.elements:
                beam_position = beam.get_intersection_with(el)  # ensure intersections are computed
                top = el.center[1] + el.size / 2
                bot = el.center[1] - el.size / 2
                if beam_position[1] > top:
                    el.size = 2*(beam_position[1]+el.center[1]) * 1.5
                if beam_position[1] < bot:
                    el.size = 2*(el.center[1]-beam_position[1]) * 1.5
                    
        return self

    def show_grid(self, visible=True, alpha=0.2):
        """Toggle the background grid.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        if visible:
            self.ax.grid(visible, alpha=alpha)
            self.ax.set_xticks(np.arange(0, self.w + 1, 1))
            self.ax.set_yticks(np.arange(0, self.h + 1, 1))
        return self

    def hide_ticks(self):
        """
        Remove tick numbers and tick marks from both axes.

        This keeps the grid visible (if enabled) but removes the
        numerical labels and the small marks on the spine.

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.ax.tick_params(
            axis="both",  # Apply to both x and y
            which="both",  # Apply to both major and minor ticks
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the left edge are off
            right=False,  # ticks along the right edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False,  # labels along the left edge are off
        )
        return self

    def hide_axis_labels(self):
        """
        Removes the axis labels (X, Y)

        Returns
        -------
        self
            The instance itself (for chaining).
        """
        self.ax.set_xlabel(None)
        self.ax.set_ylabel(None)
        return self

    def render(self):
        """Draws all elements to the axes. Called automatically by show()."""
        # Clear existing patches to avoid duplicates if called multiple times
        try:
            self.ax.patches.clear()
        except AttributeError:
            pass

        for el in self.elements:
            el.draw(self.ax)

    def show(self):
        """Render and display the plot."""
        self.render()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=400):
        """Render and save the plot."""
        self.render()
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
