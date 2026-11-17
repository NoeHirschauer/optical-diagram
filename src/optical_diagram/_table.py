from collections.abc import Iterable
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np

from ._base import OpticalElement


class OpticalTable:
    """The main container for the optical setup.

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
        size: tuple[float, float] = (10, 10),
        units: Literal["inches", "cm"] = "cm",
        title: str = "Optical Setup",
        mode: Literal["axial", "top-down"] = "top-down",
        scale_factor: float = 1,
        dpi: int = 100,
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

        # Set limits to center the origin
        self.ax.set_xlim(-self.w / 2, self.w / 2)
        self.ax.set_ylim(-self.h / 2, self.h / 2)
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

    def add(self, *elements: Iterable[OpticalElement]) -> Self:
        """Add one or multiple `OpticalElements` to the table.

        Parameters
        ----------
        elements : Iterable of OpticalElement
            The sequence of optical elements to add to the table.
        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        for el in elements:
            if isinstance(el, OpticalElement):
                self.elements.append(el)
            else:
                raise TypeError(f"Expected OpticalElement, got {type(el)}")
        return self

    def show_grid_lines(
        self, visible: bool = True, alpha: float = 0.2, step: float = 1.0
    ) -> Self:
        """Toggle the background grid.

        Parameters
        ----------
        visible : bool
            Whether to show the grid lines (default: True).
        alpha : float
            Transparency of the grid lines (default: 0.2).
        step : float
            Spacing between grid lines (default: 1.0).
        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        if visible:
            self.ax.grid(visible, alpha=alpha)
            self.ax.set_xticks(np.arange(-self.w / 2, self.w / 2 + 1, step))
            self.ax.set_yticks(np.arange(-self.h / 2, self.h / 2 + 1, step))
        return self

    def show_grid_dots(
        self,
        visible: bool = True,
        alpha: float = 0.5,
        step: float = 1.0,
        size: float = 1.0,
        **kwargs,
    ) -> Self:
        """Toggle the background dots at the grid intersections.

        Parameters
        ----------
        visible : bool
            Whether to show the dots (default: True).
        alpha : float
            Transparency of the dots (default: 0.5).
        step : float
            Spacing between dots (default: 1.0).
        size : float
            Size of the dots (default: 1.0).
        **kwargs : dict, optional
            Additional styling options passed to `matplotlib.pyplot.scatter`.
        Returns
        -------
        Self
            The instance itself (for chaining).
        """
        default_kwargs = {"color": "gray", "marker": ".", "linewidths": 0, "zorder": -10}
        kwargs = {**default_kwargs, **kwargs}

        if visible:
            self.ax.scatter(
                *np.meshgrid(
                    np.arange(-self.w / 2, self.w / 2 + 1, step),
                    np.arange(-self.h / 2, self.h / 2 + 1, step),
                ),
                alpha=alpha,
                s=size,
                **kwargs,
            )
        return self

    def hide_ticks(self) -> Self:
        """Remove tick numbers and tick marks from both axes.

        This keeps the grid visible (if enabled) but removes the
        numerical labels and the small marks on the spine.

        Returns
        -------
        Self
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

    def hide_axis_labels(self) -> Self:
        """Removes the axis labels (X, Y)

        Returns
        -------
        Self
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

        self.fig.tight_layout()
        # force the limits to crop to the specified size, even if elements are beyond it
        self.ax.set_xlim(-self.w / 2, self.w / 2)
        self.ax.set_ylim(-self.h / 2, self.h / 2)
        plt.show()

    def save(self, filename, dpi=400):
        """Render and save the plot."""
        self.render()
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
