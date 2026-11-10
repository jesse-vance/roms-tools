"""Interactive editor for ROMS grid land mask and bathymetry.

This module provides a matplotlib-based GUI for editing a ROMS grid,
equivalent to the MATLAB GridBuilder toolbox.

Usage — from a Jupyter notebook
--------------------------------
Install ipympl for interactive matplotlib in Jupyter::

    pip install ipympl

Then::

    %matplotlib widget

    from roms_tools import Grid
    from roms_tools.setup.grid_editor import GridEditor

    grid = Grid(filename="my_grid.nc")
    editor = GridEditor(grid)
    editor.show()

    # After editing, retrieve the modified grid:
    edited_grid = editor.get_grid()
    edited_grid.save("my_grid_edited.nc")

Usage — from the terminal
--------------------------
::

    python -m roms_tools.setup.grid_editor my_grid.nc [output_grid.nc]

If ``output_grid.nc`` is provided, the edited grid is saved on window close.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Button, RadioButtons, Slider


class GridEditor:
    """Interactive editor for ROMS grid land mask and bathymetry.

    Parameters
    ----------
    grid : Grid
        A roms-tools :class:`Grid` instance to edit. The original grid
        is not modified; call :meth:`get_grid` to retrieve a new Grid
        with the applied edits.

    Examples
    --------
    >>> from roms_tools import Grid
    >>> from roms_tools.setup.grid_editor import GridEditor
    >>> grid = Grid(filename="my_grid.nc")
    >>> editor = GridEditor(grid)
    >>> editor.show()
    >>> edited = editor.get_grid()
    >>> edited.save("my_grid_edited.nc")
    """

    PAINT_LAND = "Paint Land"
    PAINT_OCEAN = "Paint Ocean"
    TOGGLE = "Toggle"

    def __init__(self, grid) -> None:
        self._grid = grid
        self._lon: np.ndarray = grid.ds["lon_rho"].values
        self._lat: np.ndarray = grid.ds["lat_rho"].values
        self._mask: np.ndarray = grid.ds["mask_rho"].values.copy().astype(float)
        self._h: np.ndarray = grid.ds["h"].values.copy().astype(float)

        self._undo_stack: List[Tuple[np.ndarray, np.ndarray]] = []
        self._mode: str = self.PAINT_LAND

        self._is_dragging: bool = False
        self._drag_start_mask: Optional[np.ndarray] = None
        self._last_cell: Optional[Tuple[int, int]] = None

        self._fig: Optional[plt.Figure] = None
        self._ax_map: Optional[plt.Axes] = None
        self._pcm_bathy = None
        self._pcm_land = None
        self._status_ax = None
        self._status_text = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display the editor window.

        Blocks in a terminal session; embeds interactively in Jupyter
        when ``%matplotlib widget`` is active.
        """
        self._build_ui()
        plt.show()

    def get_grid(self):
        """Return a new :class:`Grid` with the edited mask and bathymetry.

        Returns
        -------
        Grid
            A copy of the original grid with updated ``mask_rho`` and ``h``
            variables reflecting all edits made in the editor.
        """
        new_ds = self._grid.ds.copy(deep=True)
        new_ds["mask_rho"] = xr.DataArray(
            self._mask,
            dims=self._grid.ds["mask_rho"].dims,
            attrs=self._grid.ds["mask_rho"].attrs,
        )
        new_ds["h"] = xr.DataArray(
            self._h,
            dims=self._grid.ds["h"].dims,
            attrs=self._grid.ds["h"].attrs,
        )
        return self._grid.copy_with_ds(new_ds)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._fig = plt.figure(figsize=(15, 8))
        self._fig.canvas.manager.set_window_title("ROMS Grid Editor")

        # Map axes: takes up the right ~78% of the figure
        self._ax_map = self._fig.add_axes([0.21, 0.08, 0.76, 0.87])
        self._draw_map()

        # Status text box (bottom-left)
        self._status_ax = self._fig.add_axes([0.01, 0.01, 0.17, 0.12])
        self._status_ax.set_axis_off()
        self._status_text = self._status_ax.text(
            0.05, 0.95, "Click a cell to inspect",
            transform=self._status_ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            fontfamily="monospace",
        )

        # Header label
        hdr = self._fig.add_axes([0.01, 0.91, 0.17, 0.07])
        hdr.set_axis_off()
        hdr.text(0.5, 0.5, "ROMS Grid Editor",
                 ha="center", va="center", fontsize=10, fontweight="bold")

        # Tool selection radio buttons
        ax_radio = self._fig.add_axes([0.01, 0.73, 0.17, 0.16])
        self._radio = RadioButtons(
            ax_radio,
            [self.PAINT_LAND, self.PAINT_OCEAN, self.TOGGLE],
            active=0,
        )
        self._radio.on_clicked(self._on_tool_change)

        # Minimum depth slider + button
        ax_hdr_hmin = self._fig.add_axes([0.01, 0.69, 0.17, 0.03])
        ax_hdr_hmin.set_axis_off()
        ax_hdr_hmin.text(0.05, 0.5, "Min depth (m):", va="center", fontsize=8)

        ax_sl = self._fig.add_axes([0.02, 0.63, 0.14, 0.04])
        self._hmin_slider = Slider(ax_sl, "", 0, 50, valinit=0, valstep=0.5)

        ax_hmin_btn = self._fig.add_axes([0.02, 0.57, 0.14, 0.05])
        self._hmin_btn = Button(ax_hmin_btn, "Apply Hmin")
        self._hmin_btn.on_clicked(self._apply_hmin)

        # Mask quality buttons
        ax_detect = self._fig.add_axes([0.02, 0.50, 0.14, 0.05])
        self._detect_btn = Button(ax_detect, "Find Issues")
        self._detect_btn.on_clicked(self._detect_issues)

        ax_fix = self._fig.add_axes([0.02, 0.43, 0.14, 0.05])
        self._fix_btn = Button(ax_fix, "Fix Isolated")
        self._fix_btn.on_clicked(self._fix_isolated)

        # Undo / Reset
        ax_undo = self._fig.add_axes([0.02, 0.36, 0.14, 0.05])
        self._undo_btn = Button(ax_undo, "Undo")
        self._undo_btn.on_clicked(self._undo)

        ax_reset = self._fig.add_axes([0.02, 0.29, 0.14, 0.05])
        self._reset_btn = Button(ax_reset, "Reset All")
        self._reset_btn.on_clicked(self._reset)

        # Wire events
        self._cid_press = self._fig.canvas.mpl_connect(
            "button_press_event", self._on_press
        )
        self._cid_motion = self._fig.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )
        self._cid_release = self._fig.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

    def _draw_map(self) -> None:
        ax = self._ax_map
        ax.clear()

        h_ocean = np.where(self._mask == 1, self._h, np.nan)
        land_flag = np.where(self._mask == 0, 1.0, np.nan)

        vmax = float(np.nanmax(self._h)) if np.any(np.isfinite(self._h)) else 100.0

        self._pcm_bathy = ax.pcolormesh(
            self._lon, self._lat, h_ocean,
            cmap="Blues", vmin=0, vmax=vmax,
            shading="nearest",
        )
        self._pcm_land = ax.pcolormesh(
            self._lon, self._lat, land_flag,
            cmap="Greys_r", vmin=0, vmax=1,
            shading="nearest", alpha=0.7,
        )

        if not hasattr(self, "_colorbar"):
            self._colorbar = self._fig.colorbar(
                self._pcm_bathy, ax=ax, label="Depth (m)", fraction=0.03, pad=0.02,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            "Blue = ocean  |  Grey = land\n"
            "Left-click / drag to paint  •  Scroll to zoom",
            fontsize=9,
        )

    def _update_display(self) -> None:
        if self._pcm_bathy is None:
            return
        h_ocean = np.where(self._mask == 1, self._h, np.nan)
        land_flag = np.where(self._mask == 0, 1.0, np.nan)
        self._pcm_bathy.set_array(h_ocean)
        self._pcm_land.set_array(land_flag)
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_tool_change(self, label: str) -> None:
        self._mode = label

    def _on_press(self, event) -> None:
        if event.inaxes != self._ax_map or event.button != 1:
            return
        self._drag_start_mask = self._mask.copy()
        self._is_dragging = True
        eta, xi = self._find_cell(event.xdata, event.ydata)
        self._last_cell = (eta, xi)
        self._apply_tool_at(eta, xi)
        self._update_display()
        self._update_status(eta, xi)

    def _on_motion(self, event) -> None:
        if not self._is_dragging or event.inaxes != self._ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        eta, xi = self._find_cell(event.xdata, event.ydata)
        if (eta, xi) != self._last_cell:
            self._last_cell = (eta, xi)
            self._apply_tool_at(eta, xi)
            self._update_display()
            self._update_status(eta, xi)

    def _on_release(self, event) -> None:
        if self._is_dragging and self._drag_start_mask is not None:
            if not np.array_equal(self._mask, self._drag_start_mask):
                self._undo_stack.append((self._drag_start_mask, self._h.copy()))
                if len(self._undo_stack) > 50:
                    self._undo_stack.pop(0)
        self._is_dragging = False
        self._drag_start_mask = None
        self._last_cell = None

    # ------------------------------------------------------------------
    # Tool application
    # ------------------------------------------------------------------

    def _apply_tool_at(self, eta: int, xi: int) -> None:
        if self._mode == self.PAINT_LAND:
            self._mask[eta, xi] = 0.0
        elif self._mode == self.PAINT_OCEAN:
            self._mask[eta, xi] = 1.0
        elif self._mode == self.TOGGLE:
            self._mask[eta, xi] = 1.0 - self._mask[eta, xi]

    def _find_cell(self, lon: float, lat: float) -> Tuple[int, int]:
        dist2 = (self._lon - lon) ** 2 + (self._lat - lat) ** 2
        eta, xi = np.unravel_index(int(dist2.argmin()), dist2.shape)
        return int(eta), int(xi)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _apply_hmin(self, event=None) -> None:
        hmin = float(self._hmin_slider.val)
        if hmin <= 0:
            self._set_status("Set hmin > 0 first.")
            return
        self._push_undo()
        shallow = (self._mask == 1) & (self._h < hmin)
        n = int(shallow.sum())
        self._mask[shallow] = 0.0
        self._update_display()
        self._set_status(f"Masked {n} wet cells\nshallower than {hmin:.1f} m.")

    def _detect_issues(self, event=None) -> None:
        scores = self._neighbor_scores()
        iso = int(((scores == 15) & (self._mask == 1)).sum())
        bays = int((np.isin(scores, [7, 11, 13, 14]) & (self._mask == 1)).sum())
        chans = int((np.isin(scores, [5, 10]) & (self._mask == 1)).sum())
        self._set_status(
            f"Isolated cells:  {iso}\n"
            f"Isolated bays:   {bays}\n"
            f"Narrow channels: {chans}\n\n"
            f"Use 'Fix Isolated'\nto auto-mask\nisolated cells."
        )

    def _fix_isolated(self, event=None) -> None:
        self._push_undo()
        scores = self._neighbor_scores()
        iso = (scores == 15) & (self._mask == 1)
        n = int(iso.sum())
        self._mask[iso] = 0.0
        self._update_display()
        self._set_status(f"Masked {n} isolated\nocean cells.")

    def _undo(self, event=None) -> None:
        if not self._undo_stack:
            self._set_status("Nothing to undo.")
            return
        self._mask, self._h = self._undo_stack.pop()
        self._update_display()
        self._set_status("Undo applied.")

    def _reset(self, event=None) -> None:
        self._push_undo()
        self._mask = self._grid.ds["mask_rho"].values.copy().astype(float)
        self._h = self._grid.ds["h"].values.copy().astype(float)
        self._update_display()
        self._set_status("Reset to original\ngrid.")

    # ------------------------------------------------------------------
    # Mask quality scoring
    # ------------------------------------------------------------------

    def _neighbor_scores(self) -> np.ndarray:
        """Return per-cell neighbor-land score (0–15) matching GridBuilder.

        Each bit encodes whether a cardinal neighbor is land:
        bit 3 = North (8), bit 2 = South (4), bit 1 = East (2), bit 0 = West (1).
        """
        land = (self._mask == 0).astype(int)
        N = np.zeros_like(land)
        S = np.zeros_like(land)
        E = np.zeros_like(land)
        W = np.zeros_like(land)
        N[:-1, :] = land[1:, :]
        S[1:, :] = land[:-1, :]
        E[:, :-1] = land[:, 1:]
        W[:, 1:] = land[:, :-1]
        return 8 * N + 4 * S + 2 * E + W

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo_stack.append((self._mask.copy(), self._h.copy()))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _set_status(self, msg: str) -> None:
        if self._status_text is not None and self._fig is not None:
            self._status_text.set_text(msg)
            self._fig.canvas.draw_idle()

    def _update_status(self, eta: int, xi: int) -> None:
        lon = float(self._lon[eta, xi])
        lat = float(self._lat[eta, xi])
        h = float(self._h[eta, xi])
        mval = "ocean" if self._mask[eta, xi] == 1 else "land"
        self._set_status(
            f"({eta}, {xi})\n"
            f"lon  {lon:8.3f}\n"
            f"lat  {lat:8.3f}\n"
            f"h    {h:8.1f} m\n"
            f"mask  {mval}"
        )


def _cli_main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m roms_tools.setup.grid_editor <grid.nc> [output.nc]")
        sys.exit(1)

    from roms_tools.setup.grid import Grid

    grid_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading grid from {grid_path} ...")
    grid = Grid(filename=grid_path)

    editor = GridEditor(grid)
    print("Launching editor — close the window when finished.")
    editor.show()

    if out_path:
        edited = editor.get_grid()
        edited.save(out_path)
        print(f"Saved edited grid to {out_path}")


if __name__ == "__main__":
    _cli_main()
