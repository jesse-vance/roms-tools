"""Interactive editor for ROMS grid land mask and bathymetry.

This module provides a matplotlib-based GUI for editing a ROMS grid,
equivalent to the MATLAB GridBuilder toolbox.

Features
--------
- Paint land/ocean cells by clicking or dragging
- Toggle individual cells between land and ocean
- Apply a minimum-depth threshold (hmin)
- Smooth bathymetry to meet an rx0 target (Beckmann–Haidvogel stiffness)
- Overlay rx0 and orthogonality metrics on the map
- Detect and auto-fix isolated ocean cells, isolated bays, narrow channels
- 50-level undo stack

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

    # Tool modes
    PAINT_LAND = "Paint Land"
    PAINT_OCEAN = "Paint Ocean"
    TOGGLE = "Toggle"

    # Display modes
    DISP_BATHY = "Bathymetry"
    DISP_RX0 = "rx0"
    DISP_ORTHOG = "Orthogonality"

    def __init__(self, grid) -> None:
        self._grid = grid
        self._lon: np.ndarray = grid.ds["lon_rho"].values
        self._lat: np.ndarray = grid.ds["lat_rho"].values
        self._mask: np.ndarray = grid.ds["mask_rho"].values.copy().astype(float)
        self._h: np.ndarray = grid.ds["h"].values.copy().astype(float)

        self._undo_stack: List[Tuple[np.ndarray, np.ndarray]] = []
        self._mode: str = self.PAINT_LAND
        self._disp_mode: str = self.DISP_BATHY

        self._is_dragging: bool = False
        self._drag_start_mask: Optional[np.ndarray] = None
        self._last_cell: Optional[Tuple[int, int]] = None

        self._fig: Optional[plt.Figure] = None
        self._ax_map: Optional[plt.Axes] = None
        self._pcm_bathy = None
        self._pcm_land = None
        self._pcm_metric = None
        self._colorbar = None
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
        self._fig = plt.figure(figsize=(16, 9))
        self._fig.canvas.manager.set_window_title("ROMS Grid Editor")

        self._ax_map = self._fig.add_axes([0.21, 0.08, 0.76, 0.87])
        self._draw_map()

        # Status text
        ax_st = self._fig.add_axes([0.01, 0.01, 0.17, 0.12])
        ax_st.set_axis_off()
        self._status_text = ax_st.text(
            0.05, 0.95, "Click a cell to inspect",
            transform=ax_st.transAxes,
            fontsize=7.5, va="top", ha="left", fontfamily="monospace",
        )

        # Header
        hdr = self._fig.add_axes([0.01, 0.92, 0.17, 0.06])
        hdr.set_axis_off()
        hdr.text(0.5, 0.5, "ROMS Grid Editor",
                 ha="center", va="center", fontsize=10, fontweight="bold")

        # --- Tool selection ---
        ax_tl = self._fig.add_axes([0.01, 0.84, 0.17, 0.02])
        ax_tl.set_axis_off()
        ax_tl.text(0.05, 0.5, "Edit tool:", va="center", fontsize=8, fontweight="bold")

        ax_radio = self._fig.add_axes([0.01, 0.73, 0.17, 0.10])
        self._radio_tool = RadioButtons(
            ax_radio, [self.PAINT_LAND, self.PAINT_OCEAN, self.TOGGLE], active=0,
        )
        self._radio_tool.on_clicked(self._on_tool_change)

        # --- Display mode ---
        ax_dl = self._fig.add_axes([0.01, 0.70, 0.17, 0.02])
        ax_dl.set_axis_off()
        ax_dl.text(0.05, 0.5, "Display:", va="center", fontsize=8, fontweight="bold")

        ax_disp = self._fig.add_axes([0.01, 0.59, 0.17, 0.10])
        self._radio_disp = RadioButtons(
            ax_disp, [self.DISP_BATHY, self.DISP_RX0, self.DISP_ORTHOG], active=0,
        )
        self._radio_disp.on_clicked(self._on_disp_change)

        # --- Min depth ---
        ax_hmin_lbl = self._fig.add_axes([0.01, 0.56, 0.17, 0.02])
        ax_hmin_lbl.set_axis_off()
        ax_hmin_lbl.text(0.05, 0.5, "Min depth (m):", va="center", fontsize=8, fontweight="bold")

        ax_sl = self._fig.add_axes([0.02, 0.51, 0.14, 0.03])
        self._hmin_slider = Slider(ax_sl, "", 0, 50, valinit=0, valstep=0.5)

        ax_hmin_btn = self._fig.add_axes([0.02, 0.45, 0.14, 0.05])
        self._hmin_btn = Button(ax_hmin_btn, "Apply Hmin")
        self._hmin_btn.on_clicked(self._apply_hmin)

        # --- rx0 smoothing ---
        ax_rx0_lbl = self._fig.add_axes([0.01, 0.42, 0.17, 0.02])
        ax_rx0_lbl.set_axis_off()
        ax_rx0_lbl.text(0.05, 0.5, "Smooth bathy:", va="center", fontsize=8, fontweight="bold")

        ax_sl2 = self._fig.add_axes([0.02, 0.38, 0.14, 0.03])
        self._rx0_slider = Slider(ax_sl2, "rx0 ≤", 0.05, 0.50, valinit=0.20, valstep=0.01)

        ax_rx0_btn = self._fig.add_axes([0.02, 0.32, 0.14, 0.05])
        self._smooth_rx0_btn = Button(ax_rx0_btn, "rx0 Smooth")
        self._smooth_rx0_btn.on_clicked(self._smooth_rx0)

        ax_shap_btn = self._fig.add_axes([0.02, 0.26, 0.14, 0.05])
        self._smooth_shap_btn = Button(ax_shap_btn, "Shapiro Smooth")
        self._smooth_shap_btn.on_clicked(self._smooth_shapiro)

        # --- Mask quality ---
        ax_detect = self._fig.add_axes([0.02, 0.19, 0.14, 0.05])
        self._detect_btn = Button(ax_detect, "Find Issues")
        self._detect_btn.on_clicked(self._detect_issues)

        ax_fix = self._fig.add_axes([0.02, 0.13, 0.14, 0.05])
        self._fix_btn = Button(ax_fix, "Fix Isolated")
        self._fix_btn.on_clicked(self._fix_isolated)

        # --- Undo / Reset ---
        ax_undo = self._fig.add_axes([0.02, 0.07, 0.065, 0.040])
        self._undo_btn = Button(ax_undo, "Undo")
        self._undo_btn.on_clicked(self._undo)

        ax_reset = self._fig.add_axes([0.095, 0.07, 0.065, 0.040])
        self._reset_btn = Button(ax_reset, "Reset")
        self._reset_btn.on_clicked(self._reset)

        # Wire events
        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _draw_map(self) -> None:
        ax = self._ax_map
        ax.clear()

        if self._disp_mode == self.DISP_BATHY:
            self._draw_bathymetry(ax)
        elif self._disp_mode == self.DISP_RX0:
            self._draw_rx0(ax)
        elif self._disp_mode == self.DISP_ORTHOG:
            self._draw_orthogonality(ax)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def _draw_bathymetry(self, ax) -> None:
        h_ocean = np.where(self._mask == 1, self._h, np.nan)
        land_flag = np.where(self._mask == 0, 1.0, np.nan)
        vmax = float(np.nanmax(self._h)) if np.any(np.isfinite(self._h)) else 100.0

        self._pcm_bathy = ax.pcolormesh(
            self._lon, self._lat, h_ocean,
            cmap="Blues", vmin=0, vmax=vmax, shading="nearest",
        )
        self._pcm_land = ax.pcolormesh(
            self._lon, self._lat, land_flag,
            cmap="Greys_r", vmin=0, vmax=1, shading="nearest", alpha=0.7,
        )
        self._pcm_metric = None

        self._refresh_colorbar(self._pcm_bathy, "Depth (m)")
        ax.set_title("Bathymetry  |  Blue = ocean, Grey = land", fontsize=9)

    def _draw_rx0(self, ax) -> None:
        rx0 = self._compute_rx0()
        rx0_disp = np.where(self._mask == 1, rx0, np.nan)

        self._pcm_bathy = None
        self._pcm_land = None
        self._pcm_metric = ax.pcolormesh(
            self._lon, self._lat, rx0_disp,
            cmap="RdYlGn_r", vmin=0, vmax=0.5, shading="nearest",
        )
        # Land overlay
        land_flag = np.where(self._mask == 0, 1.0, np.nan)
        ax.pcolormesh(
            self._lon, self._lat, land_flag,
            cmap="Greys_r", vmin=0, vmax=1, shading="nearest", alpha=0.5,
        )

        self._refresh_colorbar(self._pcm_metric, "rx0 (Beckmann-Haidvogel)")
        rmax = float(np.nanmax(rx0_disp)) if np.any(np.isfinite(rx0_disp)) else 0.0
        rating = "✓ good" if rmax < 0.2 else ("~ marginal" if rmax < 0.4 else "✗ poor")
        ax.set_title(
            f"Bathymetry stiffness rx0  |  max = {rmax:.3f}  {rating}\n"
            "Green < 0.2 (good)  •  Yellow < 0.4 (marginal)  •  Red > 0.4 (poor)",
            fontsize=9,
        )

    def _draw_orthogonality(self, ax) -> None:
        orthog = self._compute_orthogonality()
        self._pcm_bathy = None
        self._pcm_land = None
        self._pcm_metric = ax.pcolormesh(
            self._lon, self._lat, orthog,
            cmap="YlOrRd", vmin=0, vmax=10, shading="nearest",
        )
        land_flag = np.where(self._mask == 0, 1.0, np.nan)
        ax.pcolormesh(
            self._lon, self._lat, land_flag,
            cmap="Greys_r", vmin=0, vmax=1, shading="nearest", alpha=0.5,
        )
        self._refresh_colorbar(self._pcm_metric, "Orthogonality error (°)")
        omax = float(np.nanmax(orthog)) if np.any(np.isfinite(orthog)) else 0.0
        ax.set_title(
            f"Grid orthogonality error  |  max = {omax:.2f}°\n"
            "< 1° (perfect)  •  < 5° (good)  •  > 10° (poor)",
            fontsize=9,
        )

    def _refresh_colorbar(self, mappable, label: str) -> None:
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
        self._colorbar = self._fig.colorbar(
            mappable, ax=self._ax_map, label=label, fraction=0.03, pad=0.02,
        )

    def _update_display(self) -> None:
        if self._fig is None:
            return

        if self._disp_mode == self.DISP_BATHY and self._pcm_bathy is not None:
            h_ocean = np.where(self._mask == 1, self._h, np.nan)
            land_flag = np.where(self._mask == 0, 1.0, np.nan)
            self._pcm_bathy.set_array(h_ocean)
            self._pcm_land.set_array(land_flag)
            self._fig.canvas.draw_idle()
        else:
            # Full redraw for metric modes (or after metric state changes)
            self._draw_map()
            self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_tool_change(self, label: str) -> None:
        self._mode = label

    def _on_disp_change(self, label: str) -> None:
        self._disp_mode = label
        self._draw_map()
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self._ax_map or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
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

    def _smooth_rx0(self, event=None) -> None:
        """Smooth bathymetry using the positive adjustment algorithm.

        Raises shallow cells until the rx0 stiffness parameter at all
        wet cell faces is at or below the target value.  This is the
        same algorithm as GridBuilder's "Positive Adjustment" filter
        (Sikiric LP_Bathymetry Toolbox).
        """
        target = float(self._rx0_slider.val)
        self._push_undo()
        self._h = _smooth_positive_rx0(self._h, self._mask, target)
        self._update_display()
        rx0 = self._compute_rx0()
        rmax = float(np.nanmax(np.where(self._mask == 1, rx0, np.nan)))
        self._set_status(
            f"rx0 smooth done.\nTarget: {target:.2f}\nNew max: {rmax:.3f}"
        )

    def _smooth_shapiro(self, event=None) -> None:
        """Apply a Shapiro (order-2) filter to smooth bathymetry.

        Does not modify masked (land) cells.  This is a linear spatial
        average that reduces high-frequency noise without altering the
        overall depth structure.
        """
        self._push_undo()
        self._h = _smooth_shapiro(self._h, self._mask, order=2)
        self._update_display()
        rx0 = self._compute_rx0()
        rmax = float(np.nanmax(np.where(self._mask == 1, rx0, np.nan)))
        self._set_status(f"Shapiro smooth done.\nNew max rx0: {rmax:.3f}")

    def _detect_issues(self, event=None) -> None:
        scores = self._neighbor_scores()
        iso = int(((scores == 15) & (self._mask == 1)).sum())
        bays = int((np.isin(scores, [7, 11, 13, 14]) & (self._mask == 1)).sum())
        chans = int((np.isin(scores, [5, 10]) & (self._mask == 1)).sum())
        rx0 = self._compute_rx0()
        rmax = float(np.nanmax(np.where(self._mask == 1, rx0, np.nan)))
        self._set_status(
            f"Isolated cells:  {iso}\n"
            f"Isolated bays:   {bays}\n"
            f"Narrow channels: {chans}\n"
            f"Max rx0:         {rmax:.3f}\n"
            f"{'(> 0.4 — POOR)' if rmax > 0.4 else ('(> 0.2 — marginal)' if rmax > 0.2 else '(< 0.2 — good)')}"
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
    # Grid metrics
    # ------------------------------------------------------------------

    def _compute_rx0(self) -> np.ndarray:
        """Compute Beckmann–Haidvogel (1993) bathymetry stiffness rx0.

        Returns a (eta, xi) array where each value is the maximum rx0
        across the four cell faces, masked to zero at land points.

        Equivalent to GridBuilder's ``getRx0`` function.
        """
        return _compute_rx0(self._h, self._mask)

    def _compute_orthogonality(self) -> np.ndarray:
        """Compute maximum deviation from 90° at each rho cell.

        Uses the rho-point lon/lat of this grid.  Returns orthogonality
        error in degrees.  Equivalent to GridBuilder's ``getOrthog``.
        """
        return _compute_orthogonality(self._lon, self._lat)

    def _neighbor_scores(self) -> np.ndarray:
        """Return per-cell neighbor-land score (0–15) matching GridBuilder.

        Bit 3 = North (8), bit 2 = South (4), bit 1 = East (2), bit 0 = West (1).
        Each bit is 1 when that cardinal neighbor is land (mask_rho == 0).
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
        rx0_val = float(_compute_rx0(self._h, self._mask)[eta, xi])
        self._set_status(
            f"({eta}, {xi})\n"
            f"lon  {lon:8.3f}\n"
            f"lat  {lat:8.3f}\n"
            f"h    {h:8.1f} m\n"
            f"mask  {mval}\n"
            f"rx0  {rx0_val:.3f}"
        )


# ---------------------------------------------------------------------------
# Module-level metric functions (usable independently of the editor)
# ---------------------------------------------------------------------------


def _compute_rx0(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute Beckmann–Haidvogel (1993) bathymetry stiffness parameter rx0.

    Parameters
    ----------
    h : ndarray, shape (eta, xi)
        Bathymetry at rho-points (positive down).
    mask : ndarray, shape (eta, xi)
        Land/sea mask (1 = ocean, 0 = land).

    Returns
    -------
    rx0 : ndarray, shape (eta, xi)
        Cell-centred rx0 value: the maximum stiffness across all four
        cell faces.  Land cells are set to 0.
    """
    Lp, Mp = h.shape
    L, M = Lp - 1, Mp - 1

    umask = mask[:L, :] * mask[1:, :]   # shape (L, Mp)
    vmask = mask[:, :M] * mask[:, 1:]   # shape (Lp, M)

    # Use a large sentinel depth at land cells so faces between ocean and land
    # are dominated by the ocean depth and division stays safe.
    h_safe = np.where(mask == 1, h, np.nan)

    hx = np.zeros((L, Mp))
    hy = np.zeros((Lp, M))

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denom_x = h_safe[1:, :] + h_safe[:L, :]
        hx_raw = np.abs(h_safe[1:, :] - h_safe[:L, :]) / np.where(denom_x > 0, denom_x, np.nan)
        hx[:, :] = np.where(umask == 1, np.nan_to_num(hx_raw, nan=0.0), 0.0)

        denom_y = h_safe[:, 1:] + h_safe[:, :M]
        hy_raw = np.abs(h_safe[:, 1:] - h_safe[:, :M]) / np.where(denom_y > 0, denom_y, np.nan)
        hy[:, :] = np.where(vmask == 1, np.nan_to_num(hy_raw, nan=0.0), 0.0)

    rx0 = np.maximum(
        np.maximum(hx[:L, :M], hx[:L, 1:]),
        np.maximum(hy[:L, :M], hy[1:, :M]),
    )
    # Pad to full (Lp, Mp) shape; border cells get the edge value
    result = np.zeros((Lp, Mp))
    result[:L, :M] = rx0
    result[-1, :M] = hy[-1, :]
    result[:L, -1] = hx[:, -1]
    result[-1, -1] = max(hy[-1, -1], hx[-1, -1]) if L > 0 and M > 0 else 0.0
    return result


def _compute_orthogonality(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute maximum deviation from 90° at rho-cell corners.

    Parameters
    ----------
    lon, lat : ndarray, shape (eta, xi)
        Longitude and latitude of rho-points in degrees.

    Returns
    -------
    orthog : ndarray, shape (eta, xi)
        Orthogonality error in degrees (0 = perfect right angles).
        Computed at each interior cell using the psi-point angles;
        border cells are set to 0.
    """
    # Convert to Mercator projection (conformal — preserves angles)
    lon_r = np.deg2rad(lon)
    lat_m = np.arctanh(np.sin(np.deg2rad(lat)))  # Mercator latitude

    z = lon_r + 1j * lat_m

    eta, xi = z.shape
    # Use the rho-grid as a proxy for psi-grid angle computation
    # (approximation: works well for near-orthogonal grids)
    dzxi = np.zeros_like(z)
    dzeta = np.zeros_like(z)

    # xi-direction tangent at interior points
    dzxi[1:-1, 1:-1] = z[1:-1, 2:] - z[1:-1, :-2]
    # eta-direction tangent at interior points
    dzeta[1:-1, 1:-1] = z[2:, 1:-1] - z[:-2, 1:-1]

    dn = np.abs(dzxi) * np.abs(dzeta)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.where(
            dn > 0,
            (dzxi.real * dzeta.real + dzxi.imag * dzeta.imag) / dn,
            0.0,
        )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    orthog = np.degrees(np.abs(angle - np.pi / 2))
    orthog[0, :] = 0.0
    orthog[-1, :] = 0.0
    orthog[:, 0] = 0.0
    orthog[:, -1] = 0.0
    return orthog


def _smooth_positive_rx0(
    h: np.ndarray,
    mask: np.ndarray,
    rx0_max: float = 0.20,
    tol: float = 1e-6,
) -> np.ndarray:
    """Smooth bathymetry by increasing depths to satisfy the rx0 target.

    Uses the "positive adjustment" algorithm from the LP_Bathymetry Toolbox
    (Sikiric) as implemented in GridBuilder.  For each pair of adjacent wet
    cells whose rx0 exceeds ``rx0_max``, the shallower cell's depth is
    increased until the face satisfies the target.

    This algorithm only *increases* depths — it never makes anything shallower —
    so it cannot create new shallow areas.

    Parameters
    ----------
    h : ndarray, shape (eta, xi)
        Bathymetry at rho-points (positive down, metres).
    mask : ndarray, shape (eta, xi)
        Land/sea mask (1 = ocean, 0 = land).
    rx0_max : float
        Target maximum rx0 value.  Default is 0.2.
    tol : float
        Convergence tolerance.

    Returns
    -------
    h_smooth : ndarray, shape (eta, xi)
        Smoothed bathymetry.
    """
    Lp, Mp = h.shape
    L, M = Lp - 1, Mp - 1
    sfact = (1.0 + rx0_max) / (1.0 - rx0_max)

    hf = h.copy()

    # Pre-build neighbor masks for the 4 cardinal directions
    # Direction 0 = North (eta+1), 1 = East (xi+1), 2 = South (eta-1), 3 = West (xi-1)
    nmask = np.zeros((Lp, Mp, 4))
    nmask[:L, :, 0] = mask[1:, :]
    nmask[:, :M, 1] = mask[:, 1:]
    nmask[1:, :, 2] = mask[:L, :]
    nmask[:, 1:, 3] = mask[:, :M]

    finished = False
    while not finished:
        finished = True
        neighbors = np.zeros((Lp, Mp, 4))
        neighbors[:L, :, 0] = hf[1:, :]
        neighbors[:, :M, 1] = hf[:, 1:]
        neighbors[1:, :, 2] = hf[:L, :]
        neighbors[:, 1:, 3] = hf[:, :M]

        for d in range(4):
            nm = nmask[:, :, d]
            nb_raw = neighbors[:, :, d]
            # Compute sfact * neighbour only where the neighbour is ocean
            # (avoid overflow on the full-array multiply by masking first)
            with np.errstate(over="ignore", invalid="ignore"):
                nb = np.where(nm == 1, nb_raw * sfact, 0.0)
            diff = hf - nb
            # Where our cell is shallower than sfact*neighbour, raise it
            ind = (diff < -tol) & (mask == 1) & (nm == 1) & np.isfinite(nb)
            if ind.any():
                finished = False
                hf[ind] = nb[ind]

    return hf


def _smooth_shapiro(
    h: np.ndarray,
    mask: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """Apply a 2D Shapiro filter to bathymetry, leaving land cells unchanged.

    Parameters
    ----------
    h : ndarray, shape (eta, xi)
        Bathymetry at rho-points (positive down, metres).
    mask : ndarray, shape (eta, xi)
        Land/sea mask (1 = ocean, 0 = land).
    order : int
        Filter order (2, 4, 8, …).  Order 2 is a simple 5-point Laplacian.

    Returns
    -------
    h_smooth : ndarray, shape (eta, xi)
    """
    # Shapiro filter coefficients (from Arango/Rutgers, same as GridBuilder)
    fourk = [
        2.5e-1, 6.25e-2, 1.5625e-2, 3.90625e-3,
        9.765625e-4, 2.44140625e-4, 6.103515625e-5, 1.5258789063e-5,
    ]
    order2 = order // 2
    alpha = fourk[order2 - 1]

    hf = h.copy().astype(float)
    cor = np.zeros_like(hf)

    # Filter rows
    for _ in range(order2):
        cor[1:-1, :] = 2 * hf[1:-1, :] - hf[:-2, :] - hf[2:, :]
        cor[0, :] = 2 * (hf[0, :] - hf[1, :])
        cor[-1, :] = 2 * (hf[-1, :] - hf[-2, :])
    hf = hf - cor * alpha

    # Filter columns
    for _ in range(order2):
        cor[:, 1:-1] = 2 * hf[:, 1:-1] - hf[:, :-2] - hf[:, 2:]
        cor[:, 0] = 2 * (hf[:, 0] - hf[:, 1])
        cor[:, -1] = 2 * (hf[:, -1] - hf[:, -2])
    hf = hf - cor * alpha

    # Restore land cells and ensure no cell goes negative
    hf = np.where(mask == 1, np.maximum(hf, 0.0), h)
    return hf


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
