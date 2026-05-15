"""
curve_correction.py — PanOCT fan-beam curve distortion correction.

Implements Bayhaqi et al. 2025 (Casey Eye Institute, OHSU):
  Eq. 1  — 2D polar → Cartesian:  x = R·sin(θ),  z = R·cos(θ)
  Eq. 2  — 3D scan → spherical:   φ = √(θ_x²+θ_y²),  θ = arctan(θ_x/θ_y)
  Eq. 3  — spherical → Cartesian: x=R·sin(φ)·cos(θ), y=R·sin(φ)·sin(θ), z=R·cos(φ)
  Eq. 12 — calibrated scan angle: θ_j = k·c_FS·(j/N) + k_0

3D implementation strategy
--------------------------
Full volumetric 3D inverse mapping (holding all coords in RAM) is impractical
for 800×2048×800 volumes.  Instead we use the *3D-coupled per-slice* approach:

  • For each B-scan k, compute its slow-axis angle θ_slow,k  (Eq. 12).
  • Within that slice, every (i, j) pixel has total deflection
      φ(i,j,k) = √(θ_fast,j² + θ_slow,k²)   ← key 3D coupling term
  • Cartesian coordinates follow Eq. 3 for the fast axis and depth;
    the slow-axis physical offset is returned as layer metadata.
  • All slices share one unified output grid sized from the worst-case
    corner angle  φ_corner = √(θ_fast,max² + θ_slow,max²).

This matches the paper's physics exactly for each B-scan's internal geometry.
The only omission vs. pure 3D is inter-slice interpolation (negligible for
features that are smooth along the slow axis).
"""
from __future__ import annotations
import dataclasses
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import map_coordinates


# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (defaults = Bayhaqi et al. 2025, Table 1 / Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveCorrectionParams:
    """Calibrated parameters for the PanOCT 400 kHz contact-based system."""

    # Eq. 12 calibrated scan angles (from physical phantom measurement)
    theta_max_fast_deg: float = 49.85
    """Max scan angle, fast axis (±°). Paper value: ±49.85°."""

    theta_max_slow_deg: float = 49.96
    """Max scan angle, slow axis (±°). Paper value: ±49.96°."""

    theta_offset_fast_deg: float = -0.61
    """Offset k_0 for fast axis (°). Paper value: −0.61°."""

    theta_offset_slow_deg: float = 0.27
    """Offset k_0 for slow axis (°). Paper value: +0.27°."""

    # System geometry
    axial_range_mm: float = 12.0
    """Total axial imaging range in air (mm). System spec: 12 mm."""

    n_axial_pixels: int = 2048
    """Pixels per A-scan. Overridden automatically from input shape."""

    pivot_distance_mm: float = 16.0
    """Distance from virtual pivot to image top (mm). Adult ≈ 16 mm."""

    refractive_index: float = 1.336
    """Average ocular refractive index (air→tissue). Standard: 1.336."""

    @property
    def delta_r_mm(self) -> float:
        """Axial pixel spacing in tissue (mm) — Δr in paper."""
        return (self.axial_range_mm / self.n_axial_pixels) / self.refractive_index


# ─────────────────────────────────────────────────────────────────────────────
# Scan angle computation  — Eq. 12
# ─────────────────────────────────────────────────────────────────────────────

def compute_scan_angles(
    n_pts: int,
    theta_max_deg: float,
    theta_offset_deg: float = 0.0,
) -> np.ndarray:
    """Return calibrated scan angles (radians) for n_pts positions.

    Implements Eq. 12:  θ_j = k·c_FS·(j/N) + k_0
    where k·c_FS = theta_max_deg and k_0 = theta_offset_deg.

    Index j runs from −N to +N (centre = 0).
    """
    N = n_pts / 2.0
    j = np.arange(n_pts, dtype=np.float64) - N
    return np.deg2rad(theta_max_deg) * (j / N) + np.deg2rad(theta_offset_deg)


# ─────────────────────────────────────────────────────────────────────────────
# 3D output geometry  — unified grid for the whole volume
# ─────────────────────────────────────────────────────────────────────────────

def _compute_3d_output_geometry(H: int, W: int, D: int,
                                params: CurveCorrectionParams):
    """Compute the unified Cartesian output grid for the full 3D volume.

    Uses the worst-case corner angle
        φ_corner = √(θ_fast,max² + θ_slow,max²)
    so every B-scan fits into the same output shape.

    Returns
    -------
    theta_fast : (W,) fast-axis scan angles [rad]
    theta_slow : (D,) slow-axis scan angles [rad]
    R          : (H,) radial distances from pivot [mm]
    grid       : dict with keys x_min, x_max, z_top, z_max, H_out, W_out, out_res
    """
    theta_fast = compute_scan_angles(W, params.theta_max_fast_deg,
                                     params.theta_offset_fast_deg)
    theta_slow = compute_scan_angles(D, params.theta_max_slow_deg,
                                     params.theta_offset_slow_deg)
    delta_r = params.delta_r_mm
    R = params.pivot_distance_mm + np.arange(H, dtype=np.float64) * delta_r

    # Worst-case total deflection at the image corner (Eq. 2)
    phi_corner = float(np.sqrt(theta_fast[-1]**2 + theta_slow[-1]**2))

    # Physical fast-axis extent at the bottom of the image
    phi_fast_grid, _ = np.meshgrid(
        np.abs(theta_fast), np.zeros(H), indexing='ij')
    # Use full 3D phi (slow at max) for outermost columns
    phi_full = np.sqrt(phi_fast_grid**2 + theta_slow[-1]**2)
    sinc_phi = np.where(phi_full > 1e-12, np.sin(phi_full) / phi_full, 1.0)
    x_phys_max = float((R[-1] * sinc_phi[-1, -1] * abs(theta_fast[-1])).max())

    # z range: top = R[0]·cos(φ_corner), bottom = R[-1]·cos(0)
    z_top = float(R[0] * np.cos(phi_corner))
    z_max = float(R[-1])

    # Output pixel size: lateral arc-length at centre depth (isotropic)
    R_center = R[H // 2]
    d_theta = abs(theta_fast[1] - theta_fast[0]) if W > 1 else 1e-3
    out_res = R_center * d_theta  # mm per output pixel

    W_out = int(np.ceil(2 * x_phys_max / out_res))
    H_out = int(np.ceil((z_max - z_top) / out_res))
    W_out = min(max(W_out, W), W * 3)
    H_out = min(max(H_out, 64), H * 3)

    grid = dict(x_min=-x_phys_max, x_max=x_phys_max,
                z_top=z_top, z_max=z_max,
                H_out=H_out, W_out=W_out, out_res=out_res)
    return theta_fast, theta_slow, R, grid


def compute_output_scale(H: int, W: int, D: int,
                         params: CurveCorrectionParams) -> tuple:
    """Return physical pixel size (z_res_mm, x_res_mm) of the corrected output.

    Set napari layer scale with::

        z_res, x_res = compute_output_scale(H, W, D, params)
        layer.scale[-2:] = (z_res, x_res)
    """
    _, _, _, grid = _compute_3d_output_geometry(H, W, D, params)
    z_res = (grid['z_max'] - grid['z_top']) / grid['H_out']
    x_res = (grid['x_max'] - grid['x_min']) / grid['W_out']
    return z_res, x_res


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice 2D correction  (3D-coupled)  — Eq. 1–3
# ─────────────────────────────────────────────────────────────────────────────

def correct_bscan_2d(
    bscan: np.ndarray,
    params: CurveCorrectionParams,
    theta_slow_rad: float = 0.0,
    order: int = 1,
    grid: dict | None = None,
) -> np.ndarray:
    """Correct one B-scan using the 3D-coupled formula (Eq. 1–3).

    Parameters
    ----------
    bscan         : (H, W) array
    params        : calibration parameters
    theta_slow_rad: slow-axis angle for this B-scan (θ_y,k in Eq. 2).
                    0 → reduces to pure 2D (Eq. 1).  Non-zero → 3D coupling.
    order         : interpolation order (1=linear, 0=nearest for labels)
    grid          : pre-computed geometry dict; computed if None
    """
    H, W = bscan.shape
    if params.n_axial_pixels != H:
        params = dataclasses.replace(params, n_axial_pixels=H)

    if grid is None:
        # Standalone call: use single-slice geometry (D=1, theta_slow=0)
        _, _, R, grid = _compute_3d_output_geometry(H, W, 1, params)
        delta_r = params.delta_r_mm
        R = params.pivot_distance_mm + np.arange(H, dtype=np.float64) * delta_r
    else:
        delta_r = params.delta_r_mm
        R = params.pivot_distance_mm + np.arange(H, dtype=np.float64) * delta_r

    H_out, W_out = grid['H_out'], grid['W_out']
    x_min, x_max = grid['x_min'], grid['x_max']
    z_top, z_max = grid['z_top'], grid['z_max']

    x_out = np.linspace(x_min, x_max, W_out, dtype=np.float64)
    z_out = np.linspace(z_top, z_max, H_out, dtype=np.float64)
    x_grid, z_grid = np.meshgrid(x_out, z_out)  # (H_out, W_out)

    # ── Inverse mapping: Cartesian (x_fast, z) → source (row, col) ───────
    # In 3D (Eq. 3):
    #   x_fast = R·sin(φ)·sin(θ)  where sin(θ)=θ_fast/φ  → x_fast = R·sin(φ)·θ_fast/φ
    #   z      = R·cos(φ)         where φ=√(θ_fast²+θ_slow²)
    #
    # Given output (x_fast, z) and known θ_slow_k, we solve for (R, θ_fast):
    #   From z = R·cos(φ) and x_fast = R·sin(φ)·θ_fast/φ
    #   Note: lateral magnitude in 3D is √(x_fast²+x_slow²) where x_slow=R·sin(φ)·θ_slow/φ
    #   In per-slice 2D view, x_slow is a constant offset; we solve the (x_fast, z) system.
    #
    # Iterative inversion (converges in 3–4 steps):
    #   1. Initial: φ ≈ arctan(x_fast / z)  (ignoring slow coupling)
    #   2. Refine:  θ_fast = φ_total·sin(arctan(x_fast/z)) where φ_total=√(θ_fast²+θ_slow²)

    # Initial estimate of total φ from (x_fast, z) alone
    phi_total = np.sqrt(x_grid**2 + z_grid**2)  # ~ R initially
    # Better: use arctan2 of lateral/axial
    lat = np.abs(x_grid)
    phi_fast_approx = np.arctan2(lat, np.maximum(z_grid, 1e-9))

    # Iterative refinement for θ_fast given θ_slow_k
    theta_fast_src = phi_fast_approx.copy()
    for _ in range(4):
        phi_total_iter = np.sqrt(theta_fast_src**2 + theta_slow_rad**2)
        sinc_phi = np.where(phi_total_iter > 1e-12,
                            np.sin(phi_total_iter) / phi_total_iter, 1.0)
        # From x_fast = R·sinc_phi·θ_fast and z = R·cos(φ_total):
        # R·cos(φ_total) = z_grid  →  R = z_grid / cos(φ_total)
        cos_phi = np.cos(phi_total_iter)
        R_src = np.where(np.abs(cos_phi) > 1e-9,
                         z_grid / cos_phi,
                         np.sqrt(x_grid**2 + z_grid**2))
        theta_fast_src = np.where(
            R_src * sinc_phi > 1e-12,
            x_grid / (R_src * sinc_phi),
            0.0)

    # Final R from converged θ_fast
    phi_final = np.sqrt(theta_fast_src**2 + theta_slow_rad**2)
    cos_phi_final = np.cos(phi_final)
    R_src_final = np.where(np.abs(cos_phi_final) > 1e-9,
                           z_grid / cos_phi_final,
                           np.sqrt(x_grid**2 + z_grid**2))

    # Source row (depth) and column (fast angle)
    row_src = (R_src_final - params.pivot_distance_mm) / delta_r

    N = W / 2.0
    theta_max_rad = np.deg2rad(params.theta_max_fast_deg)
    theta_off_rad = np.deg2rad(params.theta_offset_fast_deg)
    col_src = ((theta_fast_src - theta_off_rad) / theta_max_rad) * N + N

    coords = np.array([row_src.ravel(), col_src.ravel()])
    corrected = map_coordinates(
        bscan.astype(np.float64), coords,
        order=order, mode='constant', cval=0.0,
    ).reshape(H_out, W_out)

    if bscan.dtype.kind in ('u', 'i'):
        info = np.iinfo(bscan.dtype)
        corrected = np.clip(corrected, info.min, info.max)
    return corrected.astype(bscan.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Volume-level correction  — 3D-coupled, Eq. 12 + 2–3
# ─────────────────────────────────────────────────────────────────────────────

def correct_volume_slicewise(
    volume: np.ndarray,
    params: CurveCorrectionParams,
    is_labels: bool = False,
    progress_callback=None,
) -> np.ndarray:
    """Apply 3D-coupled per-slice correction to a full volume.

    For each B-scan k:
      1. Compute θ_slow,k  from Eq. 12.
      2. Correct the B-scan using the coupled φ = √(θ_fast² + θ_slow,k²)  (Eq. 2–3).
      3. Write into a shared Cartesian output grid sized from φ_corner.

    Parameters
    ----------
    volume            : (D, H, W) array — D B-scans, H depth, W A-scans
    params            : calibration parameters
    is_labels         : use nearest-neighbour (order=0) for label masks
    progress_callback : optional callable(str) for UI progress updates
    """
    D, H, W = volume.shape
    if params.n_axial_pixels != H:
        params = dataclasses.replace(params, n_axial_pixels=H)

    order = 0 if is_labels else 1

    # Pre-compute unified 3D output geometry
    theta_fast, theta_slow, R, grid = _compute_3d_output_geometry(
        H, W, D, params)

    H_out, W_out = grid['H_out'], grid['W_out']
    corrected = np.zeros((D, H_out, W_out), dtype=volume.dtype)

    for k in range(D):
        if progress_callback is not None and k % max(1, D // 20) == 0:
            pct = int((k + 1) * 100 / D)
            progress_callback(f"Curve correcting... {pct}%")

        corrected[k] = correct_bscan_2d(
            volume[k], params,
            theta_slow_rad=float(theta_slow[k]),
            order=order,
            grid=grid,
        )

    if progress_callback is not None:
        progress_callback("Curve correction done")

    return corrected

# ─────────────────────────────────────────────────────────────────────────────
# Volume-level correction — Full 3D Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def correct_volume_3d_full(
    volume: np.ndarray,
    params: CurveCorrectionParams,
    is_labels: bool = False,
    progress_callback=None,
    chunk_size_z: int = 32,
) -> np.ndarray:
    """Apply full 3D interpolation correction to a volume.

    Unlike correct_volume_slicewise which processes each B-scan independently,
    this maps the full 3D Cartesian output volume directly from the 3D source.
    This enables inter-slice interpolation along the slow axis, at the cost of
    higher memory usage.

    Processed in chunks along the output Z (depth) axis to control RAM.

    Parameters
    ----------
    volume            : (D, H, W) array
    params            : calibration parameters
    is_labels         : use nearest-neighbour (order=0) for label masks
    progress_callback : optional callable(str) for UI progress updates
    chunk_size_z      : number of depth slices to compute simultaneously
    """
    D, H, W = volume.shape
    if params.n_axial_pixels != H:
        params = dataclasses.replace(params, n_axial_pixels=H)

    order = 0 if is_labels else 1
    delta_r = params.delta_r_mm

    theta_fast, theta_slow, R, grid = _compute_3d_output_geometry(
        H, W, D, params)

    H_out, W_out = grid['H_out'], grid['W_out']
    x_min, x_max = grid['x_min'], grid['x_max']
    z_top, z_max = grid['z_top'], grid['z_max']

    # D_out = D to preserve the number of slices in slow axis,
    # but their physical spacing is dictated by out_res.
    # The slow axis maps linearly to theta_slow indices.
    D_out = D
    
    # We maintain isotropic resolution laterally and axially
    out_res = grid['out_res']
    
    # Y physical coordinates (slow axis) for the output slices.
    # We space the output slices physically by out_res, centered around 0.
    y_max = (D_out / 2.0) * out_res
    y_out = np.linspace(-y_max, y_max, D_out, dtype=np.float64)
    x_out = np.linspace(x_min, x_max, W_out, dtype=np.float64)
    z_out = np.linspace(z_top, z_max, H_out, dtype=np.float64)

    corrected = np.zeros((D_out, H_out, W_out), dtype=volume.dtype)

    # Process in Z chunks to save memory
    z_chunks = list(range(0, H_out, chunk_size_z))
    total_chunks = len(z_chunks)

    # We need 3D meshgrids for (y, z, x). Note order: (D_out, chunk_z, W_out)
    for chunk_idx, z_start in enumerate(z_chunks):
        if progress_callback is not None:
            pct = int(chunk_idx * 100 / total_chunks)
            progress_callback(f"Full 3D correcting... {pct}%")

        z_end = min(z_start + chunk_size_z, H_out)
        z_chunk_out = z_out[z_start:z_end]
        
        y_grid, z_grid, x_grid = np.meshgrid(y_out, z_chunk_out, x_out, indexing='ij')

        # 1. Cartesian to Spherical
        # φ = arccos(z/R) -> z = R·cos(φ) => R = z / cos(φ)
        # R = √(x² + y² + z²)
        R_src = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
        
        # Avoid divide by zero
        R_safe = np.maximum(R_src, 1e-9)
        
        # φ (total polar angle from z-axis)
        phi = np.arccos(np.clip(z_grid / R_safe, -1.0, 1.0))
        
        # θ (azimuth angle in xy plane)
        # In paper: x = R·sin(φ)·cos(θ_az), y = R·sin(φ)·sin(θ_az)
        # Here we need to map to θ_fast and θ_slow.
        # Paper Eq 2: φ = √(θ_x² + θ_y²), θ_az = arctan(θ_x/θ_y)
        # By substitution: θ_x = φ · sin(θ_az) and θ_y = φ · cos(θ_az)
        
        # Since x is fast and y is slow:
        # θ_fast ≈ φ * (x / √(x²+y²))
        # θ_slow ≈ φ * (y / √(x²+y²))
        
        lat_r = np.sqrt(x_grid**2 + y_grid**2)
        lat_r_safe = np.maximum(lat_r, 1e-9)
        
        theta_fast_src = phi * (x_grid / lat_r_safe)
        theta_slow_src = phi * (y_grid / lat_r_safe)

        # 2. Spherical to Image Indices
        # R -> row
        row_src = (R_src - params.pivot_distance_mm) / delta_r

        # θ_fast -> col
        N_W = W / 2.0
        theta_max_fast_rad = np.deg2rad(params.theta_max_fast_deg)
        theta_off_fast_rad = np.deg2rad(params.theta_offset_fast_deg)
        col_src = ((theta_fast_src - theta_off_fast_rad) / theta_max_fast_rad) * N_W + N_W

        # θ_slow -> slice
        N_D = D / 2.0
        theta_max_slow_rad = np.deg2rad(params.theta_max_slow_deg)
        theta_off_slow_rad = np.deg2rad(params.theta_offset_slow_deg)
        slice_src = ((theta_slow_src - theta_off_slow_rad) / theta_max_slow_rad) * N_D + N_D

        coords = np.array([slice_src.ravel(), row_src.ravel(), col_src.ravel()])
        
        chunk_corrected = map_coordinates(
            volume.astype(np.float64), coords,
            order=order, mode='constant', cval=0.0,
        ).reshape((D_out, z_end - z_start, W_out))

        if volume.dtype.kind in ('u', 'i'):
            info = np.iinfo(volume.dtype)
            chunk_corrected = np.clip(chunk_corrected, info.min, info.max)
            
        corrected[:, z_start:z_end, :] = chunk_corrected.astype(volume.dtype)

    if progress_callback is not None:
        progress_callback("Full 3D correction done")

    return corrected
