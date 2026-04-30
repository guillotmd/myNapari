import numpy as np
from skimage.measure import marching_cubes
import trimesh
from typing import Tuple, Dict

from .scale import ScaleCalibration

def generate_tumor_mesh(
    tumor_masks: Dict[int, np.ndarray],
    bscan_shape: Tuple[int, int, int],
    scale_calib: ScaleCalibration,
    smooth_iters: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct 3D tumor mesh from the stacked 2D masks.
    Returns:
        verts: (V, 3) array of vertices in physical um coordinates
        faces: (F, 3) array of triangle indices
    """
    n_slices, h, w = bscan_shape
    
    # Create 3D binary volume
    volume_3d = np.zeros((n_slices, h, w), dtype=bool)
    
    for i in range(n_slices):
        if i in tumor_masks:
            mask_1d = tumor_masks[i]
            # The 1D mask corresponds to the width of the B-scan
            # But the height (y-axis) of the tumor is represented by the deviation
            # Wait, the prompt says:
            # "Stack all per-slice tumor masks into a 3D binary volume"
            # But tumor_masks[i] is just a 1D array of True/False for columns.
            # We need the 2D mask (fill between baseline and retina)!
            pass
            
    # Actually, let's implement the 2D fill first properly.
    # The prompt: "Stack all per-slice tumor masks into a 3D binary volume, 
    # with physical voxel dimensions"
    # To do this correctly, we need the baseline and inner_retina_edge.
    # Since we don't have them all passed here, we should assume `tumor_masks` 
    # here is actually the list of 2D boolean masks for each slice.
    
    # We will assume `tumor_masks_2d` is a list/dict of (H, W) boolean arrays.
    raise NotImplementedError("2D fill into 3D volume requires baseline and retina edges")

def generate_tumor_mesh_from_2d_masks(
    masks_3d: np.ndarray,
    scale_calib: ScaleCalibration,
    smooth_iters: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    
    # skimage marching cubes expects spacing as (spacing_z, spacing_y, spacing_x)
    # Our dims are (slice, y, x)
    # So spacing is (inter_slice_spacing, axial_resolution, lateral_resolution)
    spacing = (
        scale_calib.inter_slice_spacing,
        scale_calib.axial_resolution,
        scale_calib.lateral_resolution
    )
    
    try:
        verts, faces, normals, values = marching_cubes(
            masks_3d, 
            level=0.5,
            spacing=spacing
        )
    except RuntimeError:
        # If no surface found
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        
    if smooth_iters > 0 and len(verts) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
        verts = mesh.vertices
        faces = mesh.faces
        
    return verts, faces

def create_2d_fill_mask(
    tumor_columns: np.ndarray,
    baseline_y: np.ndarray,
    inner_retina_edge: np.ndarray,
    h: int, w: int
) -> np.ndarray:
    """
    Create a 2D boolean mask of the tumor cross-section.
    tumor_columns: 1D boolean array
    """
    mask_2d = np.zeros((h, w), dtype=bool)
    for x in range(w):
        if tumor_columns[x]:
            y_base = int(round(baseline_y[x]))
            y_ret = int(round(inner_retina_edge[x]))
            if not np.isnan(y_ret) and not np.isnan(y_base):
                # y increases downwards, so y_ret is < y_base
                y_min = min(y_base, y_ret)
                y_max = max(y_base, y_ret)
                y_min = max(0, y_min)
                y_max = min(h - 1, y_max)
                mask_2d[y_min:y_max+1, x] = True
    return mask_2d
