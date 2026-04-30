import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import gaussian
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal

@dataclass
class EnFaceResult:
    enface_mask: np.ndarray
    enface_prior: np.ndarray
    roi_x_range: Dict[int, Tuple[int, int]]
    roi_prior: Dict[int, np.ndarray]

def generate_enface_prior(enface_mask: np.ndarray, prior_sigma: float = 10.0) -> np.ndarray:
    """
    Generate a 2D distance-transform-based prior probability map from the en-face mask.
    Pixels inside the mask = probability 1.0
    Pixels outside = probability decays with distance (Gaussian falloff)
    """
    # If mask is empty, return all zeros
    if not np.any(enface_mask):
        return np.zeros_like(enface_mask, dtype=float)
        
    # If mask is full, return all ones
    if np.all(enface_mask):
        return np.ones_like(enface_mask, dtype=float)

    # distance_transform_edt computes distance to nearest background (0) pixel
    # We want distance to nearest foreground (1) pixel for the outside.
    # So we compute EDT on the inverse of the mask
    dist_to_mask = distance_transform_edt(~enface_mask)
    
    # Gaussian falloff outside the mask: exp(-d^2 / (2 * sigma^2))
    # Inside the mask, dist_to_mask is 0, so exp(0) = 1.0
    prior_map = np.exp(-(dist_to_mask**2) / (2 * prior_sigma**2))
    
    return prior_map

def map_enface_to_bscan(
    enface_mask: np.ndarray,
    enface_prior: np.ndarray,
    bscan_shape: Tuple[int, int, int],  # (slices, height, width)
    mapping_mode: Literal["linear", "custom_affine"] = "linear",
    affine_matrix: Optional[np.ndarray] = None
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, np.ndarray]]:
    """
    Map en-face ROI to B-scan coordinates.
    Assume linear mapping where rows = slices and columns = lateral x-position,
    unless affine matrix is provided.
    
    Returns:
        roi_x_range: dict mapping slice index to (x_min, x_max)
        roi_prior: dict mapping slice index to 1D prior array for that slice
    """
    roi_x_range = {}
    roi_prior = {}
    
    n_slices = bscan_shape[0]
    bscan_width = bscan_shape[2]
    
    if mapping_mode == "linear":
        # Assume 1:1 mapping if dimensions match, or nearest neighbor scale
        enface_h, enface_w = enface_mask.shape
        
        for i in range(n_slices):
            # Map slice index i to enface row
            row = int(i * enface_h / n_slices)
            row = min(row, enface_h - 1)
            
            # Map bscan columns to enface columns
            # For simplicity, if we assume 1:1 width:
            # We scale the row prior to bscan_width
            row_prior_raw = enface_prior[row, :]
            row_mask_raw = enface_mask[row, :]
            
            # Upsample/downsample to bscan_width
            x_indices = np.linspace(0, enface_w - 1, bscan_width).astype(int)
            row_prior = row_prior_raw[x_indices]
            row_mask = row_mask_raw[x_indices]
            
            if np.any(row_mask):
                valid_x = np.where(row_mask)[0]
                x_min, x_max = valid_x[0], valid_x[-1]
                roi_x_range[i] = (x_min, x_max)
                roi_prior[i] = row_prior
            else:
                # Still store prior in case it's used for inter-slice interpolation
                # but no strict roi_x_range mask
                roi_prior[i] = row_prior
                
    elif mapping_mode == "custom_affine" and affine_matrix is not None:
        raise NotImplementedError("Custom affine mapping not yet implemented")
    else:
        raise ValueError(f"Unknown mapping mode: {mapping_mode}")
        
    return roi_x_range, roi_prior

def process_enface_roi(
    enface_mask: np.ndarray,
    bscan_shape: Tuple[int, int, int],
    prior_sigma: float = 10.0,
    mapping_mode: Literal["linear", "custom_affine"] = "linear",
    affine_matrix: Optional[np.ndarray] = None
) -> EnFaceResult:
    """
    Process the drawn en-face mask, generate prior, and map to B-scan stack.
    """
    prior_map = generate_enface_prior(enface_mask, prior_sigma)
    roi_x_range, roi_prior = map_enface_to_bscan(
        enface_mask, prior_map, bscan_shape, mapping_mode, affine_matrix
    )
    
    return EnFaceResult(
        enface_mask=enface_mask,
        enface_prior=prior_map,
        roi_x_range=roi_x_range,
        roi_prior=roi_prior
    )
