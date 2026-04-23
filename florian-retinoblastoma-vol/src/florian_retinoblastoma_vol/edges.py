import numpy as np
from typing import Tuple

def extract_edges(
    segmentation_slice: np.ndarray, 
    retina_label: int = 1, 
    choroid_label: int = 2,
    min_thickness: int = 5,
    ignore_top_px: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the topmost inner retinal and inner choroidal edges from a 2D segmentation slice.
    """
    if segmentation_slice.ndim == 3 and segmentation_slice.shape[2] >= 3:
        red_channel = segmentation_slice[..., 0]
        blue_channel = segmentation_slice[..., 2]
        is_retina = red_channel > 128
        is_choroid = blue_channel > 128
    else:
        is_retina = segmentation_slice == retina_label
        is_choroid = segmentation_slice == choroid_label

    height, width = is_retina.shape
    
    # Apply ignore_top_px by zeroing out the top region
    if ignore_top_px > 0:
        is_retina = is_retina.copy()
        is_choroid = is_choroid.copy()
        is_retina[:ignore_top_px, :] = False
        is_choroid[:ignore_top_px, :] = False
    
    inner_retina_edge = np.full(width, np.nan)
    inner_choroid_edge = np.full(width, np.nan)
    
    for x in range(width):
        # Retina: Find groups, filter out thin seeds, take the TOP-most valid group
        ret_indices = np.where(is_retina[:, x])[0]
        if len(ret_indices) > 0:
            breaks = np.where(np.diff(ret_indices) > 1)[0]
            splits = np.split(ret_indices, breaks + 1)
            valid_splits = [s for s in splits if len(s) >= min_thickness]
            if len(valid_splits) > 0:
                inner_retina_edge[x] = valid_splits[0][0]  # Top layer
            else:
                largest = max(splits, key=len)
                inner_retina_edge[x] = largest[0]
                
        # Choroid: Find groups, filter out thin seeds, take the BOTTOM-most valid group
        cho_indices = np.where(is_choroid[:, x])[0]
        if len(cho_indices) > 0:
            breaks = np.where(np.diff(cho_indices) > 1)[0]
            splits = np.split(cho_indices, breaks + 1)
            valid_splits = [s for s in splits if len(s) >= min_thickness]
            if len(valid_splits) > 0:
                inner_choroid_edge[x] = valid_splits[-1][0] # Bottom layer
            else:
                largest = max(splits, key=len)
                inner_choroid_edge[x] = largest[0]
            
    return inner_retina_edge, inner_choroid_edge
