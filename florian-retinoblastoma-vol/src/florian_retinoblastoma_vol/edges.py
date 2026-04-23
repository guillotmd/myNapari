import numpy as np
from typing import Tuple

def extract_edges(segmentation_slice: np.ndarray, retina_label: int = 1, choroid_label: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the topmost (min y) inner retinal and inner choroidal edges from a 2D segmentation slice.
    
    Expected format of segmentation_slice: (height, width, 3) or (height, width, 4) RGB/RGBA image
    or a 2D array where different integer labels represent different classes.
    Assuming the problem specifies:
    - Red channel pixels (or a specific label) = inner retinal surface
    - Blue channel pixels (or a specific label) = inner choroidal surface
    
    If it's an RGB image:
    Red channel dominance -> Retina
    Blue channel dominance -> Choroid
    
    Returns:
        inner_retina_edge: 1D numpy array of shape (width,) containing the topmost y-coordinate
                           for the retina at each column. Missing values are represented as np.nan.
        inner_choroid_edge: 1D numpy array of shape (width,) containing the topmost y-coordinate
                            for the choroid at each column. Missing values are represented as np.nan.
    """
    if segmentation_slice.ndim == 3 and segmentation_slice.shape[2] >= 3:
        # RGB(A) image
        red_channel = segmentation_slice[..., 0]
        blue_channel = segmentation_slice[..., 2]
        
        is_retina = red_channel > 128
        is_choroid = blue_channel > 128
    else:
        # It's a label image
        is_retina = segmentation_slice == retina_label
        is_choroid = segmentation_slice == choroid_label

    height, width = is_retina.shape
    
    inner_retina_edge = np.full(width, np.nan)
    inner_choroid_edge = np.full(width, np.nan)
    
    for x in range(width):
        # Find first True along y-axis (topmost)
        retina_y_indices = np.where(is_retina[:, x])[0]
        if len(retina_y_indices) > 0:
            inner_retina_edge[x] = retina_y_indices[0]
            
        choroid_y_indices = np.where(is_choroid[:, x])[0]
        if len(choroid_y_indices) > 0:
            inner_choroid_edge[x] = choroid_y_indices[0]
            
    return inner_retina_edge, inner_choroid_edge
