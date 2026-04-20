"""
Widget re-exports for the florian-tools plugin manifest.
The batch widget lives here; segmentation widgets are imported
directly from florian_segmentation_mac by the manifest.
"""
from florian_batch_proc._batch_widget import BatchUNPWidget

__all__ = ["BatchUNPWidget"]
