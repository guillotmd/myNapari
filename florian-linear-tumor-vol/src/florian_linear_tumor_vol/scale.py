from dataclasses import dataclass

@dataclass
class ScaleCalibration:
    """
    Physical scale parameters for the UW-OCT volume.
    """
    axial_resolution: float = 1.0  # µm/pixel, depth/y direction
    lateral_resolution: float = 1.0  # µm/pixel, horizontal/x direction within each B-scan
    inter_slice_spacing: float = 1.0  # µm, between consecutive B-scans

    def voxel_volume_um3(self) -> float:
        """Return the volume of a single voxel in µm³."""
        return self.axial_resolution * self.lateral_resolution * self.inter_slice_spacing

    def voxel_volume_mm3(self) -> float:
        """Return the volume of a single voxel in mm³."""
        return self.voxel_volume_um3() / 1e9
