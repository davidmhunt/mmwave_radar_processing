"""Registry for Range-Doppler Detectors."""

from typing import Dict, Type

from mmwave_radar_processing.processors.range_doppler_detector import RangeDopplerDetector
from mmwave_radar_processing.processors.range_doppler_detector_2d import RangeDopplerDetector2D
from mmwave_radar_processing.processors.range_doppler_detector_sequential import RangeDopplerDetectorSequential
from mmwave_radar_processing.processors.range_doppler_ground_detector import RangeDopplerGroundDetector

_RANGE_DOPPLER_DETECTOR_REGISTRY: Dict[str, Type[RangeDopplerDetector]] = {
    "range_doppler_detector_2d": RangeDopplerDetector2D,
    "range_doppler_detector_sequential": RangeDopplerDetectorSequential,
    "range_doppler_ground_detector": RangeDopplerGroundDetector,
}

def get_range_doppler_detector_registry() -> Dict[str, Type[RangeDopplerDetector]]:
    """
    Get the registry of available Range-Doppler detectors.

    Returns:
        Dict[str, Type[RangeDopplerDetector]]: Dictionary mapping detector names to classes.
    """
    return _RANGE_DOPPLER_DETECTOR_REGISTRY
