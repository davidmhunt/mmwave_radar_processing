"""
Registry of detectors.
"""

from __future__ import annotations

from typing import Dict, Type, Any, Union

from mmwave_radar_processing.detectors.base import BaseCFAR2D, BaseCFAR1D
from mmwave_radar_processing.detectors.ca_cfar import CaCFAR2D, CaCFAR1D
from mmwave_radar_processing.detectors.os_cfar import OsCFAR2D, OsCFAR1D
from mmwave_radar_processing.detectors.go_so_cfar import GoCFAR1D, SoCFAR1D


def get_detector_registry() -> Dict[str, Type[Union[BaseCFAR2D, BaseCFAR1D]]]:
    """
    Returns a dictionary mapping detector names to their corresponding classes.
    """
    registry = {
        "ca_cfar_1d": CaCFAR1D,
        "ca_cfar_2d": CaCFAR2D,
        "os_cfar_1d": OsCFAR1D,
        "os_cfar_2d": OsCFAR2D,
        "go_cfar_1d": GoCFAR1D,
        "so_cfar_1d": SoCFAR1D,
    }
    return registry
