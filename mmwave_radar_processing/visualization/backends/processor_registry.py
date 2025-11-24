"""
Registry of processors and their associated views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from mmwave_radar_processing.logging.logger import get_logger

# Import processor classes without modifying them.
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor
from mmwave_radar_processing.processors.micro_doppler_resp import MicroDopplerProcessor
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.processors.range_resp import RangeProcessor


@dataclass
class ProcessorSpec:
    """Specification of a processor and its view."""

    key: str
    display_name: str
    processor_cls: Type[Any]
    view_cls: Optional[Type[Any]]
    required_inputs: Optional[str]
    output_schema: Optional[str]
    enabled: bool = True


def get_default_registry(logger=None) -> Dict[str, ProcessorSpec]:
    """Build the default processor registry.

    Args:
        logger: Optional logger instance.

    Returns:
        Mapping from registry keys to processor specifications.
    """
    logger = logger or get_logger(__name__)
    # Local imports to avoid circular dependencies when views import registry.
    from mmwave_radar_processing.visualization.views.doppler_azimuth_view import (
        DopplerAzimuthView,
    )
    from mmwave_radar_processing.visualization.views.micro_doppler_view import (
        MicroDopplerView,
    )
    from mmwave_radar_processing.visualization.views.range_angle_view import (
        RangeAngleView,
    )
    from mmwave_radar_processing.visualization.views.range_doppler_view import (
        RangeDopplerView,
    )
    from mmwave_radar_processing.visualization.views.range_response_view import (
        RangeResponseView,
    )

    registry = {
        "range_doppler_resp": ProcessorSpec(
            key="range_doppler_resp",
            display_name="Range-Doppler",
            processor_cls=RangeDopplerProcessor,
            view_cls=RangeDopplerView,
            required_inputs="adc_cube",
            output_schema="range_bins x velocity_bins ndarray",
            enabled=True,
        ),
        "range_resp": ProcessorSpec(
            key="range_resp",
            display_name="Range Response",
            processor_cls=RangeProcessor,
            view_cls=RangeResponseView,
            required_inputs="adc_cube",
            output_schema="range_bins ndarray",
            enabled=True,
        ),
        "range_angle_resp": ProcessorSpec(
            key="range_angle_resp",
            display_name="Range-Angle",
            processor_cls=RangeAngleProcessor,
            view_cls=RangeAngleView,
            required_inputs="adc_cube",
            output_schema="range_bins x angle_bins ndarray",
            enabled=True,
        ),
        "micro_doppler_resp": ProcessorSpec(
            key="micro_doppler_resp",
            display_name="Micro-Doppler",
            processor_cls=MicroDopplerProcessor,
            view_cls=MicroDopplerView,
            required_inputs="adc_cube",
            output_schema="time x velocity ndarray",
            enabled=True,
        ),
        "doppler_azimuth_resp": ProcessorSpec(
            key="doppler_azimuth_resp",
            display_name="Doppler-Azimuth",
            processor_cls=DopplerAzimuthProcessor,
            view_cls=DopplerAzimuthView,
            required_inputs="adc_cube",
            output_schema="angle_bins x velocity ndarray",
            enabled=True,
        ),
    }
    logger.debug("Default processor registry created with keys: %s", list(registry.keys()))
    return registry
