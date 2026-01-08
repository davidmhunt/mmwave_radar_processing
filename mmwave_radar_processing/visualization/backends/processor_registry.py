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
from mmwave_radar_processing.processors.range_doppler_detection.range_doppler_detector_2d import RangeDopplerDetector2D
from mmwave_radar_processing.processors.range_doppler_detection.range_doppler_detector_sequential import RangeDopplerDetectorSequential
from mmwave_radar_processing.processors.range_doppler_detection.range_doppler_ground_detector import RangeDopplerGroundDetector
from mmwave_radar_processing.processors.range_detector import RangeDetector
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.point_cloud_generator import PointCloudGenerator


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
    num_frames_history: int = 1
    view_keys: Optional[list[str]] = None


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
    from mmwave_radar_processing.visualization.views.range_doppler_detector_view import (
        RangeDopplerDetectorView,
    )
    from mmwave_radar_processing.visualization.views.range_detector_view import (
        RangeDetectorView,
    )
    from mmwave_radar_processing.visualization.views.altitude_view import (
        AltitudeView,
    )
    from mmwave_radar_processing.visualization.views.point_cloud_view import (
        PointCloudView,
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
            num_frames_history=1,
            view_keys=["range_bins", "vel_bins"],
        ),
        "range_resp": ProcessorSpec(
            key="range_resp",
            display_name="Range Response",
            processor_cls=RangeProcessor,
            view_cls=RangeResponseView,
            required_inputs="adc_cube",
            output_schema="range_bins ndarray",
            enabled=True,
            num_frames_history=1,
            view_keys=["data"],
        ),
        "range_angle_resp": ProcessorSpec(
            key="range_angle_resp",
            display_name="Range-Angle",
            processor_cls=RangeAngleProcessor,
            view_cls=RangeAngleView,
            required_inputs="adc_cube",
            output_schema="range_bins x angle_bins ndarray",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "angle_bins"],
        ),
        "micro_doppler_resp": ProcessorSpec(
            key="micro_doppler_resp",
            display_name="Micro-Doppler",
            processor_cls=MicroDopplerProcessor,
            view_cls=MicroDopplerView,
            required_inputs="adc_cube",
            output_schema="time x velocity ndarray",
            enabled=True,
            num_frames_history=20,
            view_keys=["time_bins", "vel_bins"],
        ),
        "doppler_azimuth_resp": ProcessorSpec(
            key="doppler_azimuth_resp",
            display_name="Doppler-Azimuth",
            processor_cls=DopplerAzimuthProcessor,
            view_cls=DopplerAzimuthView,
            required_inputs="adc_cube",
            output_schema="angle_bins x velocity ndarray",
            enabled=True,
            num_frames_history=1,
            view_keys=["angle_bins", "vel_bins"],
        ),
        "range_doppler_detector_2d": ProcessorSpec(
            key="range_doppler_detector_2d",
            display_name="Range-Doppler Detector 2D",
            processor_cls=RangeDopplerDetector2D,
            view_cls=RangeDopplerDetectorView,
            required_inputs="adc_cube",
            output_schema="N x 2 ndarray (detections)",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "vel_bins", "dets", "rng_dop_resp"],
        ),
        "range_doppler_detector_sequential": ProcessorSpec(
            key="range_doppler_detector_sequential",
            display_name="Range-Doppler Detector Sequential",
            processor_cls=RangeDopplerDetectorSequential,
            view_cls=RangeDopplerDetectorView,
            required_inputs="adc_cube",
            output_schema="N x 2 ndarray (detections)",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "vel_bins", "dets", "rng_dop_resp"],
        ),
        "range_doppler_ground_detector": ProcessorSpec(
            key="range_doppler_ground_detector",
            display_name="Range-Doppler Ground Detector",
            processor_cls=RangeDopplerGroundDetector,
            view_cls=RangeDopplerDetectorView,
            required_inputs="adc_cube",
            output_schema="N x 2 ndarray (detections)",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "vel_bins", "dets", "rng_dop_resp"],
        ),
        "range_detector": ProcessorSpec(
            key="range_detector",
            display_name="Range Detector",
            processor_cls=RangeDetector,
            view_cls=RangeDetectorView,
            required_inputs="adc_cube",
            output_schema="range_bins ndarray (detections)",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "dets", "thresholds", "range_resp"],
        ),
        "altimeter": ProcessorSpec(
            key="altimeter",
            display_name="Altimeter",
            processor_cls=Altimeter,
            view_cls=AltitudeView,
            required_inputs="adc_cube",
            output_schema="float (altitude)",
            enabled=True,
            num_frames_history=1,
            view_keys=["range_bins", "coarse_fft_data", "current_altitude_corrected_m"],
        ),
        "point_cloud_generator": ProcessorSpec(
            key="point_cloud_generator",
            display_name="Point Cloud",
            processor_cls=PointCloudGenerator,
            view_cls=PointCloudView,
            required_inputs="adc_cube",
            output_schema="N x 4 ndarray (x, y, z, vel)",
            enabled=True,
            num_frames_history=1,
            view_keys=[], # Point cloud view handles raw array or dict
        ),
    }
    logger.debug("Default processor registry created with keys: %s", list(registry.keys()))
    return registry
