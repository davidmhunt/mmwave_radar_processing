
import numpy as np
import pytest
import os
import sys

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor
from mmwave_radar_processing.processors.micro_doppler_resp import MicroDopplerProcessor
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator
from mmwave_radar_processing.processors.strip_map_SAR_processor import StripMapSARProcessor
from mmwave_radar_processing.processors.simple_synthetic_array_beamformer_processor_multiFrame import SyntheticArrayBeamformerProcessor
from mmwave_radar_processing.processors.point_cloud_generator import PointCloudGenerator
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from cpsl_datasets.cpsl_ds import CpslDS

@pytest.fixture
def config_manager():
    config_path = "configs/6843_RadVel_ods_20Hz.cfg"
    cfg_manager = ConfigManager()
    cfg_manager.load_cfg(config_path, array_geometry="ods", array_direction="down")
    cfg_manager.compute_radar_perforance(profile_idx=0)
    return cfg_manager

@pytest.fixture
def adc_cube():
    # Paths
    dataset_path = "dev_resources/CPSL_RadVel_ods_10Hz_1_sample"
    config_path = "configs/6843_RadVel_ods_20Hz.cfg"

    #initialize the config manager
    config_manager = ConfigManager()
    config_manager.load_cfg(config_path, array_geometry="ods", array_direction="down")
    config_manager.compute_radar_perforance(profile_idx=0)
    
    if not os.path.isdir(dataset_path):
        pytest.skip(f"Dataset not found at {dataset_path}")
        
    # 1. Load Data using CpslDS
    # Assuming standard folder names for this dataset
    try:
        ds = CpslDS(
            dataset_path=dataset_path,
            radar_adc_folder="radar_0_adc"
        )
    except Exception as e:
        pytest.fail(f"Failed to load dataset: {e}")
        
    if ds.num_frames == 0:
        pytest.skip("No frames found in dataset")
        
    # Get 90th frame (or last available if less)
    frame_idx = 90
    if frame_idx >= ds.num_frames:
        frame_idx = ds.num_frames - 1
        print(f"Frame 90 not available, using frame {frame_idx}")
        
    adc_cube = ds.get_radar_adc_data(frame_idx)

    # 3. Initialize Processors
    virtual_array_processor = VirtualArrayReformatter(
        config_manager=config_manager
    )

    return virtual_array_processor.process(adc_cube)

def test_range_angle_processor(config_manager, adc_cube):
    print("Testing RangeAngleProcessor...")
    processor = RangeAngleProcessor(config_manager)
    
    processor.process(adc_cube, rx_antennas=[0, 1])
    print("RangeAngleProcessor passed.")

def test_micro_doppler_processor(config_manager, adc_cube):
    print("Testing MicroDopplerProcessor...")
    # Test list input for target_ranges
    processor = MicroDopplerProcessor(config_manager, target_ranges=[0.5, 2.0])
    print("MicroDopplerProcessor passed.")

def test_velocity_estimator(config_manager, adc_cube):
    print("Testing VelocityEstimator...")
    # Test list input for valid_angle_range
    processor = VelocityEstimator(
        config_manager, 
        lower_range_bound=0.0,
        upper_range_bound=10.0,
        valid_angle_range=[-1.0, 1.0]
    )
    # Test list input for points
    processor.process(adc_cube, points=[[1.0, 0.0, 0.0]], altitude=0.0, enable_precise_responses=False)
    print("VelocityEstimator passed.")

# def test_strip_map_sar_processor(config_manager, adc_cube):
#     print("Testing StripMapSARProcessor...")
#     # Test list input for az_angle_range_rad
#     processor = StripMapSARProcessor(config_manager, az_angle_range_rad=[-0.5, 0.5])
#     print("StripMapSARProcessor passed.")

# def test_synthetic_array_beamformer_processor(config_manager, adc_cube):
#     print("Testing SyntheticArrayBeamformerProcessor...")
#     # Test list inputs for angle bins and velocity limits
#     processor = SyntheticArrayBeamformerProcessor(
#         config_manager, 
#         az_angle_bins_rad=[-0.5, 0.5], 
#         el_angle_bins_rad=[0.0],
#         min_vel=[-1.0, -1.0, -1.0],
#         max_vel=[1.0, 1.0, 1.0],
#         max_vel_stdev=[0.1, 0.1, 0.1]
#     )
#     # Test list input for current_vel
#     try:
#         processor.process(adc_cube, current_vel=[0.0, 0.0, 0.0])
#     except AttributeError:
#             # Some internal setup might be missing in mock, but we want to check type conversion
#             pass
#     print("SyntheticArrayBeamformerProcessor passed.")

def test_point_cloud_generator_real_data(config_manager,adc_cube):
    print("Testing PointCloudGenerator with Real Data...")
    
    pc_gen = PointCloudGenerator(
        config_manager=config_manager,
        detector_type="range_doppler_detector_2d",
        detector_params={
            "cfar_type": "os_cfar_2d",
            "cfar_params": {"num_train": (8, 4), "num_guard": (2, 1), "rho": 0.75, "alpha": 3.5}
        },
        num_angle_bins=64,
        az_antenna_idxs=[0, 1, 2, 3], # Assuming 4 RX
        el_antenna_idxs=[], # Assuming 2D only for now or no elevation processing if not configured
        shift_az_resp=True,
        shift_el_resp=False
    )
    
    # 4. Process
    try:
        # adc_cube = virtual_array_processor.process(adc_cube)
        point_cloud = pc_gen.process(adc_cube)
        print(f"Generated Point Cloud Shape: {point_cloud.shape}")
        
        # 5. Verify Output
        assert isinstance(point_cloud, np.ndarray)
        assert point_cloud.shape[1] == 4 # x, y, z, vel
        
        # Check if we got any detections (might be empty if no targets)
        if point_cloud.shape[0] > 0:
            print("Detections found!")
        else:
            print("No detections found (this might be expected depending on frame/thresholds)")
            
    except Exception as e:
        pytest.fail(f"Processing failed: {e}")

    print("PointCloudGenerator Real Data Test Passed.")

def test_range_doppler_detector(config_manager, adc_cube):
    """Test the RangeDopplerDetector processor."""
    print("Testing RangeDopplerDetector...")
    from mmwave_radar_processing.processors.range_doppler_detection.range_doppler_detector_2d import RangeDopplerDetector2D
    
    # Initialize processor
    processor = RangeDopplerDetector2D(
        config_manager=config_manager,
        cfar_type="ca_cfar_2d",
        cfar_params={"num_train": (4, 4), "num_guard": (2, 2), "pfa": 1e-5}
    )
    
    # Process
    detections = processor.process(adc_cube)
    
    # Verify outputs
    assert isinstance(detections, np.ndarray)
    # Shape should be (N, 2) where N is number of detections
    assert detections.ndim == 2
    assert detections.shape[1] == 2
    
    # Verify stored responses
    assert processor.rng_dop_resp_raw is not None
    assert processor.rng_dop_resp is not None
    assert isinstance(processor.rng_dop_resp_raw, np.ndarray)
    assert isinstance(processor.rng_dop_resp, np.ndarray)
    
    # Check shapes
    # Raw: (n_rx, n_range_bins, n_chirps)
    # Mag: (n_range_bins, n_chirps)
    n_rx = adc_cube.shape[0]
    n_range_bins = processor.range_bins.shape[0]
    n_chirps = processor.vel_bins.shape[0]
    
    assert processor.rng_dop_resp_raw.shape == (n_rx, n_range_bins, n_chirps)
    assert processor.rng_dop_resp.shape == (n_range_bins, n_chirps)
    
    print(f"RangeDopplerDetector test passed. Found {len(detections)} detections.")
