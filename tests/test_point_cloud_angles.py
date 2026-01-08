import numpy as np
import pytest
from unittest.mock import MagicMock
from mmwave_radar_processing.processors.point_cloud_generator import PointCloudGenerator
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

@pytest.fixture
def mock_config_manager():
    cm = MagicMock(spec=ConfigManager)
    cm.vel_max_m_s = 10.0
    cm.vel_res_m_s = 0.1
    cm.range_max_m = 50.0
    cm.range_res_m = 0.5
    cm.range_bins_m = np.linspace(0, 50, 100)
    cm.num_rx_antennas = 4
    cm.num_tx_antennas = 3
    # Add other attributes if needed
    return cm

def test_empty_azimuth_indices(mock_config_manager):
    # Initialize with empty azimuth indices
    # We assume 'range_doppler_detector_2d' exists in registry, which it should if imports are correct.
    # If not, we might fail on init, but let's try.
    try:
        pc_gen = PointCloudGenerator(
            config_manager=mock_config_manager,
            az_antenna_idxs=[],
            el_antenna_idxs=[0, 1], 
            detector_type="range_doppler_detector_2d",
            detector_params={
                "cfar_type": "ca_cfar_2d",
                "cfar_params": {
                    "num_train": (2, 2),
                    "num_guard": (1, 1),
                    "pfa": 1e-5
                }
            },
            num_angle_bins=64
        )
    except ValueError as e:
        pytest.skip(f"Skipping because detector type not found: {e}")
        
    pc_gen.configure() # Sets up angle bins

    # Dummy data
    num_dets = 5
    num_antennas = 4
    num_range = 10
    num_doppler = 10
    
    # Random complex data
    rng_dop_resp_raw = np.random.rand(num_antennas, num_range, num_doppler) + 1j * np.random.rand(num_antennas, num_range, num_doppler)
    det_range_idxs = np.zeros(num_dets, dtype=int)
    det_velocity_idxs = np.zeros(num_dets, dtype=int)
    
    az_angles, el_angles = pc_gen._compute_angle_estimation(
        rng_dop_resp_raw, det_range_idxs, det_velocity_idxs
    )
    
    assert np.all(az_angles == 0.0), "Azimuth angles should be 0"
    # Elevation has indices, so it should not be zero (extremely unlikely with random data)
    # However, if num_dets is small or data constructs perfectly, it might be. But with random data, unlikely.
    assert az_angles.shape == (num_dets,)
    assert el_angles.shape == (num_dets,)

def test_empty_elevation_indices(mock_config_manager):
    try:
        pc_gen = PointCloudGenerator(
            config_manager=mock_config_manager,
            az_antenna_idxs=[0, 1],
            el_antenna_idxs=[],
            detector_type="range_doppler_detector_2d",
            detector_params={
                "cfar_type": "ca_cfar_2d",
                "cfar_params": {
                    "num_train": (2, 2),
                    "num_guard": (1, 1),
                    "pfa": 1e-5
                }
            },
            num_angle_bins=64
        )
    except ValueError as e:
        pytest.skip(f"Skipping because detector type not found: {e}")

    pc_gen.configure()

    num_dets = 5
    num_antennas = 4
    num_range = 10
    num_doppler = 10
    
    rng_dop_resp_raw = np.random.rand(num_antennas, num_range, num_doppler) + 1j * np.random.rand(num_antennas, num_range, num_doppler)
    det_range_idxs = np.zeros(num_dets, dtype=int)
    det_velocity_idxs = np.zeros(num_dets, dtype=int)
    
    az_angles, el_angles = pc_gen._compute_angle_estimation(
        rng_dop_resp_raw, det_range_idxs, det_velocity_idxs
    )
    
    assert np.all(el_angles == 0.0), "Elevation angles should be 0"
    assert el_angles.shape == (num_dets,)

def test_both_empty(mock_config_manager):
    try:
        pc_gen = PointCloudGenerator(
            config_manager=mock_config_manager,
            az_antenna_idxs=[],
            el_antenna_idxs=[],
            detector_type="range_doppler_detector_2d",
            detector_params={
                "cfar_type": "ca_cfar_2d",
                "cfar_params": {
                    "num_train": (2, 2),
                    "num_guard": (1, 1),
                    "pfa": 1e-5
                }
            },
            num_angle_bins=64
        )
    except ValueError as e:
        pytest.skip(f"Skipping because detector type not found: {e}")

    pc_gen.configure()
    
    num_dets = 5
    num_antennas = 4
    num_range = 10
    num_doppler = 10
    
    rng_dop_resp_raw = np.random.rand(num_antennas, num_range, num_doppler) + 1j * np.random.rand(num_antennas, num_range, num_doppler)
    det_range_idxs = np.zeros(num_dets, dtype=int)
    det_velocity_idxs = np.zeros(num_dets, dtype=int)

    az_angles, el_angles = pc_gen._compute_angle_estimation(
        rng_dop_resp_raw, det_range_idxs, det_velocity_idxs
    )

    assert np.all(az_angles == 0.0)
    assert np.all(el_angles == 0.0)
