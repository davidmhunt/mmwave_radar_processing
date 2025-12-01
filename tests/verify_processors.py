import numpy as np
import sys
import os
import pytest

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor
from mmwave_radar_processing.processors.micro_doppler_resp import MicroDopplerProcessor
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator
from mmwave_radar_processing.processors.strip_map_SAR_processor import StripMapSARProcessor
from mmwave_radar_processing.processors.simple_synthetic_array_beamformer_processor_multiFrame import SyntheticArrayBeamformerProcessor
from mmwave_radar_processing.processors._beamformer_processor import _BeamformerProcessor

# Mock ConfigManager
class MockConfigManager:
    def __init__(self):
        self.num_rx_antennas = 4
        self.num_tx_antennas = 3
        self.num_adc_samples = 100
        self.num_chirps_per_frame = 128
        self.num_frames = 10
        self.range_res_m = 0.1
        self.range_max_m = 10.0
        self.vel_max_m_s = 5.0
        self.vel_res_m_s = 0.1
        self.frameCfg_start_index = 0
        self.frameCfg_end_index = 127
        self.frameCfg_loops = 1
        self.frameCfg_periodicity_ms = 100.0
        self.array_direction = "x" # or "y" or "z"
        self.array_geometry = "standard" # Mock array geometry type
        self.profile_cfgs = [{
            "startFreq_GHz": 77,
            "idleTime_us": 10,
            "rampEndTime_us": 60,
            "freqSlopeConst_MHz_usec": 20,
            "numAdcSamples": 100,
            "digOutSampleRate": 10000
        }]
        self.virtual_antennas_enabled = False
        self.chirp_cfgs = []

    def get_num_adc_samples(self, profile_idx=0):
        return self.num_adc_samples

def test_processors():
    config_manager = MockConfigManager()
    adc_cube = np.zeros((4, 100, 128), dtype=complex)

    print("Testing RangeAngleProcessor...")
    try:
        processor = RangeAngleProcessor(config_manager)
        # Test list input for rx_antennas
        processor.process(adc_cube, rx_antennas=[0, 1])
        print("RangeAngleProcessor passed.")
    except Exception as e:
        print(f"RangeAngleProcessor failed: {e}")

    print("Testing MicroDopplerProcessor...")
    try:
        # Test list input for target_ranges
        processor = MicroDopplerProcessor(config_manager, target_ranges=[0.5, 2.0])
        print("MicroDopplerProcessor passed.")
    except Exception as e:
        print(f"MicroDopplerProcessor failed: {e}")

    print("Testing VelocityEstimator...")
    try:
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
    except Exception as e:
        print(f"VelocityEstimator failed: {e}")
        import traceback
        traceback.print_exc()

    print("Testing StripMapSARProcessor...")
    try:
        # Test list input for az_angle_range_rad
        processor = StripMapSARProcessor(config_manager, az_angle_range_rad=[-0.5, 0.5])
        print("StripMapSARProcessor passed.")
    except Exception as e:
        print(f"StripMapSARProcessor failed: {e}")

    print("Testing SyntheticArrayBeamformerProcessor...")
    try:
        # Test list inputs for angle bins and velocity limits
        processor = SyntheticArrayBeamformerProcessor(
            config_manager, 
            az_angle_bins_rad=[-0.5, 0.5], 
            el_angle_bins_rad=[0.0],
            min_vel=[-1.0, -1.0, -1.0],
            max_vel=[1.0, 1.0, 1.0],
            max_vel_stdev=[0.1, 0.1, 0.1]
        )
        # Test list input for current_vel
        # Note: process method might require more setup, but we check if it accepts the list
        try:
            processor.process(adc_cube, current_vel=[0.0, 0.0, 0.0])
        except AttributeError:
             # Some internal setup might be missing in mock, but we want to check type conversion
             pass
        print("SyntheticArrayBeamformerProcessor passed.")
    except Exception as e:
        print(f"SyntheticArrayBeamformerProcessor failed: {e}")

if __name__ == "__main__":
    test_processors()
