import json
import os
import sys
import numpy as np
import scipy.constants as constants
from collections import OrderedDict


# invalid configuration exception
class InvalidConfiguration(Exception):
    pass

class ConfigNotLoaded(Exception):
    pass

class ConfigManager:
    def __init__(self):
        
        #channel_config
        self.channelCfg_tx_chan_enabled:int = 0
        self.channelCfg_rx_chan_enabled:int = 0
        self.channelCfg_cascading:bool = 0

        #adcCfg
        self.adcCfg_num_adc_bits:int = 0
        self.adcCfg_adcOutputFmt:int = 0

        #adcbufCfg
        self.adcbufCfg_sub_frame_idx:int = 0
        self.adcbufCfg_adc_output_fmt:int = 0
        self.adcbufCfg_sample_swap:bool = False
        self.adcbufCfg_channel_interleave:bool = False
        self.adcbufCfg_chirp_threshold:int = 1

        #profileCfg
        self.profile_cfgs:list = [
            {
            "profileId": -1,
            "startFreq_GHz": 77.0,
            "idleTime_us": 0.0,
            "adcStartTime_us": 0.0,
            "rampEndTime_us": 0.0,
            "txOutPower": 0.0,
            "txPhaseShifter": 0.0,
            "freqSlope_MHz_us": 0.0,
            "txStartTime_us": 0.0,
            "adcSamples": 0,
            "sampleRate_kSps": 0,
            "hpfCornerFreq1": 0.0,
            "hpfCornerFreq2": 0.0,
            "rxGain_dB": 0.0
            }
        ]

        #chirpCfg
        self.chirp_cfgs:list = [
            {"startIndex": -1,
            "endIndex": -1,
            "profile": 0,
            "startFreqVariation_Hz": 0.0,
            "freqSlopVariation_MHz_us": 0.0,
            "idleTimeVariation_us": 0.0,
            "ADCStartTimeVariation_us": 0.0,
            "txMask": 0,
            }
        ]

        #frameCfg
        self.frameCfg_start_index:int = 0
        self.frameCfg_end_index:int = 0
        self.frameCfg_loops:int = 0
        self.frameCfg_frames:int = 0
        self.frameCfg_periodicity_ms:float = 0
        self.frameCfg_hardware_trigger_enabled:bool = False
        self.frameCfg_trigger_delay_ms:float = 0.0

        #range parameters
        self.range_res_m:float = 0.0
        self.range_bin_size_m:float = 0.0
        self.range_max_m:float = 0.0
        self.range_bins_m:np.ndarray = np.empty(shape=0,dtype=float)

        #velocity parameters
        self.vel_res_m_s:float = 0.0
        self.vel_max_m_s:float = 0.0
        self.vel_bins_m_s:np.ndarray = np.empty(shape=0,dtype=float)

        #angle parameters
        self.num_tx_antennas:int = 3
        self.num_rx_antennas:int = 4
        self.virtual_antennas_enabled:bool = False
        self.angle_bins:np.ndarray = np.empty(shape=0,dtype=float)

        # status flags
        self.config_loaded = False

        return

    ####################################################################
    #Computing radar performance
    ####################################################################

    def compute_radar_perforance(self):
        
        pass

    def _compute_range_performance(self,profile_idx:int = 0):

        num_range_bins = np.power(2,np.ceil(np.log2(
            self.get_num_adc_samples(profile_idx)
        )))

        self.range_res_m = (constants.c * \
                self.get_adc_sample_rate_kSps(profile_idx) * 1e3) / \
                (2 * self.get_chirp_slope_MHz_us(profile_idx) * \
                    (1e6/1e-6) * self.get_num_adc_samples(profile_idx))
        
        self.range_bin_size_m = (constants.c * \
                self.get_adc_sample_rate_kSps(profile_idx) * 1e3) / \
                (2 * self.get_chirp_slope_MHz_us(profile_idx) * \
                    (1e6/1e-6) * num_range_bins)
        
        self.range_max_m = (constants.c * self.get_adc_sample_rate_kSps(profile_idx) * 1e3) / \
                (2 * self.get_chirp_slope_MHz_us(profile_idx) * (1e6/1e-6))
        
        self.range_bins_m = np.arange(
            start=0,
            stop=self.range_max_m,
            step=self.range_bin_size_m
        )

        return
    
    def compute_vel_performance(self):

        pass

    def compute_angular_performance(self):
        pass

    ####################################################################
    #Helper Functions
    ####################################################################

    def get_adc_sample_rate_kSps(self,profile_idx:int=0)->int:

        return self.profile_cfgs[profile_idx]["sampleRate_kSps"]

    def get_num_adc_samples(self,profile_idx:int=0)->int:

        return self.profile_cfgs[profile_idx]["adcSamples"]
    
    def get_chirp_slope_MHz_us(self,profile_idx:int=0)->float:

        return self.profile_cfgs[profile_idx]["freqSlope_MHz_us"]

    ####################################################################
    #loading the radar configuration
    ####################################################################

    def load_cfg(self,cfg_file_path:str):

        f = open(cfg_file_path)
        for line in f:

            if "%" not in line:
                self._load_cfg_command_from_line(line)
        
        self.config_loaded = True


    ####################################################################
    #loading radar configuration from .cfg file
    ####################################################################
    
    def _load_cfg_command_from_line(self,line:str):

        #split the .cfg file line into key parts
        str_split = line.strip("\n").split(" ")
        key = str_split[0]

        match key:
            case "channelCfg":
                self._load_channelCfg_from_cfg(str_split)
            case "adcCfg":
                self._load_adcCfg_from_cfg(str_split)
            case "adcbufCfg":
                self._load_adcbufCfg_from_cfg(str_split)
            case "profileCfg":
                pass
            case "chirpCfg":
                pass
            case "frameCfg":
                pass
    
    def _load_channelCfg_from_cfg(self, params: list):
        
        #determine the number of Tx antennas enabled
        self.channelCfg_rx_chan_enabled = int(params[1])
        self.num_rx_antennas = bin(self.channelCfg_rx_chan_enabled).count('1')

        #determine the number of Rx antennas enabled
        self.channelCfg_tx_chan_enabled = int(params[2])
        self.num_tx_antennas = bin(self.channelCfg_tx_chan_enabled).count('1')

        self.channelCfg_cascading = int(params[3])

        return

    def _load_adcCfg_from_cfg(self,params: list):

        match int(params[1]):
            case 0:
                self.adcCfg_num_adc_bits = 12
            case 1:
                self.adcCfg_num_adc_bits = 14
            case 2:
                self.adcCfg_num_adc_bits = 16
        
        self.adcCfg_adcOutputFmt = int(params[2])

        return
    
    def _load_adcbufCfg_from_cfg(self, params: list):

        self.adcbufCfg_sub_frame_idx = int(params[1])
        self.adcbufCfg_adc_output_fmt = int(params[2]) #0 for complex, 1 for real
        self.adcbufCfg_sample_swap = False if int(params[3]) == 0 else True
        self.adcbufCfg_channel_interleave = True if int(params[4]) == 0 else False
        self.adcbufCfg_chirp_threshold = int(params[5])

        return

    def _load_profileCfg_from_cfg(self,params:list):

        new_profile:dict = {
            "profileId": int(params[1]),
            "startFreq_GHz": float(params[2]),
            "idleTime_us": float(params[3]),
            "adcStartTime_us": float(params[4]),
            "rampEndTime_us": float(params[5]),
            "txOutPower": float(params[6]),
            "txPhaseShifter": float(params[7]),
            "freqSlope_MHz_us": float(params[8]),
            "txStartTime_us": float(params[9]),
            "adcSamples": int(params[10]),
            "sampleRate_kSps": int(params[11]),
            "hpfCornerFreq1": int(params[12]),
            "hpfCornerFreq2": int(params[13]),
            "rxGain_dB": float(params[14]),
        }

        if self.profile_cfgs[0]["profileID"] == -1:
            self.profile_cfgs[0] = new_profile
        elif new_profile["profileId"] < len(self.profile_cfgs):
            print("cfgManager: attempted to load multiple profiles with the same ID")
        else:
            self.profile_cfgs.append(new_profile)

        return
    
    def _load_chirpCfg_from_cfg(self,params: list):

        new_chirp:dict = {
            "startIndex": int(params[1]),
            "endIndex": int(params[2]),
            "profile": int(params[3]),
            "startFreqVariation_Hz": float(params[4]),
            "freqSlopVariation_MHz_us": float(params[5]),
            "idleTimeVariation_us": float(params[6]),
            "ADCStartTimeVariation_us": float(params[7]),
            "txMask": int(params[8]),
            }
        
        if self.chirp_cfgs[0]["startIndex"] == -1:
            self.chirp_cfgs[0] = new_chirp
        elif new_chirp["startIndex"] < len(self.chirp_cfgs):
            print("cfgManager: attempted to load multiple chirps with the same ID")
        else:
            self.chirp_cfgs.append(new_chirp)
    
    def _load_frame_cfg_from_cfg(self,params:list):

        self.frameCfg_start_index = int(params[1])
        self.frameCfg_end_index = int(params[2])
        self.frameCfg_loops = int(params[3])
        self.frameCfg_frames = int(params[4])
        self.frameCfg_periodicity_ms = float(params[5])
        self.frameCfg_hardware_trigger_enabled = False if int(params[6]) == 1 else True
        self.frameCfg_trigger_delay_ms = float(params[7])