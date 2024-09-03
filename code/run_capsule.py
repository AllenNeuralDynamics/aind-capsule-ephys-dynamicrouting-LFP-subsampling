"""
Capsule that performs temporal and spatial subsampling of LFP data. Default behavior ported from allen sdk.
Temporal Subsample by 2 and Spatial Channel Subsample by taking every 4th channel. Can overwrite these with input args
Saves output to zarr for each probe
"""

import pathlib
import spikeinterface as si
import spikeinterface.preprocessing as spre
import utils
import xml.etree.ElementTree as et
import argparse
import numpy as np
import zarr
import concurrent.futures as cf
import npc_session
   
DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

parser = argparse.ArgumentParser()
parser.add_argument('--lfp_subsampling_temporal_factor', help='Ratio of input samples to output samples in time. Default is 2', default=2)
parser.add_argument('--lfp_subsampling_spatial_factor', help='Controls number of channels to skip in spatial subsampling. Default is 4', default=4)
parser.add_argument('--lfp_highpass_cutoff', help='Cutoff frequency for highpass filter to apply. Default is 0.1', default=0.1)


def run():
    args = parser.parse_args()
    TEMPORAL_SUBSAMPLE_FACTOR = int(args.lfp_subsampling_temporal_factor)
    SPATIAL_CHANNEL_SUBSAMPLE_FACTOR = int(args.lfp_subsampling_spatial_factor)
    HIGHPASS_FILTER_FREQ_MIN = float(args.lfp_highpass_cutoff)

    raw_session_path = tuple(DATA_PATH.glob('*'))
    if not raw_session_path:
        raise FileNotFoundError('No raw data asset attached')
    
    if len(raw_session_path) > 1:
        raise ValueError('More than one raw data asset is attached. Check assets and remove irrelevant ones')

    session_id = npc_session.extract_aind_session_id(raw_session_path[0].stem)

    settings_xml_path =  tuple(DATA_PATH.glob('*/ecephys_clipped/*/*.xml'))
    if not settings_xml_path:
        raise FileNotFoundError(f'No settings xml file in ecephys clipped folder for session {session_id}')

    settings_xml_tree = et.parse(settings_xml_path[0].as_posix())
    utils.save_settings_xml(settings_xml_tree, session_id)

    zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys_compressed/*-LFP.zarr'))
    if not zarr_lfp_paths:
        raise FileNotFoundError(f'No compressed lfp data found for session {session_id}')

    print(f'Starting LFP Subsampling with parameters: temporal factor {TEMPORAL_SUBSAMPLE_FACTOR}, spatial factor {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR}, highpass filter frequency {HIGHPASS_FILTER_FREQ_MIN}')
    lfp_threads_info = []
    for lfp_path in zarr_lfp_paths:
        probe = f'Probe{npc_session.ProbeRecord(lfp_path)}'

        raw_lfp_recording = si.read_zarr(lfp_path)
        channel_ids = raw_lfp_recording.get_channel_ids()

        print(f'Starting LFP subsampling for session {session_id} and probe {probe}')

        # TODO: add surface channel rereferencing
        channel_ids_to_keep = [channel_ids[i] for i in range(0, len(channel_ids), SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)] 

        recording_spatial_subsampled = raw_lfp_recording.channel_slice(channel_ids_to_keep)
        recording_spatial_time_subsampled = spre.resample(recording_spatial_subsampled, int(raw_lfp_recording.sampling_frequency / TEMPORAL_SUBSAMPLE_FACTOR))

        # might run into rounding issues checking shapes
        assert (len(recording_spatial_time_subsampled.get_times()) == int(len(raw_lfp_recording.get_times()) / TEMPORAL_SUBSAMPLE_FACTOR)
        ), f"Applying {TEMPORAL_SUBSAMPLE_FACTOR} temporal factor resulted in mismatch downsampling. Got {len(recording_spatial_time_subsampled.get_times())} time samples given {len(recording.get_times())} raw time samples and factor {TEMPORAL_SUBSAMPLE_FACTOR}"
        assert (recording_spatial_time_subsampled.get_num_channels() == int(raw_lfp_recording.get_num_channels() / SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)
        ), f"Applying {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR} channel stride resulted in mismatch downsampling {recording.get_num_channels()} channels and {recording_spatial_time_subsampled.get_num_channels()} channels"

        filtered_recording = spre.highpass_filter(recording_spatial_time_subsampled, freq_min=HIGHPASS_FILTER_FREQ_MIN)
        result_output_path = (RESULTS_PATH / f'{session_id}_{probe}')
        if not result_output_path.exists():
            result_output_path.mkdir()

        lfp_threads_info.append((probe, filtered_recording, result_output_path))
        print(f'Finished LFP subsampling for session {session_id} and probe {probe}')
    
    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for lfp_thread_info in lfp_threads_info:
            probe, subsampled_recording, result_output_path = lfp_thread_info

            futures.append(executor.submit(utils.save_lfp_to_zarr, result_output_path=result_output_path, subsampled_recording=subsampled_recording,
                                            probe=probe, session_id=session_id))
        
        for future in cf.as_completed(futures):
            print(future.result())
            
if __name__ == "__main__": 
    run()