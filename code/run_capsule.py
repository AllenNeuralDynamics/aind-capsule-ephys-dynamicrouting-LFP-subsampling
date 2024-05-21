"""
Capsule that performs temporal and spatial subsampling of LFP data. Default behavior ported from allen sdk.
Temporal Subsample by 2 and Spatial Channel Subsample by taking every 4th channel. Can overwrite these with input args
"""

import pathlib
import spikeinterface as si
import spikeinterface.preprocessing as sip
import utils
import xml.etree.ElementTree as et
import argparse
import numpy as np
import zarr
import concurrent.futures as cf
   
DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

parser = argparse.ArgumentParser()
parser.add_argument('--temporal_factor', help='Ratio of input samples to output samples in time', default=2)
parser.add_argument('--spatial_factor', help='Distance between channels to keep', default=4)
parser.add_argument('--use_avg_direction', help='Apply average across channels with same position in y direction', default='False')

def save_settings_xml(settings_xml_tree: et.ElementTree(), session_id: str) -> None:
    settings_xml_root = settings_xml_tree.getroot()
    settings_xml_string = et.tostring(settings_xml_root)
    with open(RESULTS_PATH / f'{session_id}_settings.xml', 'wb') as f:
        f.write(settings_xml_string)

def save_lfp_to_zarr(result_output_path: pathlib.Path, lfp_path: pathlib.Path, resampled_recording: sip.ResampleRecording,
                    temporal_subsampling_factor: int, spatial_channel_subsampling_factor: int, 
                    raw_times: np.ndarray, subsampled_times: np.ndarray, probe:str, session_id: str) -> None:
    print(f'Started saving subsampled lfp for session {session_id} and probe {probe}')
    resampled_recording.save_to_zarr(result_output_path / f'{probe}_lfp_subsampled', overwrite=True)
    zarr.save((result_output_path / f'{probe}_lfp_timestamps.zarr').as_posix(), resampled_recording.get_times())


    utils.check_saved_subsampled_lfp_result(result_output_path / f'{probe}_lfp_subsampled.zarr', 
                                            lfp_path, temporal_subsampling_factor, spatial_channel_subsampling_factor)
        
    utils.plot_raw_and_subsampled_lfp(result_output_path / f'{probe}_lfp_subsampled.zarr', 
                                    lfp_path, raw_times, subsampled_times, 
                                    temporal_subsampling_factor, spatial_channel_subsampling_factor)

    return f'Finished saving and checking subsampling result for session {session_id} and probe {probe}'

def run():
    args = parser.parse_args()
    TEMPORAL_SUBSAMPLE_FACTOR = args.temporal_factor
    SPATIAL_CHANNEL_SUBSAMPLE_FACTOR = args.spatial_factor
    APPLY_AVERAGE_DIRECTION = args.use_avg_direction

    session_id = utils.parse_session_id()

    settings_xml_path =  tuple(DATA_PATH.glob('*/ecephys_clipped/*/*.xml'))
    if not settings_xml_path:
        raise FileNotFoundError(f'No settings xml file in ecephys clipped folder for session {session_id}')

    settings_xml_tree = et.parse(settings_xml_path[0].as_posix())
    save_settings_xml(settings_xml_tree, session_id)

    zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys_compressed/*-LFP.zarr'))
    if not zarr_lfp_paths:
        raise FileNotFoundError(f'No compressed lfp data found for session {session_id}')

    print(f'Starting LFP Subsampling with parameters: temporal factor {TEMPORAL_SUBSAMPLE_FACTOR}, spatial factor {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR}, apply average direction {APPLY_AVERAGE_DIRECTION}')
    lfp_threads_info = []
    for lfp_path in zarr_lfp_paths:
        probe = lfp_path.stem[lfp_path.stem.index('Probe'):]
        print(f'Starting LFP subsampling for session {session_id} and probe {probe}')
        recording = si.read_zarr(lfp_path)
        
        channel_ids = recording.get_channel_ids()
        channel_ids_to_keep = [channel_ids[i] for i in range(0, len(channel_ids), SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)] 
        channel_ids_to_remove = [channel_ids[i] for i in range(0, len(channel_ids)) if channel_ids[i] not in channel_ids_to_keep]

        recording_channels_removed = recording.remove_channels(channel_ids_to_remove)
        resampled_recording = sip.resample(recording_channels_removed, int(recording.sampling_frequency / TEMPORAL_SUBSAMPLE_FACTOR))

        if APPLY_AVERAGE_DIRECTION == "True":
            resampled_recording = sip.AverageAcrossDirectionRecording(resampled_recording)

        assert (len(resampled_recording.get_times()) == int(len(recording.get_times()) / TEMPORAL_SUBSAMPLE_FACTOR)
        ), f"Applying {TEMPORAL_SUBSAMPLE_FACTOR} temporal factor resulted in mismatch downsampling. Got {len(resampled_recording.get_times())} time samples given {len(recording.get_times())} raw time samples and factor {TEMPORAL_SUBSAMPLE_FACTOR}"
        assert (resampled_recording.get_num_channels() == int(recording.get_num_channels() / SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)
        ), f"Applying {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR} channel stride resulted in mismatch downsampling {recording.get_num_channels()} channels and {resampled_recording.get_num_channels()} channels"

        result_output_path = (RESULTS_PATH / f'{session_id}_{probe}')
        if not result_output_path.exists():
            result_output_path.mkdir()

        lfp_threads_info.append((probe, recording, resampled_recording, result_output_path))
        print(f'Finished LFP subsampling for session {session_id} and probe {probe}')
    
    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for lfp_thread_info in lfp_threads_info:
            probe, recording, resampled_recording, result_output_path = lfp_thread_info

            futures.append(executor.submit(save_lfp_to_zarr, result_output_path=result_output_path, lfp_path=lfp_path, resampled_recording=resampled_recording,
                            temporal_subsampling_factor=TEMPORAL_SUBSAMPLE_FACTOR, spatial_channel_subsampling_factor=SPATIAL_CHANNEL_SUBSAMPLE_FACTOR,
                            raw_times=recording.get_times(), subsampled_times=resampled_recording.get_times(), probe=probe, session_id=session_id))
        
        for future in cf.as_completed(futures):
            print(future.result())
            
if __name__ == "__main__": 
    run()