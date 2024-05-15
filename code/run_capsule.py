"""
Capsule that performs temporal and spatial subsampling of LFP data. Default behavior ported from allen sdk.
Temporal Subsample by 2 and Spatial Channel Subsample by taking every 4th channel. Can overwrite these with input args
"""

import pathlib
import spikeinterface as si
import spikeinterface.preprocessing as sip
import utils
import argparse

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

parser = argparse.ArgumentParser()
parser.add_argument('--temporal_factor', help='Value to subsample time dimension of LFP')
parser.add_argument('--spatial_factor', help='Value to subsample channel dimension of LFP')
parser.add_argument('--use_avg_direction', help='whether or not to apply average direction across same channel positions in y direction')

def run():
    args = parser.parse_args()
    
    if args.temporal_factor is not None:
        TEMPORAL_SUBSAMPLE_FACTOR = args.temporal_factor
    else:
        TEMPORAL_SUBSAMPLE_FACTOR = 2 
    
    if args.spatial_factor is not None:
        SPATIAL_CHANNEL_SUBSAMPLE_FACTOR = args.spatial_factor
    else:
        SPATIAL_CHANNEL_SUBSAMPLE_FACTOR = 4
    
    if args.use_avg_direction is not None:
        APPLY_AVERAGE_DIRECTION = args.use_avg_direction
    else:
        APPLY_AVERAGE_DIRECTION = "False"

    session_id = utils.parse_session_id()

    zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys_compressed/*-LFP.zarr'))
    if not zarr_lfp_paths:
        raise FileNotFoundError(f'No compressed lfp data found for session {session_id}')

    for lfp_path in zarr_lfp_paths:
        probe = lfp_path.stem[lfp_path.stem.index('Probe'):]

        print(f'Starting subsampling for session {session_id} and probe {probe}')
        recording = si.read_zarr(lfp_path)

        channel_ids = recording.get_channel_ids()
        channel_ids_to_keep = [channel_ids[i] for i in range(0, len(channel_ids), SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)] 
        channel_ids_to_remove = [channel_ids[i] for i in range(0, len(channel_ids)) if channel_ids[i] not in channel_ids_to_keep]

        recording_channels_removed = recording.remove_channels(channel_ids_to_remove)
        resampled_recording = sip.resample(recording_channels_removed, int(recording.sampling_frequency / TEMPORAL_SUBSAMPLE_FACTOR))

        if APPLY_AVERAGE_DIRECTION == "True":
            resampled_recording = sip.AverageAcrossDirectionRecording(resampled_recording)

        assert (len(resampled_recording.get_times()) == len(recording.get_times()) / TEMPORAL_SUBSAMPLE_FACTOR
        ), f"Applying {TEMPORAL_SUBSAMPLE_FACTOR} temporal factor resulted in mismatch downsampling. Got {len(resampled_recording.get_times())} time samples given {len(recording.get_times())} raw time samples and factor {TEMPORAL_SUBSAMPLE_FACTOR}"
        assert (resampled_recording.get_num_channels() == recording.get_num_channels() / SPATIAL_CHANNEL_SUBSAMPLE_FACTOR
        ), f"Applying {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR} channel stride resulted in mismatch downsampling {recording.get_num_channels()} channels and {resampled_recording.get_num_channels()} channels"

        result_output_path = (RESULTS_PATH / f'{session_id}_{probe}')
        resampled_recording.save_to_zarr(result_output_path, overwrite=True)
        utils.check_saved_subsampled_lfp_result(result_output_path, lfp_path, TEMPORAL_SUBSAMPLE_FACTOR, SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)
        print(f'Finished saving and checking subsampling result for session {session_id} and probe {probe}')

if __name__ == "__main__": 
    run()