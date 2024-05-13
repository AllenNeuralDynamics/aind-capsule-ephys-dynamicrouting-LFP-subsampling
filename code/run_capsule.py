import pathlib
import spikeinterface as si
import spikeinterface.preprocessing as sip
import utils

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')
# TODO: move to input args
TEMPORAL_SUBSAMPLE_FACTOR = 2 
SPAITIAL_CHANNEL_SUBSAMPLE_FACTOR = 4
IS_AVERAGE_DIRECTION = False

def run():

    session_id = utils.parse_session_id()

    zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys_compressed/*-LFP.zarr'))

    for lfp_path in zarr_lfp_paths:
        recording = si.read_zarr(lfp_path)
        if IS_AVERAGE_DIRECTION: # TODO add average direction
            pass

        channel_ids = recording.get_channel_ids()
        channel_ids_to_keep = [channel_ids[i] for i in range(0, len(channel_ids), SPAITIAL_CHANNEL_SUBSAMPLE_FACTOR)] 
        channel_ids_to_remove = [channel_ids[i] for i in range(0, len(channel_ids)) if channel_ids[i] not in channel_ids_to_keep]

        recording_channels_removed = recording.remove_channels(channel_ids_to_remove)
        resampled_recording = sip.resample(recording_channels_removed, int(recording.sampling_frequency / RESAMPLE_RATE))

        assert (resampled_recording.get_num_channels() == recording.get_num_channels() / SPAITIAL_CHANNEL_SUBSAMPLE_FACTOR
        ), f"Applying {SPAITIAL_CHANNEL_SUBSAMPLE_FACTOR} channel stride resulted in mismatch downsampling {recording.get_num_channels()} channels and {resampled_recording.get_num_channels()} channels"

        probe = lfp_path.stem[lfp.stem.index('Probe'):]
        result_output_path = (RESULTS_PATH / f'{session_id}_{probe}.zarr')
        resampled_recording.save_to_zarr(result_output_path, overwrite=True)
        utils.check_saved_subsampled_lfp_result(result_output_path, lfp_path, TEMPORAL_SUBSAMPLE_FACTOR, SPAITIAL_CHANNEL_SUBSAMPLE_FACTOR)

if __name__ == "__main__": run()