import json
import npc_session
import zarr
import pathlib
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('results')

def plot_raw_and_subsampled_lfp(result_output_path: pathlib.Path, input_data_path: pathlib.Path, raw_times: np.ndarray,
        subsampled_times: np.ndarray, temporal_subsampling_factor: int, spatial_channel_subsampling_factor: int) -> None:
    raw_lfp = zarr.open(input_data_path, mode='r')
    subsampled_lfp = zarr.open(result_output_path, mode='r')

    raw_traces = raw_lfp['traces_seg0']
    raw_channel_ids = raw_lfp['channel_ids'][:].tolist()
    subsampled_traces = subsampled_lfp['traces_seg0']
    subsampled_channel_ids = subsampled_lfp['channel_ids'][:].tolist()

    selected_channel_index = np.random.randint(0, len(subsampled_channel_ids))
    selected_channel = subsampled_channel_ids[selected_channel_index]
    raw_channel_index = raw_channel_ids.index(selected_channel)

    assert(raw_channel_ids[raw_channel_index] == selected_channel 
    ), f"Different channel ids for selected channel {selected_channel} when trying to get id from raw lfp channel ids"

    start_frame = 2000
    end_frame = 6000
    raw_times_interval = raw_times[start_frame:end_frame]
    subsampled_times_within = np.argwhere((subsampled_times >= raw_times_interval[0]) & (subsampled_times <= raw_times_interval[-1]))
    subsampled_times_to_plot = subsampled_times[subsampled_times_within]
    subsampled_times_to_plot = subsampled_times_to_plot[:int((end_frame - start_frame) / temporal_subsampling_factor)]

    plt.plot(raw_times_interval, raw_traces[start_frame:end_frame, raw_channel_index])
    plt.plot(subsampled_times_to_plot.T[0], 
                subsampled_traces[int(start_frame/temporal_subsampling_factor):int(end_frame/temporal_subsampling_factor), selected_channel_index])
    plt.legend(['Raw LFP', 'Subsampled LFP'])
    plt.title(f'Raw and Subsampled LFP')
    plt.xlabel('Time (s)')
    plt.savefig(result_output_path.parent / 'LFP_plot.png')

def check_saved_subsampled_lfp_result(result_output_path: pathlib.Path, input_data_path: pathlib.Path, temporal_subsampling_factor: int, 
                                    spatial_channel_subsampling_factor: int) -> None:
    raw_lfp = zarr.open(input_data_path, mode='r')['traces_seg0']
    subsampled_lfp = zarr.open(result_output_path, mode='r')['traces_seg0']

    assert (subsampled_lfp.shape[0] == int(raw_lfp.shape[0] / temporal_subsampling_factor)
    ), f"Temporal subsampling mismatch after saving to zarr. Got subsampled time samples {subsampled_lfp.shape[0]} given raw time samples {raw_lfp.shape[0]} and temporal factor {temporal_subsampling_factor}"

    assert (subsampled_lfp.shape[1] == int(raw_lfp.shape[1] / spatial_channel_subsampling_factor)
    ), f"Spatial subsampling mismatch after saving to zarr. Got subsampled number of channels {subsampled_lfp.shape[1]} given raw number of channels {raw_lfp.shape[1]} and spaitial factor {spatial_channel_subsampling_factor}"

def parse_session_id() -> str:
    """
    parses the session id following the aind format, test assumes that data asset SmartSPIM_695464_2023-10-18_20-30-30 is attached
    >>> parse_session_id()
    'SmartSPIM_695464_2023-10-18_20-30-30'
    """
    session_paths = tuple(DATA_PATH.glob('*'))
    print(session_paths)
    
    session_id = None
    for session_path in session_paths:
        try:
            session_id = npc_session.parsing.extract_aind_session_id(session_path.stem)
        except ValueError:
            pass

    if session_id is None:
        raise FileNotFoundError('No data asset attached that follows aind session format')
    
    return session_id

if __name__ == '__main__':
    import doctest
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    