import json
import npc_session
import zarr
import spikeinterface.preprocessing as spre
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

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

def save_settings_xml(settings_xml_tree: et.ElementTree(), session_id: str) -> None:
    settings_xml_root = settings_xml_tree.getroot()
    settings_xml_string = et.tostring(settings_xml_root)
    with open(RESULTS_PATH / f'{session_id}_settings.xml', 'wb') as f:
        f.write(settings_xml_string)

def save_lfp_to_zarr(result_output_path: pathlib.Path, subsampled_recording: spre.HighpassFilterRecording, probe:str, session_id: str) -> None:
    print(f'Started saving subsampled lfp for session {session_id} and probe {probe}')
    subsampled_recording.save_to_zarr(result_output_path / f'{probe}_lfp_subsampled', overwrite=True)
    zarr.save((result_output_path / f'{probe}_lfp_time_samples.zarr').as_posix(), subsampled_recording.get_times())
    zarr.save((result_output_path / f'{probe}_lfp_selected_channels.zarr').as_posix(), subsampled_recording.get_channel_ids())

    return f'Finished saving and checking subsampling result for session {session_id} and probe {probe}'

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
    