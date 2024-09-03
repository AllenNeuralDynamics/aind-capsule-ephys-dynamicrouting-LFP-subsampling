import json
import npc_session
import zarr
import spikeinterface.preprocessing as spre
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

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

    return f'Finished saving LFP subsampling result for session {session_id} and probe {probe}'

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
    