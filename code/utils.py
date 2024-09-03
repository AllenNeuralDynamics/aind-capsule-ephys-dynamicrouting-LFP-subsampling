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
    subsampled_recording.save_to_zarr(result_output_path / f'{probe}_lfp_subsampled')
    zarr.save((result_output_path / f'{probe}_lfp_time_samples.zarr').as_posix(), subsampled_recording.get_times())
    zarr.save((result_output_path / f'{probe}_lfp_selected_channels.zarr').as_posix(), subsampled_recording.get_channel_ids())

    return f'Finished saving LFP subsampling result for session {session_id} and probe {probe}'

if __name__ == '__main__':
    import doctest
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    