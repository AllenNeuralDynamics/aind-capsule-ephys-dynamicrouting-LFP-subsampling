import json
import npc_session
import zarr
import spikeinterface.preprocessing as spre
import pathlib
import numpy as np
import datetime
import matplotlib.pyplot as plt

from aind_data_schema.core.data_description import (
    Organization,
    Modality,
    Modality,
    Platform,
    Funding,
    DataLevel,
)
from aind_data_schema_models.pid_names import PIDName

DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

def parse_session_id() -> str:
    """
    parses the session id following the aind format, test assumes that data asset SmartSPIM_695464_2023-10-18_20-30-30 is attached
    >>> parse_session_id()
    'SmartSPIM_695464_2023-10-18_20-30-30'
    """
    session_paths = tuple(DATA_PATH.glob('*'))
    print(session_paths)
    if not session_paths:
        raise FileNotFoundError('No session data assets attached')
    
    session_id = None
    for session_path in session_paths:
        try: # avoid parsing model folder, better way to do this?
            session_id = npc_session.parsing.extract_aind_session_id(session_path.stem)
        except ValueError:
            pass

    if session_id is None:
        raise FileNotFoundError('No data asset attached that follows aind session format')
    
    return session_id

def get_data_description_dict() -> dict:
    session_id = parse_session_id()

    data_description_dict = {}
    data_description_dict["creation_time"] = datetime.datetime.now()
    data_description_dict["name"] = session_id
    data_description_dict["institution"] = Organization.AIND
    data_description_dict["data_level"] = DataLevel.DERIVED
    data_description_dict["investigators"] = [PIDName(name="Unknown")]
    data_description_dict["funding_source"] = [Funding(funder=Organization.AI)]
    data_description_dict["modality"] = [Modality.ECEPHYS]
    data_description_dict["platform"] = Platform.ECEPHYS
    data_description_dict["subject_id"] = str(npc_session.SessionRecord(session_id).subject)
    
    return data_description_dict

def get_processing_dict(start_date_time: datetime.datetime, end_date_time: datetime.datetime, parameters: dict) -> dict:
    data_processing_dict = {}
    data_processing_dict["name"] = "Other"
    data_processing_dict["software_version"] = "0.1.0"
    data_processing_dict["start_date_time"] = str(start_date_time)
    data_processing_dict["end_date_time"] = str(end_date_time)
    data_processing_dict["input_location"] = DATA_PATH.as_posix()
    data_processing_dict["output_location"] = RESULTS_PATH.as_posix()
    data_processing_dict["code_url"] = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-dynamicrouting-LFP-subsampling"
    data_processing_dict["notes"] = "LFP Subsampling"
    data_processing_dict["parameters"] = {}
    data_processing_dict["outputs"] = {}

    return data_processing_dict

def save_lfp_to_zarr(result_output_path: pathlib.Path, subsampled_recording: spre.HighpassFilterRecording, probe:str, session_id: str) -> None:
    subsampled_recording.save_to_zarr()
    print(f'Started saving subsampled lfp for session {session_id} and probe {probe}')
    subsampled_recording.save_to_zarr(result_output_path / f'{probe}_lfp_subsampled')

    return f'Finished saving LFP subsampling result for session {session_id} and probe {probe}'

if __name__ == '__main__':
    import doctest
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
    