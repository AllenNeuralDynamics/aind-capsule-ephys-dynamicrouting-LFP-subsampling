"""
Capsule that performs temporal and spatial subsampling of LFP data.
Saves output to zarr files for each probe
"""

import pathlib
import spikeinterface as si
import spikeinterface.preprocessing as spre
import utils
import argparse
import numpy as np
import shutil
import zarr
import concurrent.futures as cf
import npc_session
import npc_sessions
import datetime

from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
)
from aind_data_schema.core.processing import DataProcess, Processing, PipelineProcess
   
DATA_PATH = pathlib.Path('/data')
RESULTS_PATH = pathlib.Path('/results')

parser = argparse.ArgumentParser()
parser.add_argument('--lfp_subsampling_temporal_factor', help='Ratio of input samples to output samples in time. Default is 2', default=2)
parser.add_argument('--lfp_subsampling_spatial_factor', help='Controls number of channels to skip in spatial subsampling. Default is 4', default=4)
parser.add_argument('--lfp_highpass_cutoff', help='Cutoff frequency for highpass filter to apply. Default is 0.1', default=0.1)

def run():
    start_date_time = datetime.datetime.now()
    args = parser.parse_args()
    TEMPORAL_SUBSAMPLE_FACTOR = int(args.lfp_subsampling_temporal_factor)
    SPATIAL_CHANNEL_SUBSAMPLE_FACTOR = int(args.lfp_subsampling_spatial_factor)
    HIGHPASS_FILTER_FREQ_MIN = float(args.lfp_highpass_cutoff)

    processing_parameters = {'Temporal_subsampling_factor': TEMPORAL_SUBSAMPLE_FACTOR, 'Spatial_subsampling_factor': SPATIAL_CHANNEL_SUBSAMPLE_FACTOR}
    session_json_path = tuple(utils.DATA_PATH.glob('*/session.json'))
    procedures_json_path = tuple(utils.DATA_PATH.glob('*/procedures.json'))
    subject_json_path = tuple(utils.DATA_PATH.glob('*/subject.json'))

    if not session_json_path:
        print('No session json found')
    else:
        shutil.copy(session_json_path[0].as_posix(), (utils.RESULTS_PATH / 'session.json').as_posix())

    if not subject_json_path:
        print('No subject json found')
    else:
        shutil.copy(subject_json_path[0].as_posix(), (utils.RESULTS_PATH / 'subject.json').as_posix())

    if not procedures_json_path:
        print('No procedures json found')
    else:
        shutil.copy(procedures_json_path[0].as_posix(), (utils.RESULTS_PATH / 'procedures.json').as_posix())

    data_description_dict = utils.get_data_description_dict()
    data_description = DataDescription(**data_description_dict)

    derived_data_description = DerivedDataDescription.from_data_description(
        data_description=data_description, process_name="LFPSubsampled"
    )
    with (utils.RESULTS_PATH / "data_description.json").open("w") as f:
        f.write(derived_data_description.model_dump_json(indent=3))

    raw_session_path = tuple(DATA_PATH.glob('*'))
    if not raw_session_path:
        raise FileNotFoundError('No data asset attached')
    
    if len(raw_session_path) > 1:
        raise ValueError('More than one data asset is attached. Check assets and remove irrelevant ones')

    session_id = npc_session.extract_aind_session_id(raw_session_path[0].stem)
    is_duragel = utils.is_duragel(session_id)

    settings_xml_path =  tuple(DATA_PATH.glob('*/ecephys_clipped/*/*.xml'))
    if not settings_xml_path:
        raise FileNotFoundError(f'No settings xml file in ecephys clipped folder for session {session_id}')

    shutil.copy(settings_xml_path[0], utils.RESULTS_PATH / f'{session_id}_settings.xml')

    zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys/ecephys_compressed/*-LFP.zarr'))
    if not zarr_lfp_paths:
        zarr_lfp_paths = tuple(DATA_PATH.glob('*/ecephys_compressed/*-LFP.zarr')) # try old way and then raise
        if not zarr_lfp_paths:
            raise FileNotFoundError(f'No compressed lfp data found for session {session_id}')

    print(f'Starting LFP Subsampling with parameters: temporal factor {TEMPORAL_SUBSAMPLE_FACTOR}, spatial factor {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR}, highpass filter frequency {HIGHPASS_FILTER_FREQ_MIN}')
    lfp_threads_info = []

    electrodes = None
    try:
        electrodes = npc_sessions.DynamicRoutingSession(session_id).electrodes[:]
    except FileNotFoundError:
        pass
    
    for lfp_path in zarr_lfp_paths:
        probe = f'Probe{npc_session.ProbeRecord(lfp_path)}'

        raw_lfp_recording = si.read_zarr(lfp_path)
        channel_ids = raw_lfp_recording.get_channel_ids()

        print(f'Starting LFP subsampling for session {session_id} and probe {probe}')

        if not is_duragel:
            # re-reference only for agar - subtract median of channels out of brain using surface channel index arg
            # similar processing to allensdk
            if electrodes is None:
                print(f'No ccf annotations for surface channel for common median referencing. Skipping {probe}')
                continue

            electrodes_probe = electrodes[electrodes['group_name'] == probe[0].lower() + probe[1:]]

            if len(electrodes_probe) == 0:
                print(f'No ccf annotations for {probe}. Thus, no surface channel for common median referencing. Skipping')
                continue

            surface_channel_index = electrodes_probe[electrodes_probe['structure'] != 'out of brain']['channel'].max() + 10
            processing_parameters[f'Surface_channel_reference_index_{probe}'] = surface_channel_index
            print(f'Common median referecing using out of brain channels from ccf annotations for {probe} with surface index {surface_channel_index}')
            reference_channel_indices = np.arange(surface_channel_index, len(channel_ids))
            reference_channel_ids = channel_ids[reference_channel_indices]
            # common median reference to channels out of brain
            recording_lfp_cmr = spre.common_reference(
                raw_lfp_recording,
                reference="global",
                ref_channel_ids=reference_channel_ids.tolist(),
            )


        channel_ids_to_keep = [channel_ids[i] for i in range(0, len(channel_ids), SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)] 
        recording_spatial_subsampled = recording_lfp_cmr.channel_slice(channel_ids_to_keep)

        recording_spatial_time_subsampled = spre.resample(recording_spatial_subsampled, int(raw_lfp_recording.sampling_frequency / TEMPORAL_SUBSAMPLE_FACTOR))

        assert (len(recording_spatial_time_subsampled.get_times()) == int(len(raw_lfp_recording.get_times()) / TEMPORAL_SUBSAMPLE_FACTOR)
        ), f"Applying {TEMPORAL_SUBSAMPLE_FACTOR} temporal factor resulted in mismatch downsampling. Got {len(recording_spatial_time_subsampled.get_times())} time samples given {len(recording.get_times())} raw time samples and factor {TEMPORAL_SUBSAMPLE_FACTOR}"
        assert (recording_spatial_time_subsampled.get_num_channels() == int(raw_lfp_recording.get_num_channels() / SPATIAL_CHANNEL_SUBSAMPLE_FACTOR)
        ), f"Applying {SPATIAL_CHANNEL_SUBSAMPLE_FACTOR} channel stride resulted in mismatch downsampling {recording.get_num_channels()} channels and {recording_spatial_time_subsampled.get_num_channels()} channels"

        filtered_recording = spre.highpass_filter(recording_spatial_time_subsampled, freq_min=HIGHPASS_FILTER_FREQ_MIN)

        lfp_threads_info.append((probe, filtered_recording, RESULTS_PATH))
        print(f'Finished LFP subsampling for session {session_id} and probe {probe}')
    
    with cf.ProcessPoolExecutor(max_workers=50) as executor:
        futures = []
        for lfp_thread_info in lfp_threads_info:
            probe, subsampled_recording, result_output_path = lfp_thread_info

            futures.append(executor.submit(utils.save_lfp_to_zarr, result_output_path=RESULTS_PATH, subsampled_recording=subsampled_recording,
                                            probe=probe, session_id=session_id))
        
        for future in cf.as_completed(futures):
            print(future.result())

    end_date_time = datetime.datetime.now()
    processing_dict = utils.get_processing_dict(start_date_time, end_date_time, processing_parameters)
    processing_model = DataProcess(**processing_dict)
    processing_pipeline = PipelineProcess(data_processes = [processing_model], processor_full_name='Arjun Sridhar')
    processing = Processing(processing_pipeline=processing_pipeline)
    processing.write_standard_file(utils.RESULTS_PATH)
            
if __name__ == "__main__": 
    run()