import json
import npc_session
import zarr
import pathlib

DATA_PATH = pathlib.Path('/data')

def check_saved_subsampled_lfp_result(result_output_path: pathlib.Path, input_data_path: pathlib.Path, temporal_subsampling_factor: int, 
                                    spatial_channel_subsampling_factor: int) -> None:
    raw_lfp = zarr.open(input_data_path, mode='r')['traces_seg0']
    subsampled_lfp = zarr.open(result_output_path, mode='r')['traces_seg0']

    assert (subsampled_lfp.shape[0] == raw_lfp.shape[0] / temporal_subsampling_factor
    ), f"Temporal subsampling mismatch after saving to zarr. Got subsampled time samples {subsampled_lfp.shape[0]} given raw time samples {raw_lfp.shape[0]} and temporal factor {temporal_subsampling_factor}"

    assert (subsampled_lfp.shape[1] == raw_lfp.shape[1] / spaital_channel_subsampling_factor
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