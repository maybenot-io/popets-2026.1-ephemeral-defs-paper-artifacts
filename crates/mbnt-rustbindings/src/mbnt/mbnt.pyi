import numpy as np

def sim_trace(
    raw_trace: str,
    machines_client: list[str],
    machines_server: list[str],
    network_delay_millis: int,
    max_trace_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def sim_trace_advanced(
    raw_trace: str,
    machines_client: list[str],
    machines_server: list[str],
    network_delay_millis: int,
    max_trace_length: int,
    max_padding_frac_client: float,
    max_padding_frac_server: float,
    max_blocking_frac_client: float,
    max_blocking_frac_server: float,
    random_state: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def sim_trace_from_file(
    path: str,
    machines_client: list[str],
    machines_server: list[str],
    network_delay_millis: int,
    max_trace_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def sim_trace_from_file_advanced(
    path: str,
    machines_client: list[str],
    machines_server: list[str],
    network_delay_millis: int,
    max_trace_length: int,
    max_padding_frac_client: float,
    max_padding_frac_server: float,
    max_blocking_frac_client: float,
    max_blocking_frac_server: float,
    events_multiplier: int,
    random_state: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def load_trace_to_np(
    path: str,
    network_delay_millis: int,
    max_trace_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def load_trace_to_str(path: str) -> str: ...
def compute_overheads(
    path_orig: str, path_defended: str, max: int, real_world: bool
) -> dict[str, float]: ...
