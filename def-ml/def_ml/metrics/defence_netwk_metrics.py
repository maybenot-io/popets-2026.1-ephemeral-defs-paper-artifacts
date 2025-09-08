from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from mbnt import compute_overheads, load_trace_to_str
from tqdm import tqdm

from def_ml.data.assets import DATASET, TRACE_F_PATH
from def_ml.data.utils import tensor_dict_to_str
from def_ml.defences.base import _Def
from def_ml.logging.logger import TQDM_W, get_logger
from def_ml.trace.params import MAX_TRACE_LENGTH

logger = get_logger(__name__)


def _make_tmp(
    defence: _Def,
    meta_df: pd.DataFrame,
    dir_orig: Path,
    dir_defended: Path,
):

    if meta_df.loc[:, DATASET].nunique() != 1:
        raise ValueError("Multiple datasets detected")

    dataset = meta_df.loc[:, DATASET].unique()[0]

    trace_paths = meta_df.loc[:, TRACE_F_PATH].values

    with tqdm(trace_paths, desc="tmp files", ncols=TQDM_W) as pbar:

        for i, trace_path in enumerate(pbar):
            mach_idx = None
            if defence.FIXED_PER_TRACE:
                mach_idx = i
            trace_d = defence._simulate(trace_path, mach_idx)

            sub_folder = trace_path.split(dataset)[-1][1:]

            orig_path = dir_orig.joinpath(sub_folder)
            def_path = dir_defended.joinpath(sub_folder)

            defended_str_trace = tensor_dict_to_str(trace_d)
            original_str_trace = load_trace_to_str(trace_path)
            for p_, trace_str in (
                (orig_path, original_str_trace),
                (def_path, defended_str_trace),
            ):
                if not p_.parent.exists():
                    p_.parent.mkdir()

                with open(p_, "w", encoding="utf-8") as f:
                    f.write(trace_str)


def get_overheads(
    defence: _Def,
    meta_df: pd.DataFrame,
    max_len: int = MAX_TRACE_LENGTH,
    real_world: bool = False,
    full_output: bool = False,
) -> dict[str, float]:

    logger.info("Compute overheads for: %s", defence.name)
    with ExitStack() as stack:
        dirs = [
            stack.enter_context(TemporaryDirectory(suffix=".traces", prefix=prefix))
            for prefix in ("orig", f"defended_{defence.name}")
        ]

        _make_tmp(defence, meta_df, Path(dirs[0]), Path(dirs[1]))

        overheads = compute_overheads(dirs[0], dirs[1], max_len, real_world)

    overheads_fin: dict[str, float] = {}
    overheads_fin["def.bandwidth"] = overheads["defended"] / overheads["base"] - 1.0
    overheads_fin["def.delay"] = overheads["delay"]
    overheads_fin["sim.missing"] = overheads["missing"]

    if full_output:
        for k, v in overheads.items():
            if k in {"delay", "missing"}:
                continue
            overheads_fin[k] = v
    return overheads_fin
