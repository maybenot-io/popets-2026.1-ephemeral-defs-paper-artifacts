import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from def_ml.data import assets
from def_ml.data.utils import (
    METADF_FNAME,
    Datasets,
    generate_xv_splits,
    get_dataset_root,
)
from def_ml.logging.logger import get_logger
from def_ml.trace.params import DOWNLOAD, UPLOAD

logger = get_logger(__name__)
load_dotenv()

DATA_DIR = os.getenv("WF_DATA_DIR")

assert DATA_DIR is not None


def _load_pickle_data(data_path: str) -> dict[int, list[list[np.ndarray | list]]]:
    """
    Load samples from pickle file
    """
    logger.info(f"Loading data from {data_path}...")
    with open(data_path, "rb") as fi:
        raw_data = pkl.load(fi)

    return raw_data


def _data_to_meta_row(
    data: np.ndarray,
    page_label: int,
    sub_page_label: int,
    sample_id: int,
    orig_path: str,
    dataset: str,
    trace_id: str,
) -> pd.Series:
    """
    Convert data to metadata series
    """

    def _n_packets(ud: str) -> int:
        if len(data) == 0:
            return 0
        if ud == "up":
            return (data[:, 1] > 0).sum()
        if ud == "down":
            return (data[:, 1] < 0).sum()

    def _time() -> int:
        if len(data) == 0:
            return 0
        return data[-1, 0] - data[0, 0]

    row_df = (
        pd.Series(
            {
                assets.PAGE_LABEL: page_label,
                assets.SUB_PAGE_LABEL: sub_page_label,
                assets.DATASET: dataset,
                "n_packets": _n_packets("up") + _n_packets("down"),
                "time [ns]": _time(),
                assets.TRACE_ID: trace_id,
                assets.SAMPLE_ID: sample_id,
                "n_packets_up": _n_packets("up"),
                "n_packets_down": _n_packets("down"),
                assets.TRACE_F_PATH: orig_path,
            }
        )
        .to_frame()
        .T.astype(
            {
                assets.PAGE_LABEL: int,
                assets.SUB_PAGE_LABEL: int,
                "n_packets": int,
                "time [ns]": float,
                assets.TRACE_ID: str,
                assets.SAMPLE_ID: int,
                assets.TRACE_F_PATH: str,
                assets.DATASET: str,
                "n_packets_up": int,
                "n_packets_down": int,
            }
        )
    )

    return row_df


def _save_meta_df(seq_rows: list[pd.DataFrame], dataset: str):
    path = get_dataset_root(dataset).joinpath(METADF_FNAME)
    if not path.parent.exists():
        os.makedirs(path.parent, exist_ok=False)

    pd.concat(seq_rows, axis=0).reset_index(drop=True).to_hdf(path, key="metadf")


def _save_gong_surakav_to_standard(save_np: bool = True):
    """
    Convert data to standard format [seq_len, n_features], where features:
        - 0: t = time
        - 1: x = direction
        - 2: s = size
    """

    root = os.path.join(DATA_DIR, Datasets.GONG_SURAKAV)

    def parse_row(row: str, idx: int) -> str:
        return row.split(",")[idx]

    trace_dfs = []
    for dir_ in sorted(os.listdir(root), key=int):
        if not os.path.isdir(os.path.join(root, dir_)):
            continue

        page_label = int(dir_)
        trace_dir = os.path.join(root, dir_)
        sub_pages = []
        for log_f in sorted(
            os.listdir(trace_dir), key=lambda x: [int(v) for v in x.split(",")[:-1]]
        ):
            if not log_f.endswith(".log"):
                logger.warning("Skipping %s", log_f)
            orig_path_ = os.path.join(trace_dir, log_f)
            with open(orig_path_, "r", encoding="utf-8") as fi:
                seq = fi.readlines()

            times = np.array([parse_row(p, 0) for p in seq], dtype=float)

            # Here (s)end and (r)eceive from the client perspective.
            dirs = np.array(
                [{"s": UPLOAD, "r": DOWNLOAD}[parse_row(p, 1)] for p in seq],
                dtype=float,
            )
            sizes = np.ones_like(times)

            # Check def_ml.data.assets for the indices!
            trace = np.vstack([times, dirs, sizes]).T
            log_f_ = log_f.replace(".log", "")

            sub_page_label = page_label
            sample_id = int(log_f_.split("-")[-1])

            sub_pages.append(sub_page_label)
            print(log_f_, page_label, sub_page_label)

            trace_df = _data_to_meta_row(
                data=trace,
                page_label=page_label,
                sub_page_label=sub_page_label,
                sample_id=sample_id,
                orig_path=orig_path_,
                dataset=Datasets.GONG_SURAKAV,
                trace_id=log_f_,
            )

            trace_dfs.append(trace_df)

    _save_meta_df(trace_dfs, Datasets.GONG_SURAKAV)


def _save_big_enough_to_standard(save_np: bool = True):
    """
    Convert data to standard format [seq_len, n_features], where features:
        - 0: t = time
        - 1: x = direction
        - 2: s = size
    """

    root = os.path.join(DATA_DIR, Datasets.BIGENOUGH)

    def parse_row(row: str, idx: int) -> str:
        return row.split(",")[idx]

    trace_dfs = []
    for dir_ in sorted(os.listdir(root), key=int):
        if not os.path.isdir(os.path.join(root, dir_)):
            continue

        page_label = int(dir_)
        trace_dir = os.path.join(root, dir_)
        sub_pages = []
        for log_f in sorted(
            os.listdir(trace_dir), key=lambda x: [int(v) for v in x.split(",")[:-1]]
        ):
            if not log_f.endswith(".log"):
                logger.warning("Skipping %s", log_f)
            orig_path_ = os.path.join(trace_dir, log_f)
            with open(orig_path_, "r", encoding="utf-8") as fi:
                seq = fi.readlines()

            times = np.array([parse_row(p, 0) for p in seq], dtype=float)

            # Here (s)end and (r)eceive from the client perspective.
            dirs = np.array(
                [{"s": UPLOAD, "r": DOWNLOAD}[parse_row(p, 1)] for p in seq],
                dtype=float,
            )
            sizes = np.ones_like(times)

            # Check def_ml.data.assets for the indices!
            trace = np.vstack([times, dirs, sizes]).T
            log_f_ = log_f.replace(".log", "")

            sub_page = int(log_f_.split("-")[1])
            sub_page_label = sub_page + page_label * 10
            sample_id = int(log_f_.split("-")[2])

            sub_pages.append(sub_page_label)
            print(log_f_, page_label, sub_page_label)

            trace_df = _data_to_meta_row(
                data=trace,
                page_label=page_label,
                sub_page_label=sub_page_label,
                sample_id=sample_id,
                orig_path=orig_path_,
                dataset=Datasets.BIGENOUGH,
                trace_id=log_f_,
            )

            trace_dfs.append(trace_df)

    _save_meta_df(trace_dfs, Datasets.BIGENOUGH)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[Datasets.GONG_SURAKAV, Datasets.BIGENOUGH],
    )
    argparser.add_argument("--save-np", action="store_true")

    args = argparser.parse_args()

    if args.dataset == Datasets.BIGENOUGH:
        _save_big_enough_to_standard(save_np=args.save_np)

        generate_xv_splits(args.dataset, n_splits=10, label_asset=assets.PAGE_LABEL)
        generate_xv_splits(args.dataset, n_splits=5, label_asset=assets.PAGE_LABEL)
    elif args.dataset == Datasets.GONG_SURAKAV:
        _save_gong_surakav_to_standard(save_np=args.save_np)

        generate_xv_splits(args.dataset, n_splits=10, label_asset=assets.PAGE_LABEL)
        generate_xv_splits(args.dataset, n_splits=5, label_asset=assets.PAGE_LABEL)

    else:
        raise NotImplementedError("Only 'bigenough' exits atm.")
