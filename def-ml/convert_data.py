import argparse

from def_ml.data import assets
from def_ml.data.conversion import (
    _save_big_enough_to_standard,
    _save_gong_surakav_to_standard,
)
from def_ml.data.utils import (
    Datasets,
    generate_xv_splits,
)

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
        raise NotImplementedError(
            f"Only '{Datasets.BIGENOUGH}' and '{Datasets.GONG_SURAKAV}'"
            "exits atm. You gave: {args.dataset}"
        )
