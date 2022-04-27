
from model import KeypointNetContainer
from data import get_dataset_from_json

from tensorflow.python.data.ops.dataset_ops import ParallelBatchDataset

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List
import logging


def _add_data_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--datasets", nargs="+", type=Path, required=True,
        help="Locations of dataset for training and validating."
    )
    parser.add_argument(
        "--train-labels", nargs="+", type=Path, required=True,
        help="Locations of train label files containing list of image file path."
    )
    parser.add_argument(
        "--train-mapping", nargs="+", type=int, required=True,
        help=(
            "Index mapping from train split index to index of datasets.\n"
            "Example: --datasets A B --train-splits X Y Z --train-mapping 0 0 1\n"
            "    means: train_split X Y for dataset A and Z for B."
        )
    )
    parser.add_argument(
        "--val-labels", nargs="*", type=Path,
        help="Locations of val label files containing list of image file path."
    )
    parser.add_argument(
        "--val-mapping", nargs="*", type=int,
        help="Index mapping from val split index to index of datasets."
    )


def _add_training_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Times of dataset iteration. Default: %(default)s"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Maximum size of training batch. Default: %(default)s"
    )


def _add_misc_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Location of checkpoint directory."
    )
    parser.add_argument(
        "--debug", action="store_true"
    )


def get_list_of_ds(dataset_locations: List[Path],
                   label_files: List[Path],
                   mappings: List[int],
                   batch_size: int = 16,
                   is_val = False):
    list_of_ds: List[ParallelBatchDataset] = list()
    for lf, ds_idx in zip(label_files, mappings):
        list_of_ds.append(get_dataset_from_json(
            dataset_locations[ds_idx], lf, is_val=is_val))
    ds = list_of_ds[0]
    ds = ds.unbatch()
    for other in list_of_ds[1:]:
        ds = ds.concatenate(other)
    ds = ds.batch(batch_size).prefetch(-1)
    return ds


def main(args: Namespace):
    if args.debug is True:
        logging.basicConfig(level=logging.DEBUG)
    
    ds_train = get_list_of_ds(args.datasets,
                              args.train_labels,
                              args.train_mapping,
                              batch_size=args.batch_size,
                              is_val=False)
    ds_val = None
    if len(args.val_mapping) > 0:
        ds_val = get_list_of_ds(args.datasets,
                                args.val_labels,
                                args.val_mapping,
                                batch_size=args.batch_size,
                                is_val=True)
    container = KeypointNetContainer(runtime_output_dir=args.checkpoint_dir)
    container.train(ds_train, args.epochs, ds_val)


if __name__ == "__main__":
    parser = ArgumentParser()
    _add_data_args(parser)
    _add_training_args(parser)
    _add_misc_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
