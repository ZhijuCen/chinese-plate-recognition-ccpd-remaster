
from argparse import ArgumentParser
from pathlib import Path


def main(args):
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", type=Path, required=True,
        help="Locations of dataset for training and validating."
    )
    parser.add_argument(
        "--train-splits", nargs="+", type=Path, required=True,
        help="Locations of train split files containing list of image file path."
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
        "--val-splits", nargs="*", type=Path,
        help="Locations of val split files containing list of image file path."
    )
    parser.add_argument(
        "--val-mapping", nargs="*", type=int,
        help="Index mapping from val split index to index of datasets."
    )
