
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union


def copy_image_here(dataset_dir: Union[Path, str], img_subpath: Union[Path, str]):
    src_img_path = dataset_dir / img_subpath
    dst_img_path = Path(__file__).parent / img_subpath
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)

    dst_img_path.write_bytes(src_img_path.read_bytes())


def main(args: Namespace):
    with open(args.label_file, "rt") as f:
        for line in f:
            copy_image_here(args.dataset_dir, line.rstrip("\r\n "))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset-dir", type=Path, required=True,
        help="Location of the dataset."
    )
    parser.add_argument(
        "-l", "--label-file", type=Path, required=True,
        help="Path to the label file."
    )
    args = parser.parse_args()
    main(args)
