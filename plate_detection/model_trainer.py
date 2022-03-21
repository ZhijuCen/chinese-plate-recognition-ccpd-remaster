
import data
from model import KeypointRCNNContainer, optimizer_map

from argparse import ArgumentParser, Namespace
from pathlib import Path


def get_list_of_ds(splits, mapping, char_annot_maps):
    list_of_ds = list()
    for split_loc, ds_idx in zip(splits, mapping):
        img_paths, boxes, labels, keypoints, license_plate_annots = (
            data.parse_split_file_to_arrays(
                args.datasets[ds_idx], split_loc, remap_lp_annot=char_annot_maps)
        )
        ds = data.get_dataset(img_paths, labels, boxes, keypoints)
        list_of_ds.append(ds)
    return list_of_ds


def main(args: Namespace):
    char_annot_maps = data.load_char_annots(str(args.annot_file))

    train_ds = get_list_of_ds(args.train_splits, args.train_mapping, char_annot_maps)
    train_ds = data.concat_ds(*train_ds)

    val_ds = get_list_of_ds(args.val_splits, args.val_mapping, char_annot_maps)
    if val_ds:  # is not empty
        val_ds = data.concat_ds(*val_ds)
    else:
        val_ds = None

    train_loader = data.get_loader(train_ds, num_workers=6)
    if val_ds is not None:
        val_loader = data.get_loader(val_ds, shuffle=False, num_workers=6)
    else:
        val_loader = None

    optimizer_params = {str(s.split('=')[0]): eval(s.split('=')[1])
                        for s in args.optimizer_params}
    model = KeypointRCNNContainer.new_model(
        optimizer_map[args.optimizer], optimizer_params, "cuda")
    model.train(train_loader, args.epochs, val_loader)
    model.prune_model()
    model.export_onnx("export.onnx")


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
    parser.add_argument(
        "--optimizer", choices=["SGD", "Adam"], default="SGD",
        help="Optimizer for training."
    )
    parser.add_argument(
        "--optimizer-params", nargs="+", default=["lr=1e-3"],
        help="Parameters for optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Train epochs, default: %(default)s"
    )
    parser.add_argument(
        "--annot-file", type=Path, default="../char-annotations.yaml",
        help="Location of Character Annotation map."
    )
    args = parser.parse_args()
    print(args)
    main(args)
