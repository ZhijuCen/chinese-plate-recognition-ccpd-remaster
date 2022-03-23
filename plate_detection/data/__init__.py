
import sys

from .dataset import get_loader, get_dataset, concat_ds
try:
    from ...utils.filename_parser import parse_split_file_to_arrays, load_char_annots
except ValueError:
    sys.path.append("../..")
    from utils.filename_parser import parse_split_file_to_arrays, load_char_annots
