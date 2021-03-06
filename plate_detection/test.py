
from .data import (load_char_annots, get_loader,
                   parse_split_file_to_arrays, concat_ds, get_dataset)
from .model import KeypointRCNNContainer, SSDLiteContainer

from pathlib import Path
import unittest
import traceback
import logging
logging.basicConfig(level=logging.NOTSET)


class TestTrainingKeypointRCNN(unittest.TestCase):

    def setUp(self) -> None:
        self.test_suite_path = Path(__file__).parents[1] / "test_suite"
        self.test_filelist_path = self.test_suite_path / "val_for_test.txt"
        annot_path = Path(__file__).parents[1] / "char-annotations.yaml"
        self.annot_obj = load_char_annots(annot_path)

        img_paths, boxes, labels, kps, lpas = parse_split_file_to_arrays(
            self.test_suite_path,
            self.test_filelist_path,
            1, remap_lp_annot=self.annot_obj)
        dataset = get_dataset(img_paths, labels, boxes, kps, 1.)
        dataset = concat_ds(dataset, dataset)
        self.data_loader = get_loader(dataset)
        self.model = KeypointRCNNContainer.new_model(device="cuda")

        self.onnx_export_path = Path(__file__).parent.joinpath("model.onnx")
        self.model_state_dict_path = Path(__file__).parent.joinpath("model.pt")
        self.opt_state_dict_path = Path(__file__).parent.joinpath("optimizer.pt")

    def tearDown(self) -> None:
        self.onnx_export_path.unlink(missing_ok=True)
        self.model_state_dict_path.unlink(missing_ok=True)
        self.opt_state_dict_path.unlink(missing_ok=True)

    def test_train_noerror(self):
        raised = False
        msg = ""
        try:
            self.model.train(self.data_loader, 2, self.data_loader)
            self.model.prune_model()

            self.model.save_model_state_dict(self.model_state_dict_path)
            self.model.save_optimizer(self.opt_state_dict_path)
            self.model.load_model_state_dict(self.model_state_dict_path)
            self.model.load_optimizer(self.opt_state_dict_path)
            self.model.export_onnx(self.onnx_export_path)
            fhdlr: logging.FileHandler = self.model.logger.handlers[0]
            Path(fhdlr.baseFilename).unlink(missing_ok=True)

        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


class TestTrainingSSDLite(unittest.TestCase):

    def setUp(self) -> None:
        self.test_suite_path = Path(__file__).parents[1] / "test_suite"
        self.test_filelist_path = self.test_suite_path / "val_for_test.txt"
        annot_path = Path(__file__).parents[1] / "char-annotations.yaml"
        self.annot_obj = load_char_annots(annot_path)

        img_paths, boxes, labels, kps, lpas = parse_split_file_to_arrays(
            self.test_suite_path,
            self.test_filelist_path,
            1, remap_lp_annot=self.annot_obj)
        dataset = get_dataset(img_paths, labels, boxes, kps, 1.)
        dataset = concat_ds(dataset, dataset)
        dataset_val = get_dataset(img_paths, labels, boxes, kps, is_val=True)
        dataset_val = concat_ds(*[dataset_val for _ in range(10)])
        self.loader_train = get_loader(dataset, num_workers=6)
        self.loader_val = get_loader(dataset_val, shuffle=False, num_workers=6)
        self.model = SSDLiteContainer.new_model(device="cuda") 

    def tearDown(self) -> None:
        pass

    def test_train_noerror(self):
        raised = False
        msg = ""
        try:
            self.model.train(self.loader_train, 4, self.loader_val)
            self.model.load_checkpoint()
            self.model.prune_model()
            self.model.export_onnx()

            fhdlr: logging.FileHandler = self.model.logger.handlers[0]

            # Path(fhdlr.baseFilename).unlink(missing_ok=True)
            # (self.model.runtime_output_dir / "best.pt").unlink(missing_ok=True)
            # (self.model.runtime_output_dir / "latest.pt").unlink(missing_ok=True)
            # (self.model.runtime_output_dir / "model.onnx").unlink(missing_ok=True)
            # self.model.runtime_output_dir.rmdir()
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
