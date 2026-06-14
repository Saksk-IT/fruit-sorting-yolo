# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_config import infer_label_dir, normalize_names, resolve_split_images  # noqa: E402


class DatasetConfigTest(unittest.TestCase):
    def test_resolves_validation_images_relative_to_yaml_file(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            valid_images = dataset / "valid" / "images"
            valid_images.mkdir(parents=True)
            (dataset / "data.yaml").write_text(
                "train: train/images\nval: valid/images\nnc: 3\nnames: [ripe, half-ripe, raw]\n",
                encoding="utf-8",
            )

            image_dir = resolve_split_images(str(dataset / "data.yaml"), "val")

        self.assertEqual(image_dir, valid_images.resolve())

    def test_infers_labels_from_current_layout(self):
        image_dir = Path("/project/dataset/valid/images")

        self.assertEqual(infer_label_dir(image_dir), Path("/project/dataset/valid/labels"))

    def test_normalizes_dict_names_in_class_id_order(self):
        names = {"2": "raw", "0": "ripe", "1": "half-ripe"}

        self.assertEqual(normalize_names(names), ["ripe", "half-ripe", "raw"])


if __name__ == "__main__":
    unittest.main()
