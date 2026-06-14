# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_config import infer_label_dir, normalize_names, resolve_split_images  # noqa: E402


class DatasetConfigTest(unittest.TestCase):
    def test_resolves_current_dataset_validation_images(self):
        image_dir = resolve_split_images(str(ROOT / "dataset" / "data.yaml"), "val")

        self.assertEqual(image_dir, ROOT / "dataset" / "valid" / "images")
        self.assertTrue(image_dir.exists())

    def test_infers_labels_from_current_layout(self):
        image_dir = ROOT / "dataset" / "valid" / "images"

        self.assertEqual(infer_label_dir(image_dir), ROOT / "dataset" / "valid" / "labels")

    def test_normalizes_dict_names_in_class_id_order(self):
        names = {"2": "raw", "0": "ripe", "1": "half-ripe"}

        self.assertEqual(normalize_names(names), ["ripe", "half-ripe", "raw"])


if __name__ == "__main__":
    unittest.main()
