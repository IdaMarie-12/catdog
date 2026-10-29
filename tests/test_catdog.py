import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from dataset import get_loaders, load_split
from model import build_model
from split import collect_items, make_split


class CatDogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        items, bad = collect_items()
        if bad:
            raise AssertionError(f"Corrupt images found in test data: {bad}")
        cls.split = make_split(items)

    def test_splits_do_not_overlap(self):
        paths = [set(path for path, _ in self.split[name])
                 for name in ("train", "val", "test")]
        self.assertEqual(set(), paths[0] & paths[1])
        self.assertEqual(set(), paths[0] & paths[2])
        self.assertEqual(set(), paths[1] & paths[2])
        self.assertEqual(sum(map(len, paths)), len(set.union(*paths)))

    def test_saved_split_matches_recomputed_split(self):
        self.assertEqual(load_split(), self.split)

    def test_class_balance_is_preserved(self):
        for name, expected_size in (("train", 1680), ("val", 360), ("test", 360)):
            labels = [label for _, label in self.split[name]]
            counts = Counter(labels)
            self.assertEqual(len(labels), expected_size)
            self.assertEqual(counts, Counter({0: expected_size // 2, 1: expected_size // 2}))

    def test_loader_batches_have_expected_shape(self):
        train_loader, val_loader, test_loader = get_loaders(
            augment=False, batch_size=4, num_workers=0)
        for loader in (train_loader, val_loader, test_loader):
            images, labels = next(iter(loader))
            expected_batch_size = min(loader.batch_size, len(loader.dataset))
            self.assertEqual(
                tuple(images.shape),
                (expected_batch_size, 3, config.IMG_SIZE, config.IMG_SIZE),
            )
            self.assertEqual(tuple(labels.shape), (expected_batch_size,))

    def test_model_forward_pass(self):
        model = build_model(config.DEFAULTS)
        output = model(torch.randn(4, 3, config.IMG_SIZE, config.IMG_SIZE))
        self.assertEqual(tuple(output.shape), (4, len(config.CLASSES)))
