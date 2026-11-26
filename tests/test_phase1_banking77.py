"""Integration tests for src/phase1_banking77/data.py."""

import unittest

from src.phase1_banking77.data import (
    Banking77DecisionDataset,
    load_banking77_splits,
)


class TestBanking77Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = load_banking77_splits(val_ratio=0.2, seed=42)

    def test_split_sizes_and_stratification(self) -> None:
        self.assertEqual(self.splits.num_classes, 77)
        self.assertEqual(len(self.splits.train) + len(self.splits.val), 10_003)
        self.assertEqual(len(self.splits.test), 3_080)

        train_labels = set(self.splits.train["label"])
        val_labels = set(self.splits.val["label"])
        test_labels = set(self.splits.test["label"])
        all_labels = set(range(77))

        self.assertEqual(train_labels, all_labels)
        self.assertEqual(val_labels, all_labels)
        self.assertEqual(test_labels, all_labels)

    def test_torch_dataset_epoch_shuffle_and_determinism(self) -> None:
        ds = Banking77DecisionDataset(
            self.splits.train,
            self.splits.label_names,
            shuffle_options=True,
            seed=42,
        )
        self.assertEqual(len(ds), len(self.splits.train))

        ex_epoch0_a = ds[0]
        ex_epoch0_b = ds[0]
        self.assertEqual(ex_epoch0_a.prompt, ex_epoch0_b.prompt)
        self.assertEqual(ex_epoch0_a.correct_option_idx, ex_epoch0_b.correct_option_idx)

        ds.set_epoch(1)
        ex_epoch1 = ds[0]
        self.assertNotEqual(ex_epoch0_a.option_to_class, ex_epoch1.option_to_class)
        assert ex_epoch1.correct_option_idx is not None
        self.assertEqual(
            ex_epoch1.option_to_class[ex_epoch1.correct_option_idx],
            self.splits.train[0]["label"],
        )

    def test_val_unshuffled_is_canonical(self) -> None:
        val_ds = Banking77DecisionDataset(
            self.splits.val,
            self.splits.label_names,
            shuffle_options=False,
        )
        ex = val_ds[5]
        self.assertEqual(ex.option_to_class, list(range(77)))
        self.assertEqual(ex.correct_option_idx, self.splits.val[5]["label"])


if __name__ == "__main__":
    unittest.main()
