"""Synthetic CPU smoke tests; no dataset download or full training required."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from tqdm import tqdm

import train_robust
import train_standard as standard_training
from attacks import PGDAttack
from model import CNN
from preprocessing import normalize
from train_standard import train_standard
from utils import load_checkpoint, save_checkpoint


class TinyClassifier(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.loader = [(normalize(torch.rand(4, 3, 4, 4)), torch.tensor([0, 1, 0, 1]))]
        self.progress = lambda iterable, **kwargs: tqdm(iterable, disable=True)

    def test_cnn_output_shape(self):
        model = CNN().eval()
        with torch.no_grad():
            logits = model(normalize(torch.rand(2, 3, 32, 32)))
        self.assertEqual(logits.shape, (2, 10))

    def test_standard_and_mixed_training_update_weights(self):
        for ratio in (None, 0, 0.5, 1):
            with self.subTest(adversarial_ratio=ratio):
                model = TinyClassifier()
                before = model[1].weight.detach().clone()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                if ratio is None:
                    with patch("train_standard.tqdm", self.progress):
                        loss, accuracy = train_standard(
                            model,
                            self.loader,
                            optimizer,
                            nn.CrossEntropyLoss(),
                            "cpu",
                            1,
                        )
                else:
                    with patch("train_robust.tqdm", self.progress):
                        loss, accuracy = train_robust.train_robust(
                            model,
                            self.loader,
                            optimizer,
                            nn.CrossEntropyLoss(),
                            PGDAttack(model, num_iter=1),
                            "cpu",
                            1,
                            adv_ratio=ratio,
                        )
                self.assertTrue(math.isfinite(loss))
                self.assertTrue(0 <= accuracy <= 100)
                self.assertFalse(torch.equal(before, model[1].weight))

    def test_checkpoint_round_trip(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(model, optimizer, 3, 75.0, path)
            restored = TinyClassifier()
            epoch, accuracy = load_checkpoint(restored, optimizer, path, "cpu")
        self.assertEqual((epoch, accuracy), (3, 75.0))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_short_robust_run_saves_best_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                adv_ratio=1.0,
                eval_freq=10,
                pretrain_path=None,
                output_dir=directory,
                batch_size=4,
                num_workers=0,
                data_dir=directory,
                dropout=0.5,
                lr=0.01,
                momentum=0.9,
                weight_decay=0,
                lr_milestones=[75, 90],
                lr_gamma=0.1,
                train_epsilon=4 / 255,
                train_alpha=2 / 255,
                train_steps=1,
                epochs=1,
                eval_epsilon=4 / 255,
                save_freq=25,
            )
            with (
                patch("train_robust.CNN", TinyClassifier),
                patch("train_robust.get_cifar10_loaders", return_value=(self.loader, self.loader)),
                patch("train_robust.get_device", return_value=torch.device("cpu")),
                patch("train_robust.tqdm", self.progress),
                patch("utils.tqdm", self.progress),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_robust.main(args)
            self.assertTrue((Path(directory) / "models/robust_best.pth").is_file())
            self.assertTrue((Path(directory) / "models/robust_final.pth").is_file())
            history = json.loads((Path(directory) / "robust_history.json").read_text())
            self.assertEqual(history["epochs"], [1])
            self.assertEqual(history["robust_epochs"], [1])
            self.assertEqual(len(history["test_robust_acc"]), 1)

    def test_missing_requested_pretraining_checkpoint_fails_early(self):
        args = Namespace(adv_ratio=1, eval_freq=1, pretrain_path="/missing/checkpoint.pth")
        with patch("train_robust.get_cifar10_loaders") as loaders:
            with self.assertRaises(FileNotFoundError):
                train_robust.main(args)
            loaders.assert_not_called()

    def test_short_standard_run_saves_evaluation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                output_dir=directory,
                batch_size=4,
                num_workers=0,
                data_dir=directory,
                dropout=0.5,
                lr=0.01,
                momentum=0.9,
                weight_decay=0,
                lr_milestones=[75, 90],
                lr_gamma=0.1,
                epochs=1,
                eval_epsilon=4 / 255,
                save_freq=25,
            )
            with (
                patch("train_standard.CNN", TinyClassifier),
                patch(
                    "train_standard.get_cifar10_loaders", return_value=(self.loader, self.loader)
                ),
                patch("train_standard.get_device", return_value=torch.device("cpu")),
                patch("train_standard.tqdm", self.progress),
                patch("utils.tqdm", self.progress),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                standard_training.main(args)
            self.assertTrue((Path(directory) / "models/standard_best.pth").is_file())
            self.assertTrue((Path(directory) / "models/standard_final.pth").is_file())
            history = json.loads((Path(directory) / "standard_history.json").read_text())
            self.assertEqual(history["robust_epochs"], [1])
            self.assertEqual(len(history["test_robust_acc"]), 1)


if __name__ == "__main__":
    unittest.main()
