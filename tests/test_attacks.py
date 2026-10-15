"""Regression checks for the pixel-space threat model."""

import unittest

import torch
from torch import nn

from attacks import FGSMAttack, PGDAttack
from preprocessing import denormalize, normalize


class AttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
        self.pixels = torch.rand(3, 3, 4, 4)
        self.pixels[0, :, 0, 0] = 0
        self.pixels[1, :, 0, 0] = 1
        self.labels = torch.tensor([0, 1, 0])

    def test_normalized_attacks_respect_pixel_budget_and_range(self):
        inputs = normalize(self.pixels)
        original = inputs.clone()
        for attack_class in (PGDAttack, FGSMAttack):
            with self.subTest(attack=attack_class.__name__):
                adversarial = attack_class(self.model, epsilon=4 / 255)(inputs, self.labels)
                pixels = denormalize(adversarial)
                self.assertLessEqual((pixels - self.pixels).abs().max().item(), 4 / 255 + 1e-6)
                self.assertGreaterEqual(pixels.min().item(), -1e-6)
                self.assertLessEqual(pixels.max().item(), 1 + 1e-6)
                self.assertFalse(adversarial.requires_grad)
                torch.testing.assert_close(inputs, original)

    def test_zero_budget_leaves_images_unchanged(self):
        inputs = normalize(self.pixels)
        for attack_class in (PGDAttack, FGSMAttack):
            with self.subTest(attack=attack_class.__name__):
                result = attack_class(self.model, epsilon=0)(inputs, self.labels)
                torch.testing.assert_close(result, inputs, atol=1e-6, rtol=0)

    def test_attack_ascends_loss_for_raw_pixel_inputs(self):
        inputs = torch.full_like(self.pixels, 0.5)
        attack = PGDAttack(self.model, epsilon=0.1, alpha=0.02, num_iter=3, normalized_inputs=False)
        before = nn.functional.cross_entropy(self.model(inputs), self.labels).item()
        result = attack(inputs, self.labels, random_start=False)
        after = nn.functional.cross_entropy(self.model(result), self.labels).item()
        self.assertGreater(after, before)
        self.assertLessEqual((result - inputs).abs().max().item(), 0.1 + 1e-6)

    def test_attack_preserves_training_state_and_parameter_gradients(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.model.train(training)
                for parameter in self.model.parameters():
                    parameter.grad = torch.full_like(parameter, 0.125)
                PGDAttack(self.model, num_iter=2)(normalize(self.pixels), self.labels)
                self.assertEqual(self.model.training, training)
                for parameter in self.model.parameters():
                    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.125))

    def test_invalid_attack_settings_fail_early(self):
        for settings in ({"epsilon": -1}, {"alpha": -1}, {"num_iter": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PGDAttack(self.model, **settings)


if __name__ == "__main__":
    unittest.main()
