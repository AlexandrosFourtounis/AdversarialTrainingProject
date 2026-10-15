"""PGD and FGSM attacks with perturbation budgets measured in pixel space."""

import torch
import torch.nn.functional as F

from preprocessing import denormalize, normalize


class PGDAttack:
    """L-infinity projected gradient ascent with an optional random start.

    Inputs and outputs use CIFAR-10 normalization by default. Both epsilon and
    alpha are specified in raw pixel units, such as 4/255 and 2/255. Set
    normalized_inputs=False for models accepting images directly in [0, 1].
    """

    def __init__(
        self,
        model,
        epsilon=4 / 255,
        alpha=2 / 255,
        num_iter=10,
        device="cpu",
        normalized_inputs=True,
    ):
        if epsilon < 0 or alpha < 0 or num_iter < 1:
            raise ValueError("epsilon and alpha must be nonnegative; num_iter must be positive")
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device
        self.normalized_inputs = normalized_inputs

    def project(self, x_adv, x_natural):
        """Project pixel tensors onto the epsilon ball and the valid [0, 1] range."""
        delta = (x_adv - x_natural).clamp(-self.epsilon, self.epsilon)
        return (x_natural + delta).clamp(0, 1)

    def _model_input(self, pixels):
        return normalize(pixels) if self.normalized_inputs else pixels

    def generate(self, x, y, random_start=True):
        """Generate adversarial inputs without modifying model parameter gradients."""
        x = x.detach().to(self.device)
        y = y.to(self.device)
        natural = denormalize(x) if self.normalized_inputs else x
        natural = natural.clamp(0, 1)
        adversarial = natural.clone()
        if random_start:
            noise = torch.empty_like(natural).uniform_(-self.epsilon, self.epsilon)
            adversarial = self.project(natural + noise, natural)

        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(self.num_iter):
                with torch.enable_grad():
                    adversarial = adversarial.detach().requires_grad_(True)
                    loss = F.cross_entropy(self.model(self._model_input(adversarial)), y)
                    gradient = torch.autograd.grad(loss, adversarial)[0]
                adversarial = self.project(
                    adversarial.detach() + self.alpha * gradient.sign(), natural
                )
            return self._model_input(adversarial).detach()
        finally:
            self.model.train(was_training)

    def __call__(self, x, y, random_start=True):
        return self.generate(x, y, random_start)


class FGSMAttack(PGDAttack):
    """One gradient ascent step constrained by a pixel-space epsilon budget."""

    def __init__(self, model, epsilon=4 / 255, device="cpu", normalized_inputs=True):
        super().__init__(
            model,
            epsilon=epsilon,
            alpha=epsilon,
            num_iter=1,
            device=device,
            normalized_inputs=normalized_inputs,
        )

    def generate(self, x, y):
        return super().generate(x, y, random_start=False)

    def __call__(self, x, y):
        return self.generate(x, y)
