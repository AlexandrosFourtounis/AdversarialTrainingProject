# Adversarial Training via Min-Max Saddle Point Optimization

Course project for **CS-573: Optimization Methods**, University of Crete.
**Authors:** Anna Makridou and Alexandros Fourtounis.

Investigates adversarial robustness in CIFAR-10 image classification using a convolutional neural network. The min-max approach uses **Projected Gradient Descent (PGD)** to generate adversarial examples and **Stochastic Gradient Descent (SGD)** to train the model against them.

Eleven experiments explore clean pre-training, learning rates, training duration, and mixtures of clean and adversarial examples, with robustness evaluated across multiple attack strengths.

## Results

Reported on the CIFAR-10 test set. Robust accuracy is measured under **PGD-10 attacks** with an L∞ perturbation budget of **ε = 4/255**:

| Training strategy | Clean accuracy | Robust accuracy |
| --- | ---: | ---: |
| Standard training | 69.07% | 15.39% |
| Pre-training + adversarial fine-tuning | 76.44% | 36.40% |
| Mixed training (75% adversarial, 25% clean) | 79.29% | 36.03% |

Adversarial fine-tuning improves robust accuracy by **21.01 percentage points (~2.4×)** over the baseline while also improving clean accuracy.

## Project report

[Read the full report](adversarial-training-report.pdf) for the formulation, algorithms, experimental setup, and analysis.
