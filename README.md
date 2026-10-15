# Adversarial Training via Min-Max Optimization

PyTorch implementation of adversarial training for CIFAR-10, developed for **CS-573: Optimization Methods** at the **University of Crete** by **Anna Makridou and Alexandros Fourtounis**.

The project combines **PGD** attacks for the inner maximization with **SGD** for model training. Eleven experiments explore pre-training, learning rates, training duration, and mixtures of clean and adversarial losses.

## Reported results

Historical results from the [project report](docs/adversarial-training-report.pdf), using PGD-10 with ε = 4/255:

| Training strategy | Clean accuracy | Robust accuracy |
| --- | ---: | ---: |
| Standard baseline | 69.07% | 15.39% |
| Pre-training + adversarial fine-tuning | 76.44% | 36.40% |
| Mixed training (75% adversarial loss) | 79.29% | 36.03% |

## Quick start

Use Python 3.10+ and run commands from the repository root. CIFAR-10 downloads automatically. MPS, CUDA, or CPU is selected automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Train a clean baseline, then fine-tune on adversarial examples.
python train_standard.py --epochs 10 --batch-size 128
python train_robust.py --epochs 5 --lr 0.01 --adv-ratio 1.0 \
  --pretrain-path output/models/standard_final.pth --eval-freq 1

# Evaluate the resulting checkpoint.
python -m scripts.analyze_robustness \
  --checkpoint output/models/robust_best.pth \
  --save-results output/analysis_robust.json
```

Use `--adv-ratio 0.75 --epochs 30` for mixed training. Each command supports `--help`. [usage notes](docs/reproducibility.md#analysis-and-visualization) cover plots and comparison reports.

## Repository

- `train_standard.py`, `train_robust.py`: training entry points.
- `model.py`, `attacks.py`, `preprocessing.py`, `utils.py`: model, attacks, and shared utilities.
- `scripts/`: evaluation and visualization tools.
- `results/`: original analysis files and figures.
- `docs/`: report and reproduction notes.
- `tests/`: CPU smoke tests and attack constraint checks.

Datasets, checkpoints, and generated runs stay local in `data/`, `experiments/`, and `output/` and are excluded from Git. 