# Reproduction and usage notes

## Historical results and current implementation

The original implementation normalized images before attacking them, then clipped those normalized tensors to [0, 1]. This changes pixels outside the claimed perturbation budget. PGD and FGSM now optimize and project in raw pixel space, with normalization applied before the model forward pass. Epsilon and step sizes therefore use pixel units consistently. Visualization uses the same normalization constants as data loading.

Other differences to account for when reproducing the report:

- **Architecture:** the supplied implementation has four convolutional layers in two blocks, followed by two fully connected layers. The report describes three blocks and approximately 4.8M parameters. The implementation's architecture is preserved for compatibility with the supplied local checkpoints.
- **Learning-rate schedule:** the code uses `MultiStepLR`, with default milestones at epochs 75 and 90. The report describes cosine annealing for pre-training. With the short README commands, the learning rate stays constant.
- **Attack step size:** training defaults to α = 2/255; evaluation uses α = ε/4, giving 1/255 at ε = 4/255. Both use ten steps and a random start unless stated otherwise.
- **Checkpoint selection:** the scripts select checkpoints using the CIFAR-10 test set. There is no independent validation split. 
- **Saved metrics:** the original baseline JSON records 15.39% robust accuracy in the epsilon sweep and 15.36% in the separate attack-success pass. 
- **Mixed training:** `--adv-ratio` weights losses computed over the same batch. It does not partition the batch into clean and adversarial subsets.

The corrected robust trainer evaluates at the final epoch even when `--eval-freq` exceeds the number of epochs, so short runs still create `robust_best.pth`. 

## Training

Run the quick-start commands in the root README after activating `.venv`. Training outputs include model checkpoints and `standard_history.json` or `robust_history.json`. For the mixed configuration described in the report:

```bash
python train_robust.py --epochs 30 --lr 0.01 --adv-ratio 0.75 \
  --pretrain-path output/models/standard_final.pth --eval-freq 1
```

The command expresses the historical configuration using the current implementation; it is not an exact reproduction of the report. Training uses seed 42 by default, but device kernels and software versions may affect results. Use `--num-workers 0` if multiprocessing is unavailable. Adversarial training and full robustness sweeps are substantially slower than clean training, especially on CPU.

## Analysis and visualization

Invoke scripts as modules from the repository root so project imports resolve correctly.

```bash
# Evaluate both models to generate comparable JSON files.
python -m scripts.analyze_robustness \
  --checkpoint output/models/standard_final.pth \
  --save-results output/analysis_standard.json
python -m scripts.analyze_robustness \
  --checkpoint output/models/robust_best.pth \
  --save-results output/analysis_robust.json

# Summarize those evaluations and plot adversarial examples.
python -m scripts.compare_results
python -m scripts.visualize_attacks \
  --checkpoint output/models/robust_best.pth --num-images 5
python -m scripts.plot_training --log-file output/robust_history.json

# Run the full analysis pipeline on both trained models.
bash scripts/run_all_analysis.sh
```

The shell pipeline uses the active environment's `python`. Set `PYTHON_BIN` to choose another interpreter.Οptional positional arguments select standard and robust checkpoints. Output goes to the ignored `output/` directory. The example visualization displays the first test images, including both successful and failed attacks.

The stored results can be plotted without training or downloading data:

```bash
python -m scripts.compare_results \
  --standard results/analysis_standard.json \
  --robust results/analysis_robust.json
```

## Validation

```bash
python -m unittest discover -s tests -v
```

The tests check pixel-space attack constraints, gradient direction, zero-budget behavior, model state preservation, and training/checkpoint smoke cases. They use synthetic CPU data and do not download CIFAR-10 or verify the historical metrics. GitHub Actions runs these tests on pushes and pull requests.
