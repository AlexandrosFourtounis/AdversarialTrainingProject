# Historical coursework artifacts

They predate the pixel-space attack correction described in the [reproduction notes](../docs/reproducibility.md) and have not been regenerated with the corrected implementation.

- `analysis_standard.json`, `analysis_robust.json`: recorded epsilon sweeps and attack-success statistics.
- `summary.txt`: original text summary; its percentage differences are percentage-point changes.
- `plots/`: original aggregate comparisons.
- `visualizations/`: original example images; attack preprocessing and display normalization differ from the current code.

The baseline epsilon sweep reports 15.39% at ε = 4/255, while the separate attack-success pass reports 15.36%. The report and root README use the sweep value. 