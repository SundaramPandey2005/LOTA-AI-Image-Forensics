# Results of the reported run

Produced by one execution of `notebooks/run_experiments.ipynb` on Kaggle
(Tesla T4, 109 minutes, 10 Nov 2026), i.e. by

    python scripts/run_experiments.py --data-root <Tiny-GenImage> --out outputs --time-budget-hours 9
    python scripts/summarize_results.py --out outputs

| File | Content |
|---|---|
| `summary.md`, `summary.json` | all result tables (built by `scripts/summarize_results.py`) |
| `environment.json` | Python / PyTorch / CUDA versions, GPU, exact command |
| `latency.json` | parameter count and per-image latency for K = 1, 2, 4, 8 |
| `run_experiments.log` | log of the whole run |
| `runs/<name>_<generator>_s<seed>/` | one directory per trained model: `config.yaml`, `train.log`, `history.json`, `eval_<perturbation>.json` (metrics per test generator), `logits_<perturbation>.npz` (raw per-patch logits; every table can be recomputed from them) |

Run names: `lota` = baseline (K = 1), `lota_mp` = proposed (K = 4), `lota_mp_inst` = LOTA-MP with the
additional per-patch loss, `lota_resize_first` = baseline with bit-planes extracted after resizing to 256 px.

The weights (`best.pth`, 94 MB each) are not stored in git; see the README for the download links.
