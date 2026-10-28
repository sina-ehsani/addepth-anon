# Recency-weighted per-position depth for spatio-temporal forecasting

Anonymized code release accompanying the submission. A lightweight positional depth router
assigns a processing depth to each input time-position of a recurrent forecaster (ConvLSTM,
PredRNN), trained end-to-end with a straight-through estimator and an optional compute budget.

## Layout
```
src/models/PredRNN.py        PredRNN with variable per-position depth (VarDepthPredRNN) and the
                             positional router (PredRNNRouted)
src/models/                  ConvLSTM-family models and depth routers used in the pilots;
                             BiDepth.py re-implements the fixed linear-schedule baseline of
                             Ehsani et al. (2025) used in the ConvLSTM pilots
src/train_loop.py            training / evaluation loops
bench_predrnn.py             Moving MNIST 10->10 (5.1M / 20M / patchify) -- main entry point
bench_predrnn_precip.py      precipitation (IMERG crop)
bench_predrnn_taxi.py        NYC taxi
bench_predrnn_taxibj.py      TaxiBJ
bench_latency.py             wall-clock inference latency of trained checkpoints
pilot_*.py, bench_mmnist10.py  ConvLSTM pilots
run_*.sh, *_gpu*.sh          the exact launch commands used for the paper's runs
results/*.json               per-run metrics behind every table (test MSE, schedule, compute)
scripts/collect_paper_numbers.py  aggregates results/ into the paper's mean +- std (sample std)
figures/make_paper_figures.py     regenerates the paper figures from results/
test_*.py                    sanity checks for the variable-depth / routed models
```

## Environment
Python 3.10+, PyTorch 2.x (tested with 2.10 + CUDA 12.8), numpy, h5py, matplotlib.

## Data
Datasets are not included. Scripts expect them one level up (`../`), overridable by flags:
Moving MNIST (`../mnist_test_seq.npy`), precipitation (`../train_test_2015_2020.h5`, built with
`PreProcessing_Weather.ipynb` from NASA GPM IMERG), NYC taxi (`../all_tensors_7_zone2.h5`, from NYC
TLC trip records), TaxiBJ (`BJ{13..16}.h5`).

## Reproduce
```bash
# e.g. 20M Moving MNIST: full depth, hand-set ramp, router, and the uniform depth-3 control
python bench_predrnn.py --help
bash run_tuned_multiseed.sh          # launch scripts set GPUs in their headers
bash run_uniform3.sh
# tables and figures from the per-run JSONs
python scripts/collect_paper_numbers.py
python figures/make_paper_figures.py
```
Set `PYTHON=/path/to/python` to choose the interpreter used by the launch scripts.
