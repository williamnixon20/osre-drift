# Drift Management Benchmarking Pipeline

## Overview

This project designs pipeline to evaluate and benchmark various state-of-the-art algorithms for handling drift. The pipeline is extensible, allowing easy integration of new datasets and algorithms. Refer to the project link here [https://ucsc-ospo.github.io/report/osre24/anl/last/20240610-williamn/].

## Project Structure

- `algorithm/`: Contains implementations of various drift management algorithms.
  - `algorithm.py`: Core algorithm implementation.
  - `alwaysretrain.py`, `aue.py`, `driftsurf.py`, `matchmaker.py`, `noretrain.py`: Specific algorithms and strategies.
- `datastream/`: Defines different data stream types.
  - `circle.py`, `covcon.py`, `datastream.py`, `ioadmission.py`, `sine.py`: Data stream generators.
- `drift-detector/`: Contains drift detection algorithms.
  - `alwaysdrift.py`, `driftdetector.py`: Drift detection mechanisms.
- `metric-evaluator/`: Metrics and evaluation tools.
  - `metricevaluator.py`: Metrics evaluation.
- `misc/`: Miscellaneous files, including large data files.
- `pipeline/`: Main pipeline components.
  - `prequential.py`: Prequential evaluation setup.
- `results/`: Results from pipeline executions.
- `OSRE-DriftBenchmark.ipynb`: Jupyter notebook for ease of use and demonstration.

## Getting Started

You can use the provided .ipynb notebook to get started with the experiments! Results from prior executions are also available at the /results folder.