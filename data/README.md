# Data

## Synthetic Data

Synthetic SVAR data is generated on-the-fly by the experiment scripts (see `experiments/exp_utils.py`). No download is needed.

## CausalTime Benchmark

The CausalTime datasets (AQI, Medical, Traffic) are from:

> Cheng et al., "CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery", NeurIPS 2024.

**Download**: [https://www.causaltime.cc/](https://www.causaltime.cc/)

After downloading, place the datasets in this directory:

```
data/
  causaltime/
    AQI/
    Medical/
    Traffic/
```

## Electricity Consumption Data

The sector-level electricity consumption dataset uses publicly available aggregate statistics from China's National Bureau of Statistics. The raw data and LLM-generated prior matrix are provided upon request.
