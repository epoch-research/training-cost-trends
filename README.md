# training-cost-trends

Code for the paper "The rising costs of training frontier AI models" [1].

Install required packages using

```
pip install numpy pandas
```

The main results for training cost are produced using `cost_analysis_standalone.py` Python file.
All results are saved in a `results/` folder, with subfolders for each method and variant.

Results are currently found in 'results/all-methods-top_n=10-2025-03/17_exclude_finetunes_at_threshold_stage/cost_dataset_3_estimates.

Raw data is available in the `data/` folder:

- `All ML Systems - full view.csv` is a snapshot of the Epoch database: https://epoch.ai/data/ai-models?view=table#explore-the-data
- `Chip dataset-Grid view.csv` is a snapshot of our chip database, with technical information about chips such as FLOP/s performance.
- `Hardware prices.csv` is a snapshot of our hardware price database, including both purchase prices and cloud rental prices.
- `PCU518210518210.csv` is a snapshot of this [price index](https://fred.stlouisfed.org/series/PCU518210518210), used to adjust for inflation


[1] Ben Cottier, Robi Rahman, Loredana Fattorini, Nestor Maslej, Tamay Besiroglu, and David Owen. ‘The rising costs of training frontier AI models’. ArXiv [cs.CY], 2024. arXiv. https://arxiv.org/abs/2405.21015.
