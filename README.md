# training-cost-trends

Code for the paper "The rising costs of training frontier AI models" [1].

Install required packages using

```
pip install -r requirements.txt
```

The main results for training cost are produced using `cost_analysis.ipynb` notebook.
All results are saved in a `results/` folder, with subfolders for each method and variant.
The third cell of the notebook (under the "Setup" heading) assigns the following parameters, which can be varied to produce the different results:

```
estimation_method = 'hardware-capex-energy'  # e.g. cloud
compute_threshold_method = 'top_n'  # e.g. top_n, window_percentile
compute_threshold = 10  # e.g. 10 to select top 10; 75 to select top 25%
variant = 'original'  # for anything else being varied
exclude_models_containing = []  # e.g. ['AlphaGo Master', 'AlphaGo Zero']
```

The total amortized model development cost approach is implemented in the `development_cost.ipynb` notebook.
There is a parameter `INCLUDE_EQUITY` to toggle including equity in the R&D staff costs.

Raw data is available in the `data/` folder:

- `All ML Systems - full view.csv` is a snapshot of the Epoch database: https://epochai.org/data/epochdb/table
- `Chip dataset-Grid view.csv` is a snapshot of our chip database, with technical information about chips such as FLOP/s performance.
- `Hardware prices.csv` is a snapshot of our hardware price database, including both purchase prices and cloud rental prices.
- `PCU518210518210.csv` is a snapshot of this [price index](https://fred.stlouisfed.org/series/PCU518210518210), used to adjust for inflation


[1] Ben Cottier, Robi Rahman, Loredana Fattorini, Nestor Maslej, and David Owen. ‘The rising costs of training frontier AI models’. ArXiv [cs.CY], 2024. arXiv. https://arxiv.org/abs/2405.21015.
