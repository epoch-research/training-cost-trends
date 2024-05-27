# training-cost-trends

Code for the paper "How much does frontier AI cost?" [1].

Install required packages using

```
pip install -r requirements.txt
```

The main results for training cost are produced using `cost_analysis.ipynb` notebook.
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


[1] Ben Cottier, Robi Rahman, Loredana Fattorini, Nestor Maslej, and David Owen. 2024. How much does frontier AI cost?
