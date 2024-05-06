import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import *


def fit_ols_regression(data, features, target, logy=False):
    data = data.dropna(subset=features + [target])
    X = data[features].to_numpy()
    y = data[target].to_numpy()

    if logy:
        y = np.log10(y)
    
    X = sm.add_constant(X)  # Add a constant term to the features
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results


def get_predictions(model, data, features):
    X = data[features].to_numpy()
    X = sm.add_constant(X)
    pred_df = model.get_prediction(X).summary_frame()
    pred_df[features] = data[features]
    return pred_df


def print_growth_rates(model):
    print(f"R^2: {model.rsquared:.2f}")
    print(f"{model.params[1]:.2f} OOMs/year (95% CI: {model.conf_int()[1][0]:.2f}, {model.conf_int()[1][1]:.2f})")
    print(f"{ooms_to_factor_per_year(model.params[1]):.1f}x/year (95% CI: {ooms_to_factor_per_year(model.conf_int()[1][0]):.1f}x, {ooms_to_factor_per_year(model.conf_int()[1][1]):.1f}x)")
    print(f"doubling time of {ooms_to_doubling_time_months(model.params[1]):.0f} months (95% CI: {ooms_to_doubling_time_months(model.conf_int()[1][1]):.0f}, {ooms_to_doubling_time_months(model.conf_int()[1][0]):.0f})")
