import numpy as np
import pandas as pd
import statsmodels.api as sm


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
