import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

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
    print(f"N={model.nobs}")
    print(f"R^2={model.rsquared:.2f}")
    print(f"{model.params[1]:.2f} OOMs/year (95% CI: {model.conf_int()[1][0]:.2f}, {model.conf_int()[1][1]:.2f})")
    print(f"{ooms_to_factor_per_year(model.params[1]):.1f}x/year (95% CI: {ooms_to_factor_per_year(model.conf_int()[1][0]):.1f}x, {ooms_to_factor_per_year(model.conf_int()[1][1]):.1f}x)")
    print(f"doubling time of {ooms_to_doubling_time_months(model.params[1]):.0f} months (95% CI: {ooms_to_doubling_time_months(model.conf_int()[1][1]):.0f}, {ooms_to_doubling_time_months(model.conf_int()[1][0]):.0f})")


def chow_test(data1, data2, features, target, logy=False):
    data1 = data1.dropna(subset=features + [target])
    X1 = data1[features].to_numpy()
    X1 = sm.add_constant(X1)  # Add a constant term to the features
    y1 = data1[target].to_numpy()

    data2 = data2.dropna(subset=features + [target])
    X2 = data2[features].to_numpy()
    X2 = sm.add_constant(X2)
    y2 = data2[target].to_numpy()

    if logy:
        y1 = np.log10(y1)
        y2 = np.log10(y2)

    # Separate regressions
    model1 = sm.OLS(y1, X1).fit()
    model2 = sm.OLS(y2, X2).fit()

    # Pooled regression
    pooled_y = np.concatenate([y1, y2])
    pooled_X = np.concatenate([X1, X2])
    pooled_model = sm.OLS(pooled_y, pooled_X).fit()


    # Number of parameters and observations
    k = pooled_model.df_model + 1
    n1 = len(y1)
    n2 = len(y2)

    # Degrees of freedom
    df1 = k
    df2 = n1 + n2 - 2 * k

    # Chow test statistic
    ssr_pooled = pooled_model.ssr
    ssr1 = model1.ssr
    ssr2 = model2.ssr
    F_stat = ((ssr_pooled - (ssr1 + ssr2)) / df1) / ((ssr1 + ssr2) / df2)

    # p-value
    p_value = 1 - f.cdf(F_stat, df1, df2)

    print(f"Chow Test F-statistic: {F_stat}")
    print(f"p-value: {p_value}")

    # if p_value < 0.05:
    #     print("There is a statistically significant difference between the two regressions.")
    # else:
    #     print("There is no statistically significant difference between the two regressions.")

    return F_stat, p_value
