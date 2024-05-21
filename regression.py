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

    return F_stat, p_value


def regression_slope_t_test(data1, data2, features, target, logy=False):
    common_systems = data1['System'].isin(data2['System'])

    data1 = data1.dropna(subset=features + [target])
    X1 = data1[features].to_numpy()
    X1 = sm.add_constant(X1)  # Add a constant term to the features
    y1 = data1[target].to_numpy()

    data2 = data2.dropna(subset=features + [target])
    X2 = data2[features].to_numpy()
    X2 = sm.add_constant(X2)
    y2 = data2[target].to_numpy()

    data12 = data1.loc[common_systems].dropna(subset=features + [target])
    X12 = data12[features].to_numpy()
    X12 = sm.add_constant(X12)
    y12 = data12[target].to_numpy()

    if logy:
        y1 = np.log10(y1)
        y2 = np.log10(y2)
        y12 = np.log10(y12)

    # Separate regressions
    model1 = sm.OLS(y1, X1).fit()
    model2 = sm.OLS(y2, X2).fit()

    # Get the slopes and standard errors
    b1, SE1 = model1.params[1], model1.bse[1]
    b2, SE2 = model2.params[1], model2.bse[1]

    print(f"Slope 1: {b1:.2f} (SE: {SE1:.2f})")
    print(f"Slope 2: {b2:.2f} (SE: {SE2:.2f})")

    # get residuals of overlapping data as predicted by each model
    residuals_model1 = y12 - model1.predict(exog=X12)
    residuals_model2 = y12 - model2.predict(exog=X12)

    # get the correlation coefficient between residuals according to each model
    rho = np.corrcoef(residuals_model1, residuals_model2)[0, 1]

    # Calculate the test statistic
    t_stat = (b1 - b2) / np.sqrt(SE1 ** 2 + SE2 ** 2 - 2 * SE1 * SE2 * rho)

    # Degrees of freedom
    df1 = X1.shape[0] - 2
    df2 = X2.shape[0] - 2
    df = df1 + df2

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    print(f"Correlation between methods: {rho:.2f}")
    print(f"Test statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2f}")