"""Data imputation methods: KNN and most-common-value imputation."""

import logging

import numpy as np
import pandas as pd

from .constants import ORG_TO_CLOUD_VENDOR
from .utils import datetime_to_float_year, float_year_to_datetime, get_top_models

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPERS
# ==============================================================================


def get_one_hot_df(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns in a DataFrame."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return pd.get_dummies(df, columns=categorical_cols)


# ==============================================================================
# KNN IMPUTATION
# ==============================================================================


def knn_impute_categorical_column(
    dataframe: pd.DataFrame, target_col: str, num_neighbors: int = 5
) -> pd.DataFrame:
    """Use KNeighborsClassifier to impute missing values in a categorical column."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder

    full_features = dataframe.drop(target_col, axis=1)
    full_target = dataframe[target_col]

    missing_target_mask = full_target.isna()
    features = full_features[~missing_target_mask]
    target = full_target[~missing_target_mask]

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    knc = KNeighborsClassifier(n_neighbors=num_neighbors)
    knc.fit(features, target_encoded)

    features_with_missing_target = full_features[missing_target_mask]
    predicted = knc.predict(features_with_missing_target)
    predicted_labels = label_encoder.inverse_transform(predicted)

    dataframe.loc[missing_target_mask, target_col] = predicted_labels
    return dataframe


def knn_impute_numerical_pcd_data(
    one_hot_pcd_df: pd.DataFrame, num_neighbors: int = 5
) -> pd.DataFrame:
    """Impute numerical columns using KNNImputer, then recover categorical Training hardware."""
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=num_neighbors)
    imputed = imputer.fit_transform(one_hot_pcd_df)
    imputed_pcd_df = pd.DataFrame(imputed, columns=one_hot_pcd_df.columns)

    # Convert Training hardware back from one-hot to categorical
    imputed_pcd_df["Training hardware"] = ""
    for col in imputed_pcd_df.columns:
        if col.startswith("Training hardware_"):
            training_hardware = col.split("Training hardware_")[1]
            imputed_pcd_df["Training hardware"] = imputed_pcd_df["Training hardware"] + pd.Series(
                [int(_) * training_hardware for _ in imputed_pcd_df[col]]
            )
    imputed_pcd_df["Training hardware"].replace("", np.nan, inplace=True)

    return imputed_pcd_df


def knn_impute_pcd(pcd_df: pd.DataFrame, num_neighbors: int = 5) -> pd.DataFrame:
    """Full KNN imputation pipeline with feature engineering."""
    irrelevant_columns = [
        "Notability criteria", "Notability criteria notes", "Link", "Citations",
        "Parameters notes", "Training compute notes", "Training dataset notes",
        "Dataset size notes", "Inference compute notes", "Approach", "Confidence",
        "Last modified", "Created By", "Benchmark data", "Exclude",
        "Authors by country", "Training cost trends", "Abstract",
        "Compute cost notes", "Training time notes", "Authors",
        "Training compute cost (2020 USD)", "Organization categorization",
        "Training dataset", "Inference compute (FLOP)", "Compute sponsor categorization",
        "Finetune compute notes",
    ]
    pcd_df = pcd_df.drop(columns=irrelevant_columns)

    # Fill cloud vendor column using org_to_cloud_vendor mapping (case-insensitive)
    org_lower = pcd_df["Organization"].str.lower()
    pcd_df["Training cloud compute vendor"] = "Amazon Web Services"  # default
    for key, vendor in ORG_TO_CLOUD_VENDOR.items():
        pcd_df.loc[org_lower.str.contains(key, na=False), "Training cloud compute vendor"] = vendor

    # Convert large number columns to logarithmic
    parameters_col = pcd_df["Parameters"]
    training_compute_col = pcd_df["Training compute (FLOP)"]
    dataset_size_col = pcd_df["Training dataset size (datapoints)"]
    pcd_df["log_params"] = np.log10(parameters_col)
    pcd_df["log_compute"] = np.log10(training_compute_col)
    pcd_df["log_dataset"] = np.log10(dataset_size_col)
    raw_columns = ["Parameters", "Training compute (FLOP)", "Training dataset size (datapoints)"]
    pcd_df.drop(columns=raw_columns, inplace=True)

    # Convert datetime to float for kNN
    pcd_df["Publication date"] = datetime_to_float_year(pcd_df["Publication date"])

    pcd_df = pcd_df.set_index("Model")
    one_hot_pcd_df = get_one_hot_df(pcd_df)
    imputed_pcd_df = knn_impute_numerical_pcd_data(one_hot_pcd_df, num_neighbors=num_neighbors)

    # Impute training hardware separately (categorical variable)
    imputed_pcd_df = knn_impute_categorical_column(
        imputed_pcd_df, num_neighbors=num_neighbors, target_col="Training hardware"
    )

    imputed_pcd_df["Model"] = pcd_df.index
    imputed_pcd_df = imputed_pcd_df.set_index("Model")

    # Insert imputed values back
    pcd_df["Training hardware"] = imputed_pcd_df["Training hardware"]
    pcd_df["Hardware quantity"] = imputed_pcd_df["Hardware quantity"]
    pcd_df["Hardware utilization (MFU)"] = imputed_pcd_df["Hardware utilization (MFU)"]
    pcd_df["Training time (hours)"] = imputed_pcd_df["Training time (hours)"]
    pcd_df["Training time (chip hours)"] = pcd_df["Training time (hours)"] * pcd_df["Hardware quantity"]

    # Restore columns that were dropped
    pcd_df["Parameters"] = parameters_col
    pcd_df["Training compute (FLOP)"] = training_compute_col
    pcd_df["Training dataset size (datapoints)"] = dataset_size_col

    assert all(pcd_df["Training time (chip hours)"].notna())

    pcd_df["Model"] = pcd_df.index
    pcd_df["Publication date"] = pcd_df["Publication date"].apply(float_year_to_datetime)

    return pcd_df


# ==============================================================================
# MOST COMMON VALUE IMPUTATION
# ==============================================================================


def most_common_impute(
    dataframe: pd.DataFrame, target_col: str, time_col: str
) -> pd.DataFrame:
    """Impute missing values with the most common value for each year."""
    times = dataframe[time_col]
    years = times.apply(int)

    def get_most_common_target(group: pd.Series) -> str:
        split_values = group.str.split(",").explode()
        return split_values.mode().values[0]

    if target_col == "Training hardware":
        groups = dataframe.dropna(subset=[target_col]).groupby(
            [dataframe["Training hardware"].str.contains("TPU"), years]
        )
        grouped_targets = groups[target_col]
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for (is_tpu, year), most_common_target in most_common_targets.items():
            mask = (years == year) & (
                (dataframe["Training hardware"].str.contains("TPU") == is_tpu)
                | dataframe["Training hardware"].isna()
            )
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)
    else:
        grouped_targets = dataframe.dropna(subset=[target_col]).groupby(years)[target_col]
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for year, most_common_target in most_common_targets.items():
            mask = years == year
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)

    return dataframe


def most_common_impute_training_hardware(
    full_pcd_df: pd.DataFrame, pcd_df: pd.DataFrame
) -> pd.DataFrame:
    """Impute missing Training hardware values using the most common value per year."""
    full_pcd_df_copy = full_pcd_df.copy()
    full_pcd_df_copy["Publication date"] = datetime_to_float_year(full_pcd_df_copy["Publication date"])

    imputed_pcd_df = most_common_impute(full_pcd_df_copy.copy(), "Training hardware", "Publication date")

    frontier_systems = get_top_models(full_pcd_df_copy)
    pcd_df.loc[:, "Training hardware"] = imputed_pcd_df.loc[frontier_systems.index, "Training hardware"]

    for _, row in pcd_df.iterrows():
        if not (pd.isna(row["Training time (hours)"]) or pd.isna(row["Hardware quantity"])):
            pcd_df["Training time (chip hours)"] = (
                pcd_df["Training time (hours)"] * pcd_df["Hardware quantity"]
            )

    return pcd_df
