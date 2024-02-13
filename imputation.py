import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from utils import *


def knn_impute_categorical_column(dataframe, target_col, num_neighbors=5):
    """
    Use `KNeighborsClassifier` to impute the missing values in-place in `target_col`.
    """
    # Separate the target and features
    full_features = dataframe.drop(target_col, axis=1)
    full_target = dataframe[target_col]

    # Remove rows with missing target
    missing_target_mask = full_target.isna()
    features = full_features[~missing_target_mask]
    target = full_target[~missing_target_mask]

    # Encode the target column
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # Train a KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=num_neighbors)
    knc.fit(features, target_encoded)

    # Predict the missing values
    features_with_missing_target = full_features[missing_target_mask]
    predicted = knc.predict(features_with_missing_target)

    # Decode the predictions
    predicted_labels = label_encoder.inverse_transform(predicted)

    # Replace the missing values with the predictions
    dataframe.loc[missing_target_mask, target_col] = predicted_labels

    return dataframe


def knn_impute_numerical_pcd_data(pcd_df, num_neighbors=5):
    # instantiate the imputer
    imputer = KNNImputer(n_neighbors=num_neighbors)

    # Identify categorical columns
    categorical_cols = pcd_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # one-hot encode all categorical columns
    one_hot_pcd_df = pd.get_dummies(pcd_df, columns=categorical_cols)

    # impute the missing values in Training hardware, hardware quantity, Training time (hours)
    imputed = imputer.fit_transform(one_hot_pcd_df)

    # convert the numpy array back to a dataframe
    imputed_pcd_df = pd.DataFrame(imputed, columns=one_hot_pcd_df.columns)

    # convert Training hardware back to categorical
    imputed_pcd_df['Training hardware'] = ''
    for col in imputed_pcd_df.columns:
        if col.startswith('Training hardware_'):
            training_hardware = col.split('Training hardware_')[1]
            imputed_pcd_df['Training hardware'] = imputed_pcd_df['Training hardware'] + pd.Series([int(_) * training_hardware for _ in imputed_pcd_df[col]])
    imputed_pcd_df['Training hardware'].replace('', np.nan, inplace=True)

    return imputed_pcd_df


def drop_random_values(dataframe, col, num_drop):
    """
    Set `num_drop` random values in `col` to NaN in `dataframe`.
    Returns a new dataframe with the NaNs and a dataframe with just the original dropped values.
    """
    known_values = dataframe[col].notna()
    # select num_drop random rows that have known values
    filtered_df = dataframe[known_values]
    holdout_values = filtered_df.sample(n=num_drop)
    dropped_df = dataframe.copy()
    dropped_df.loc[holdout_values.index, col] = np.nan
    return dropped_df, holdout_values


def diff_with_imputation(
    dataframe, impute_col, reference_col, imputer_fn, num_drop=5, **imputer_kwargs,
):
    """
    Compare the original dataframe with the dataframe after dropping and imputing `num_drop` 
    values in `target_col`.

    `imputer_kwargs` are passed to `imputer_fn`, which imputes the missing values in `impute_col`.
    """
    dropped_df, _ = drop_random_values(dataframe, impute_col, num_drop)
    imputed_df = dropped_df.copy()
    imputer_fn(imputed_df, impute_col, **imputer_kwargs)

    # Calculate the mean absolute error in reference_col between the original and imputed dataframes
    mae = np.mean(np.abs(dropped_df[reference_col] - imputed_df[reference_col]))
    return mae
