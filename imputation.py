import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from utils import *


def knn_impute_column(dataframe, target_col, num_neighbors=5):
    """
    Use `KNeighborsClassifier` to impute the missing values in-place in `target_col`.
    """
    # Separate the target and features
    features = dataframe.drop(target_col, axis=1)
    target = dataframe[target_col]

    # Encode the target column
    label_encoder = LabelEncoder()
    target_filled = target.fillna('Unknown')  # Temporarily fill missing values
    target_encoded = label_encoder.fit_transform(target_filled)

    # Train a KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=num_neighbors)
    knc.fit(features, target_encoded)

    # Predict the missing values
    missing_values = features[target.isna()]
    predicted = knc.predict(missing_values)

    # Decode the predictions
    predicted_labels = label_encoder.inverse_transform(predicted)

    # Replace the missing values with the predictions
    dataframe.loc[target.isna(), target_col] = predicted_labels

    # replace all 'Unknown' with np.nan
    dataframe[target_col] = dataframe[target_col].replace('Unknown', np.nan)

    return dataframe


def impute_pcd_data(pcd_df, num_neighbors=5):
    # instantiate the imputer
    imputer = KNNImputer(n_neighbors=num_neighbors)

    # convert datetime to float
    pcd_df['Publication date'] = datetime_to_float(pcd_df['Publication date'])

    # set the System column as the index
    pcd_df = pcd_df.set_index('System')

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

    return imputed_pcd_df, one_hot_pcd_df


def drop_random_values(dataframe, col, num_drop):
    """
    Set `num_drop` random values in `col` to NaN in `dataframe`.
    """
    known_values = dataframe[col].notna()
    # select num_drop random rows that have known values
    filtered_df = dataframe[known_values]
    holdout_values = filtered_df.sample(n=num_drop)
    dropped_df = dataframe.copy()
    dropped_df.loc[holdout_values.index, col] = np.nan
    return dropped_df


def diff_with_imputation(dataframe, impute_col, reference_col, num_neighbors=5, num_drop=5):
    """
    Compare the original dataframe with the dataframe after dropping and imputing `num_drop` values in `target_col`.
    """
    dropped_df = drop_random_values(dataframe, impute_col, num_drop)
    imputed_df = dropped_df.copy()
    knn_impute_column(imputed_df, impute_col, num_neighbors)

    # Calculate the mean absolute error in reference_col between the original and imputed dataframes
    mae = np.mean(np.abs(dropped_df[reference_col] - imputed_df[reference_col]))
    return mae
