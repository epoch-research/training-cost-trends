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


def most_common_over_time_impute_categorical_column(dataframe, target_col, time_col):
    """
    Impute the missing values in-place in `target_col` with the most common value for each year in `time_col`.
    Assumes `time_col` represents dates as a fractional year.
    """
    times = dataframe[time_col]
    # Time is a float year e.g. 2017.2. We want to group by the integer year e.g. 2017
    years = times.apply(int)

    def get_most_common_target(group):
        # Some values are multiple values separated by a comma
        # We want to split these and count each value separately
        split_values = group.str.split(',').explode()
        return split_values.mode().values[0]

    # Group by year
    # For hardware models, we also want to group Google TPU and GPUs separately
    if target_col == 'Training hardware':
        groups = dataframe.dropna(subset=[target_col]).groupby(
            [dataframe['Training hardware'].str.contains('TPU'), years]
        )
        grouped_targets = groups[target_col]

        # Impute the missing values with the most common value for each year
        most_common_targets = grouped_targets.apply(get_most_common_target)
        print(most_common_targets)
        for (is_tpu, year), most_common_target in most_common_targets.iteritems():
            mask = (years == year) & ((dataframe['Training hardware'].str.contains('TPU') == is_tpu) | dataframe['Training hardware'].isna())
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)
    else:
        grouped_targets = dataframe.dropna(subset=[target_col]).groupby(years)[target_col]

        # Impute the missing values with the most common value for each year
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for year, most_common_target in most_common_targets.iteritems():
            mask = years == year
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)

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


def drop_random_values(target_df, target_col, reference_df, reference_col, num_drop):
    """
    Set `num_drop` random values in `target_col` to NaN in `target_df`.
    Only drops values that are known in both `target_df[target_col]` and `reference_df[reference_col]`.
    Returns a new dataframe with the NaNs and a dataframe with just the original dropped values.
    """
    known_values = target_df[target_col].notna() & reference_df[reference_col].notna()
    # select num_drop random rows that have known values
    filtered_df = target_df[known_values]
    holdout_values = filtered_df.sample(n=num_drop)
    dropped_df = target_df.copy()
    dropped_df.loc[holdout_values.index, target_col] = np.nan
    return dropped_df, holdout_values


if __name__ == '__main__':
    from cost import *

    # Load the data
    pcd_df = pd.read_csv('data/All ML Systems - full view.csv')

    # Publication date in datetime format
    pcd_df.dropna(subset=['Publication date'], inplace=True)
    pcd_df['Publication date'] = pd.to_datetime(pcd_df['Publication date'])
    pcd_df['Publication date'] = datetime_to_float_year(pcd_df['Publication date'])

    # Impute missing values in Training hardware
    imputed_pcd_df = most_common_over_time_impute_categorical_column(pcd_df, 'Training hardware', 'Publication date')

    frontier_pcd_df, hardware_df, price_df = load_data_for_cost_estimation()
    print(frontier_pcd_df['Training hardware'])
    frontier_pcd_df.loc[:, 'Training hardware'] = imputed_pcd_df.loc[imputed_pcd_df['System'].isin(frontier_systems), 'Training hardware']
    print(frontier_pcd_df['Training hardware'])
