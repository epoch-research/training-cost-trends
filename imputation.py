import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from data import *
from utils import *


def get_one_hot_df(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # one-hot encode all categorical columns
    one_hot_df = pd.get_dummies(df, columns=categorical_cols)
    return one_hot_df


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


def knn_impute_numerical_pcd_data(one_hot_pcd_df, num_neighbors=5):
    # instantiate the imputer
    imputer = KNNImputer(n_neighbors=num_neighbors)

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


def knn_impute_pcd(pcd_df, num_neighbors_general=5, num_neighbors_training_hardware=5):
    # Use k nearest neighbors
    # drop unneeded columns from pcd_df
    # TODO: drop Reference column? It's the title of the paper, which is unique
    irrelevant_columns = ['Notability criteria', 'Notability criteria notes', 'Link', 'Citations', 'Parameters notes',
                        'Training compute notes', 'Training dataset notes', 'Dataset size notes',
                        'Inference compute notes', 'Approach', 'Confidence', 'Last modified', 'Created By', 'Benchmark data',
                        'Exclude', 'Authors by country', 'Training cost trends', 'Abstract', 'Compute cost notes',
                        'Training time notes', 'Authors',
                        'Training compute cost (2020 USD)', 'Organization categorization',
                        'Training dataset', 'Inference compute (FLOP)', 'Compute sponsor categorization',
                        'Finetune compute notes']
    pcd_df = pcd_df.drop(columns=irrelevant_columns)
    # fill column 'Training cloud compute vendor' using org_to_cloud_vendor dictionary
    org_to_cloud_vendor = {
        'Google': 'Google Cloud',
        'DeepMind': 'Google Cloud',
        'Google DeepMind': 'Google Cloud',
        'Google Brain': 'Google Cloud',
        'Microsoft': 'Microsoft Azure',
        'OpenAI': 'Microsoft Azure',
    }
    pcd_df['Training cloud compute vendor'] = pcd_df['Organization'].map(org_to_cloud_vendor)
    pcd_df['Training cloud compute vendor'] = pcd_df['Training cloud compute vendor'].fillna('Amazon Web Services')

    # convert large number columns to logarithmic
    parameters_col = pcd_df['Parameters']
    training_compute_col = pcd_df['Training compute (FLOP)']
    dataset_size_col = pcd_df['Training dataset size (datapoints)']
    pcd_df['log_params'] = np.log10(parameters_col)
    pcd_df['log_compute'] = np.log10(training_compute_col)
    pcd_df['log_dataset'] = np.log10(dataset_size_col)
    # drop raw columns
    raw_columns = ['Parameters', 'Training compute (FLOP)', 'Training dataset size (datapoints)']
    pcd_df.drop(columns=raw_columns, inplace=True)

    # convert datetime to float so that it can be used in kNN
    pcd_df['Publication date'] = datetime_to_float_year(pcd_df['Publication date'])

    # set the System column as the index for formatting purposes
    pcd_df = pcd_df.set_index('System')
    one_hot_pcd_df = get_one_hot_df(pcd_df)
    imputed_pcd_df = knn_impute_numerical_pcd_data(one_hot_pcd_df, num_neighbors=num_neighbors_general)

    # Impute training hardware separately, because it is a categorical variable
    # There could be a better solution to this, but it seems complicated no matter what - see https://stackoverflow.com/questions/64900801/implementing-knn-imputation-on-categorical-variables-in-an-sklearn-pipeline
    imputed_pcd_df = knn_impute_categorical_column(
        imputed_pcd_df,
        num_neighbors=num_neighbors_training_hardware,
        target_col='Training hardware'
    )

    # Restore the System column
    imputed_pcd_df['System'] = pcd_df.index

    # set the System column as the index
    imputed_pcd_df = imputed_pcd_df.set_index('System')

    # insert imputed values into pcd_df
    pcd_df['Training hardware'] = imputed_pcd_df['Training hardware']
    pcd_df['Hardware quantity'] = imputed_pcd_df['Hardware quantity']
    pcd_df['Hardware utilization'] = imputed_pcd_df['Hardware utilization']
    pcd_df['Training time (hours)'] = imputed_pcd_df['Training time (hours)']
    # calculate training time (chip hours) from training time and hardware quantity
    # TODO: try estimating this from compute and FLOP/s instead. Compare the results.
    pcd_df['Training time (chip hours)'] = pcd_df['Training time (hours)'] * pcd_df['Hardware quantity']
    # Restore columns that were dropped
    pcd_df['Parameters'] = parameters_col
    pcd_df['Training compute (FLOP)'] = training_compute_col
    pcd_df['Training dataset size (datapoints)'] = dataset_size_col

    assert all(pcd_df['Training time (chip hours)'].notna())

    pcd_df['System'] = pcd_df.index
    # Imputation converted datetime to float
    # Need to convert back to datetime
    pcd_df['Publication date'] = pcd_df['Publication date'].apply(float_year_to_datetime)


def most_common_impute(dataframe, target_col, time_col):
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
        for (is_tpu, year), most_common_target in most_common_targets.items():
            mask = (years == year) & ((dataframe['Training hardware'].str.contains('TPU') == is_tpu) | dataframe['Training hardware'].isna())
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)
    else:
        grouped_targets = dataframe.dropna(subset=[target_col]).groupby(years)[target_col]

        # Impute the missing values with the most common value for each year
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for year, most_common_target in most_common_targets.items():
            mask = years == year
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)

    return dataframe


def most_common_impute_training_hardware(pcd_df):
    """
    Impute the missing values in the `Training hardware` of `pcd_df` with the most common value
    for each year in the full PCD data.
    """
    # Load full PCD data to get as much data as possible
    full_pcd_df = load_pcd_df()

    # Publication date in datetime format
    full_pcd_df.dropna(subset=['Publication date'], inplace=True)
    full_pcd_df['Publication date'] = pd.to_datetime(full_pcd_df['Publication date'])
    full_pcd_df['Publication date'] = datetime_to_float_year(full_pcd_df['Publication date'])

    # Impute missing values in Training hardware
    imputed_pcd_df = most_common_impute(full_pcd_df, 'Training hardware', 'Publication date')

    frontier_systems = load_frontier_systems()
    pcd_df.loc[:, 'Training hardware'] = imputed_pcd_df.loc[
        imputed_pcd_df['System'].isin(frontier_systems), 'Training hardware'
    ]

    # TODO: probably want to move this part one level up in the functions, like `knn_impute_pcd`
    for _, row in pcd_df.iterrows():
        if not(pd.isna(row['Training time (hours)']) or pd.isna(row['Hardware quantity'])):
            pcd_df['Training time (chip hours)'] = pcd_df['Training time (hours)'] * pcd_df['Hardware quantity']

    return pcd_df


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

