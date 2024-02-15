import numpy as np
import os
import pandas as pd

from plotting import *
from prices import *
from imputation import *
from inflation import *


# Manually copied from "Training cost trends" Airtable
frontier_systems = [
    "Gemini Ultra",
    "Qwen-72B",
    "Inflection-2",
    "Yi-34B",
    "ChatGLM3",
    "Falcon 180B",
    "Llama 2-70B",
    "Claude 2",
    "PaLM 2",
    "PanGu-Σ",
    "GPT-4",
    "LLaMA-65B",
    "GPT-3.5 (text-davinci-003)",
    "U-PaLM (540B)",
    "BlenderBot 3",
    "GLM-130B",
    "Minerva (540B)",
    "Parti",
    "OPT-175B",
    "Flamingo",
    "PaLM (540B)",
    "Chinchilla",
    "LaMDA",
    "GPT-NeoX-20B",
    "AlphaCode",
    "ERNIE 3.0 Titan",
    "GLaM",
    "Gopher (280B)",
    "Yuan 1.0",
    "Megatron-Turing NLG 530B",
    "HyperClova",
    "GOAT",
    "ALIGN",
    "ByT5-XXL",
    "ProtT5-XXL",
    "Meta Pseudo Labels",
    "Switch",
    "DALL-E",
    "mT5-XXL",
    "GShard (dense)",
    "iGPT-XL",
    "GPT-3 175B (davinci)",
    "Turing-NLG",
    "Meena",
    "OpenAI Five",
    "OpenAI Five Rerun",
    "AlphaStar",
    "T5-3B",
    "T5-11B",
    "Megatron-LM (8.3B)",
    "Megatron-BERT",
    "RoBERTa Large",
    "GPT-2 (1.5B)",
    "BERT-Large",
    "BigGAN-deep 512x512",
    "FTW",
    "ResNeXt-101 32x48d",
    "AmoebaNet-A (F=448)",
    "IMPALA",
    "AlphaZero",
    "AlphaGo Zero",
    "OpenAI TI7 DOTA 1v1",
    "JFT",
    "AlphaGo Master",
    "NASv3 (CIFAR-10)",
    "GNMT",
    "AlphaGo Lee",
    "AlphaGo Fan",
]


def load_data_for_cost_estimation():
    """
    Load the data needed for cost estimation

    Returns a tuple of the frontier systems PCD dataframe, hardware dataframe, and price dataframe
    """
    pcd_df = pd.read_csv('data/All ML Systems - full view.csv')

    # Publication date in datetime format
    pcd_df.dropna(subset=['Publication date'], inplace=True)
    pcd_df['Publication date'] = pd.to_datetime(pcd_df['Publication date'])

    frontier_pcd_df = pcd_df[pcd_df['System'].isin(frontier_systems)]
    # Temporary fix for string type error
    frontier_pcd_df['Training compute (FLOP)'] = pd.to_numeric(frontier_pcd_df['Training compute (FLOP)'], errors='coerce')
    assert len(frontier_pcd_df) == len(frontier_systems)

    ## Prices
    price_df = pd.read_csv('data/Hardware prices.csv')

    # Price date in datetime format
    price_df.dropna(subset=['Price date'], inplace=True)
    price_df['Price date'] = pd.to_datetime(price_df['Price date'])
    pcd_hardware_model_colname = 'Name of the hardware (from Training hardware)'

    ## Hardware data
    hardware_df = pd.read_csv('data/Chip dataset-Grid view.csv')

    return frontier_pcd_df, hardware_df, price_df


def estimate_costs(frontier_pcd_df, hardware_df, price_df, impute_pcd_data=False):
    """
    Full cost estimation pipeline
    """
    # Imputation
    if impute_pcd_data:
        # Use k nearest neighbors
        # drop unneeded columns from frontier_pcd_df
        irrelevant_columns = ['Notability criteria', 'Notability criteria notes', 'Link', 'Citations', 'Parameters notes',
                            'Training compute notes', 'Training dataset notes', 'Dataset size notes',
                            'Inference compute notes', 'Approach', 'Confidence', 'Last modified', 'Created By', 'Benchmark data',
                            'Exclude', 'Authors by country', 'Training cost trends', 'Abstract', 'Compute cost notes',
                            'Training time notes', 'Authors',
                            'Training compute cost (2020 USD)', 'Organization categorization',
                            'Training dataset', 'Inference compute (FLOP)', 'Compute sponsor categorization',
                            'Finetune compute notes']
        frontier_pcd_df = frontier_pcd_df.drop(columns=irrelevant_columns)
        # fill column 'Training cloud compute vendor' using org_to_cloud_vendor dictionary
        org_to_cloud_vendor = {
            'Google': 'Google Cloud',
            'DeepMind': 'Google Cloud',
            'Google DeepMind': 'Google Cloud',
            'Google Brain': 'Google Cloud',
            'Microsoft': 'Microsoft Azure',
            'OpenAI': 'Microsoft Azure',
        }
        frontier_pcd_df['Training cloud compute vendor'] = frontier_pcd_df['Organization'].map(org_to_cloud_vendor)
        frontier_pcd_df['Training cloud compute vendor'] = frontier_pcd_df['Training cloud compute vendor'].fillna('Amazon Web Services')
        # TODO: logarithmic scaling of big-valued columns (just copy this from validate_imputation.ipynb)
        # convert datetime to float so that it can be used in kNN
        frontier_pcd_df['Publication date'] = datetime_to_float_year(frontier_pcd_df['Publication date'])

        # set the System column as the index for formatting purposes
        frontier_pcd_df = frontier_pcd_df.set_index('System')
        # kNN parameters
        num_neighbors_general = 5  # for training time, hardware quantity, utilization
        num_neighbors_training_hardware = 5
        imputed_pcd_df = knn_impute_numerical_pcd_data(frontier_pcd_df, num_neighbors=num_neighbors_general)

        # Impute training hardware separately, because it is a categorical variable
        # There could be a better solution to this, but it seems complicated no matter what - see https://stackoverflow.com/questions/64900801/implementing-knn-imputation-on-categorical-variables-in-an-sklearn-pipeline
        imputed_pcd_df = knn_impute_categorical_column(
            imputed_pcd_df,
            num_neighbors=num_neighbors_training_hardware,
            target_col='Training hardware'
        )

        # Restore the System column
        imputed_pcd_df['System'] = frontier_pcd_df.index

        # set the System column as the index
        imputed_pcd_df = imputed_pcd_df.set_index('System')

        # insert imputed values into frontier_pcd_df
        frontier_pcd_df['Training hardware'] = imputed_pcd_df['Training hardware']
        frontier_pcd_df['Hardware quantity'] = imputed_pcd_df['Hardware quantity']
        frontier_pcd_df['Hardware utilization'] = imputed_pcd_df['Hardware utilization']
        frontier_pcd_df['Training time (hours)'] = imputed_pcd_df['Training time (hours)']
        # calculate training time (chip hours) from training time and hardware quantity
        # TODO: try estimating this from compute and FLOP/s instead. Compare the results.
        frontier_pcd_df['Training time (chip hours)'] = frontier_pcd_df['Training time (hours)'] * frontier_pcd_df['Hardware quantity']

        assert all(frontier_pcd_df['Training time (chip hours)'].notna())

        frontier_pcd_df['System'] = frontier_pcd_df.index
        # Imputation converted datetime to float
        # Need to convert back to datetime
        frontier_pcd_df['Publication date'] = frontier_pcd_df['Publication date'].apply(float_year_to_datetime)
    else:
        # set the System column as the index for formatting purposes
        frontier_pcd_df = frontier_pcd_df.set_index('System')
        frontier_pcd_df['System'] = frontier_pcd_df.index
        for _, row in frontier_pcd_df.iterrows():
            if not(pd.isna(row['Training time (hours)']) or pd.isna(row['Hardware quantity'])):
                frontier_pcd_df['Training time (chip hours)'] = frontier_pcd_df['Training time (hours)'] * frontier_pcd_df['Hardware quantity']

    """
    Price selection
    1. Use a fixed mapping from Organization to cloud provider. If no mapping found, default to "Amazon Web Services".
    2. If there's a match for the hardware model, use that. Else, discard the ML system from the dataset.
    3. Use the price that is nearest to, but prior to, training time + 2 months before the publication date
    4. If there are no prices prior to that time, use the nearest price after that time
    5. If there are no prices for that hardware model and cloud provider at all, repeat steps 3 and 4 for "Microsoft Azure", then "Google Cloud" as the cloud provider.
    6. If there are no prices found from step 5, discard the ML system from the dataset.
    """
    
    # TODO: use the vendor mapping from imputation to reduce repetition
    org_to_cloud_vendor = {
        'google': 'Google Cloud',
        'deepmind': 'Google Cloud',
        'microsoft': 'Microsoft Azure',
        'openai': 'Microsoft Azure',
    }

    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price per chip-hour (3-year CUD)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        price = find_price(row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, org_to_cloud_vendor)
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price

    # Cost estimation
    # cost = price_per_chip_hour * chip_hours
    # TODO move outside of the function
    def estimate_cost(row, system_to_price):
        system = row['System']
        price = system_to_price.get(system)
        if price is None:
            return None

        chip_hours = row['Training time (chip hours)']
        if np.isnan(chip_hours):
            return None

        cost = price * chip_hours

        # Check for base model
        if not pd.isna(row['Base model']):
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            base_cost = estimate_cost(base_model, system_to_price)
            if base_cost is None:
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            continue
        system_to_cost[row['System']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df
