import pandas as pd


def load_frontier_systems():
    """
    Load the frontier systems from the file

    Returns a list of the frontier systems
    """
    with open('data/frontier_systems.txt', 'r') as f:
        frontier_systems = [line.strip() for line in f]

    return frontier_systems


def load_pcd_df():
    dtype = {'Training compute (FLOP)': 'float64'}
    return pd.read_csv('data/All ML Systems - full view.csv', dtype=dtype)


def load_hardware_df():
    return pd.read_csv('data/Chip dataset-Grid view.csv')


def load_price_df():
    return pd.read_csv('data/Hardware prices.csv')


def load_data_for_cost_estimation():
    """
    Load the data needed for cost estimation

    Returns a tuple of the frontier systems PCD dataframe, hardware dataframe, and price dataframe
    """
    pcd_df = load_pcd_df()

    # Publication date in datetime format
    pcd_df.dropna(subset=['Publication date'], inplace=True)
    pcd_df['Publication date'] = pd.to_datetime(pcd_df['Publication date'])

    frontier_systems = load_frontier_systems()
    frontier_systems = [_.replace('Î£', 'Σ') for _ in frontier_systems]
    frontier_pcd_df = pcd_df[pcd_df['System'].isin(frontier_systems)]

    assert len(frontier_pcd_df) == len(frontier_systems)

    ## Prices
    price_df = load_price_df()

    # Price date in datetime format
    price_df.dropna(subset=['Price date'], inplace=True)
    price_df['Price date'] = pd.to_datetime(price_df['Price date'])
    pcd_hardware_model_colname = 'Name of the hardware (from Training hardware)'

    ## Hardware data
    hardware_df = load_hardware_df()

    return frontier_pcd_df, hardware_df, price_df
