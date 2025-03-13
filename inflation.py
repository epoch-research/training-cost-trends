import pandas as pd


def adjust_value_for_inflation(row, cost_colname, price_index, to_year_month):
    if pd.isna(row[cost_colname]):
        return row[cost_colname]
    
    from_date = str(row['Publication date'])
    from_year_month = from_date.rsplit('-', maxsplit=1)[0] + '-01'
    from_price_index = price_index[price_index['observation_date'] == from_year_month]['PCU518210518210'].values[0]
    to_price_index = price_index[price_index['observation_date'] == to_year_month]['PCU518210518210'].values[0]
    adjust_factor = to_price_index / from_price_index
    return row[cost_colname] * adjust_factor


def adjust_column_for_inflation(df, cost_colname, path_to_price_index, to_year_month):
    price_index = pd.read_csv(path_to_price_index)
    df[cost_colname + ' (inflation-adjusted)'] = df.apply(
        adjust_value_for_inflation, axis=1, args=(cost_colname, price_index, to_year_month)
    )
    return df


if __name__ == '__main__':
    input_df = pd.DataFrame({
        'Model': ['A', 'B', 'C'],
        'Publication date': ['2017-01-01', '2019-07-01', '2021-01-01'],
        'Training cost (USD)': [100, 100, None],
    })
    output_df = adjust_column_for_inflation(input_df, 'Training cost (USD)', 'data/PCU518210518210.csv', '2020-01-01')
    print(output_df)
