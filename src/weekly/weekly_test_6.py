#1
import pandas as pd

df = pd.read_parquet('sp500.parquet', engine='fastparquet')

#2
df = pd.read_parquet('ff_factors.parquet', engine='fastparquet')

#3
sp500_df = pd.read_parquet('sp500.parquet', engine='fastparquet')

ff_factors_df = pd.read_parquet('ff_factors.parquet', engine='fastparquet')

merged_df = sp500_df.merge(ff_factors_df, on='Date', how='left')

#4
merged_df['Excess Return'] = merged_df['Monthly Return'] - merged_df['RF']

#5
sp500_df = pd.read_parquet('sp500.parquet', engine='fastparquet')

ff_factors_df = pd.read_parquet('ff_factors.parquet', engine='fastparquet')

merged_df = sp500_df.merge(ff_factors_df, on='Date', how='left')

merged_df = merged_df.sort_values(by='Date')

merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)

#6
merged_df = merged_df.dropna(subset=['ex_ret_1'])

merged_df = merged_df.dropna(subset=['HML'])