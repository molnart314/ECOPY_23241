import pandas as pd

#1
returns_df = pd.read_parquet('sp500.parquet', engine='fastparquet')

#2
factors_df = pd.read_parquet('ff_factors.parquet', engine='fastparquet')

#3
merged_df = returns_df.merge(factors_df, on='Date', how='left')

#4
merged_df['Excess Return'] = merged_df['Return'] - merged_df['RF']

#5
merged_df = merged_df.sort_values(by='Date')
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)

#6
merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['HML'])

amazon_df = merged_df[merged_df['Symbol'] == 'AMZN']
amazon_df = amazon_df.drop(columns='Symbol')

#7-12
import numpy as np
from scipy.stats import t
class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        if not isinstance(left_hand_side, pd.DataFrame) or not isinstance(right_hand_side, pd.DataFrame):
            raise ValueError("Both inputs should be DataFrames.")

        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        self.right_hand_side['Constant'] = 1

        X = self.right_hand_side.values
        y = self.left_hand_side.values
        beta = np.linalg.inv(X.T @ X) @ X.T @ y

        self.beta = beta[:-1]

        return self.beta

    def get_params(self):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")
        return pd.Series(self.beta, name='Beta coefficients')

    def get_pvalues(self):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")
        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        dof = n - k
        residuals = self.left_hand_side - self.right_hand_side @ self.beta
        mse = np.sum(residuals ** 2) / dof
        t_values = self.beta / np.sqrt(np.diag(np.linalg.inv(self.right_hand_side.T @ self.right_hand_side) * mse))
        p_values = np.min(np.array([1 - t.cdf(np.abs(t_values), dof), t.cdf(np.abs(t_values), dof)]), axis=0) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")

        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]

        y_mean = self.left_hand_side.mean().values
        y_hat = self.right_hand_side @ self.beta
        rss = np.sum((self.left_hand_side - y_hat) ** 2)
        tss = np.sum((self.left_hand_side - y_mean) ** 2)
        crs = 1 - rss / tss

        ars = 1 - (rss / (n - k - 1)) / (tss / (n - 1))

        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"





