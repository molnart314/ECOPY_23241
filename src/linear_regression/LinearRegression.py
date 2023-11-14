
import numpy as np
import pandas as pd
import scipy.stats as stats

class LinearRegressionNP():

    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2
        self.alpha = None
        self.beta = None
        self.p_values = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values

        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        alpha = beta[0]
        beta = beta[1:]
        self.alpha = alpha
        self.beta = beta

    def get_params(self):
        beta_coeffs = pd.Series([self.alpha] + list(self.beta), name="Beta coefficients")
        return beta_coeffs

    def get_pvalues(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values
        n, k = X.shape
        df = n - k

        residuals = y - X.dot(np.concatenate(([self.alpha], self.beta)))
        sigma = np.sqrt((residuals.dot(residuals)) / df)
        se = np.sqrt(np.diagonal(sigma**2 * np.linalg.inv(X.T.dot(X))))
        t_stats = np.concatenate(([self.alpha / se[0]], self.beta / se[1:]))
        p_values = pd.Series([2 * (1 - stats.t.cdf(np.abs(t), df)) for t in t_stats], name="P-values for the corresponding coefficients")
        self.p_values = p_values
        return p_values
    def get_wald_test_result(self, R):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")

        R = np.hstack((R, np.zeros((R.shape[0], 1))))

        wald_value = (R @ self.beta - np.zeros(R.shape[0])) @ np.linalg.inv(R @ np.linalg.inv(self.right_hand_side.T @ self.right_hand_side) @ R.T)

        dof1 = R.shape[0]
        dof2 = self.right_hand_side.shape[0] - self.right_hand_side.shape[1]
        p_value = 1 - f.cdf(wald_value, dof1, dof2)

        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values

        n, k = X.shape
        df_residuals = n - k
        df_total = n - 1
        y_mean = np.mean(y)
        y_pred = X.dot(np.concatenate(([self.alpha], self.beta)))

        ss_residuals = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - y_mean) ** 2)
        centered_r_squared = 1 - (ss_residuals / ss_total)
        adjusted_r_squared = 1 - (ss_residuals / df_residuals) / (ss_total / df_total)
        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"