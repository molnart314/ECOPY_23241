

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

###############################################


import pandas as pd
import numpy as np
from scipy.linalg import cholesky


class LinearRegressionGLS:
    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2
        self.alpha = None
        self.beta = None
        self.p_values = None

    def fit(self):
        self.right_hand_side.insert(0, 'Constant', 1)

        X = self.right_hand_side.values
        y = self.left_hand_side.values
        n, k = X.shape
        Omega = np.eye(n)

        Omega[0, 0] = 0

        residuals = y - self.right_hand_side @ np.zeros(k)
        log_squared_errors = np.log(residuals ** 2)
        sqrt_log_squared_errors = np.sqrt(log_squared_errors)

        Omega[1:, 1:] = np.diag(1 / sqrt_log_squared_errors)

        y_w = Omega @ y
        X_w = Omega @ X

        R = cholesky(X_w.T @ X_w, lower=True)
        Q, _ = np.linalg.qr(X_w)
        Q1 = Q[:, k:]
        Q2 = Q[:, :k]
        y_tilde = Q2.T @ y_w
        R_inv = np.linalg.inv(R)
        beta_tilde = R_inv @ y_tilde

        self.beta = np.zeros(k)
        self.beta[:k] = beta_tilde
        self.beta[k:] = -R_inv.T @ Q1.T @ y_w

        return self.beta

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
        se = np.sqrt(np.diagonal(sigma ** 2 * np.linalg.inv(X.T.dot(X))))
        t_stats = np.concatenate(([self.alpha / se[0]], self.beta / se[1:]))
        p_values = pd.Series([2 * (1 - stats.t.cdf(np.abs(t), df)) for t in t_stats],
                             name="P-values for the corresponding coefficients")
        self.p_values = p_values
        return p_values

    def get_wald_test_result(self, restriction_matrix):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")

        r = np.array(restriction_matrix)
        q, _ = np.linalg.qr(self.right_hand_side @ np.linalg.inv(self.right_hand_side.T @ self.right_hand_side) @ r)
        wald_value = ((r @ self.beta).T @ np.linalg.inv(q.T @ q) @ (r @ self.beta) / r.shape[0]).item()
        dof = r.shape[0]
        p_value = 1 - f.cdf(wald_value, dof, self.right_hand_side.shape[0] - self.right_hand_side.shape[1])

        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

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