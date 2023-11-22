
import pandas as pd
import numpy as np
import scipy

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


class LinearRegressionGLS:

    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side.insert(0, 'alfa', 1)
        self.left_hand_side = self.left_hand_side.values
        self.right_hand_side = self.right_hand_side.values
        self._model = None

    def fit(self):
        Y_OLS = self.left_hand_side
        X_OLS = self.right_hand_side
        self.XtX_OLS = X_OLS.T@X_OLS
        self.XtX_inv_OLS = np.linalg.inv(self.XtX_OLS)
        self.Xty_OLS = X_OLS.T@Y_OLS
        self.betas_OLS = self.XtX_inv_OLS@self.Xty_OLS
        self.residuals_OLS = Y_OLS - X_OLS@self.betas_OLS
        Y_new = np.log(self.residuals_OLS**2)
        X_new = self.right_hand_side
        self.XtX_feas = X_new.T@X_new
        self.XtX_inv_feas = np.linalg.inv(self.XtX_feas)
        self.Xty_feas = X_new.T@Y_new
        self.betas_feas = self.XtX_inv_feas@self.Xty_feas
        pred_Y = X_OLS@self.betas_feas
        pred_Y = np.sqrt(np.exp(pred_Y))
        pred_Y = pred_Y**-1
        self.V_inv = np.diag(pred_Y)

        return

    def get_params(self):
        self.XtX_gls = self.right_hand_side.T@self.V_inv@self.right_hand_side
        self.XtX_inv_gls = np.linalg.inv(self.XtX_gls)
        self.Xty_gls = self.right_hand_side.T@self.V_inv@self.left_hand_side
        self.betas_gls = self.XtX_inv_gls@self.Xty_gls
        return pd.Series(self.betas_gls, name='Beta coefficients')

    def get_pvalues(self):
        self.residuals_gls = self.left_hand_side - self.right_hand_side@self.betas_gls
        self.n = self.right_hand_side.shape[0]
        self.K = self.right_hand_side.shape[1]
        self.df = self.n - self.K
        self.variance_gls = self.residuals_gls.T@self.residuals_gls / self.df
        self.stderror_gls = np.sqrt(self.variance_gls * np.diag(self.XtX_inv_gls))
        self.t_stat_gls = np.divide(self.betas_gls, self.stderror_gls)
        term = np.minimum(scipy.stats.t.cdf(self.t_stat_gls, self.df), 1 - scipy.stats.t.cdf(self.t_stat_gls, self.df))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        term_1 = restr_matrix@self.betas_gls
        term_2 = np.linalg.inv(restr_matrix@self.XtX_inv_gls@np.array(restr_matrix).T)
        m = len(restr_matrix)
        f_stat = (term_1.T@term_2@term_1/m)/self.variance_gls
        p_value = 1 - scipy.stats.f.cdf(f_stat, m, self.df)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        Y = self.left_hand_side
        X = self.right_hand_side
        SST = Y.T@self.V_inv@Y
        SSE = Y.T @ self.V_inv @ X @ np.linalg.inv(X.T @ self.V_inv @ X) @ X.T @ self.V_inv @ Y
        r2 = 1 - SSE / SST
        adj_r2 = 1 - ((self.n - 1) / self.df * (1 - r2))
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'

#######################################

import pandas as pd
import numpy as np
from scipy.optimize import minimize

class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        if not isinstance(left_hand_side, pd.DataFrame) or not isinstance(right_hand_side, pd.DataFrame):
            raise ValueError("Both inputs should be DataFrames.")

        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.beta = None

    def fit(self):
        n_predictors = self.right_hand_side.shape[1]
        initial_params = np.full(n_predictors + 2, 0.1)  # Setting initial values to 0.1 for all parameters
        result = minimize(self._negative_log_likelihood, initial_params, method='L-BFGS-B')

        if result.success:
            self.beta = result.x[1:]
            self._correct_mle_variance(result.hess_inv)
        else:
            raise ValueError("Optimization did not converge. Please check your input data.")

    def _negative_log_likelihood(self, params):
        alpha = params[0]
        beta = params[1:]
        y = self.left_hand_side.values
        X = np.column_stack((np.ones(len(self.left_hand_side)), self.right_hand_side.values))
        epsilon = y - X @ beta - alpha
        neg_log_likelihood = 0.5 * np.sum(epsilon ** 2)
        return neg_log_likelihood

    def _correct_mle_variance(self, hessian_inv):
        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        sigma_squared = np.sum((self.left_hand_side - self.right_hand_side @ self.beta) ** 2) / (n - k)
        cov_matrix = hessian_inv * sigma_squared
        std_errors = np.sqrt(np.diag(cov_matrix))
        self.std_errors = std_errors

    def get_params(self):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")
        params_with_alpha = [np.nan] + list(self.beta)
        return pd.Series(params_with_alpha, name='Beta coefficients')

    def get_pvalues(self):
        if not hasattr(self, 'beta'):
            raise ValueError("Model parameters have not been estimated. Please call fit() first.")

        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        dof = n - k
        residuals = self.left_hand_side - self.right_hand_side @ self.beta
        mse = np.sum(residuals ** 2) / dof
        t_values = self.beta / np.sqrt(np.diag(np.linalg.inv(self.right_hand_side.T @ self.right_hand_side) * mse))
        p_values = np.min(np.array([min(t.cdf(tv, dof), 1 - t.cdf(tv, dof)) * 2 for tv in np.abs(t_values)]), axis=0)

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

        if k > 1:
            ars = 1 - (rss / (n - k)) / (tss / (n - 1))
        else:
            ars = crs

        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"
