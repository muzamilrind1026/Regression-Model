# Regression-Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

np.random.seed(42)


n = 100
x = np.linspace(1, 10, n)

true_signal = 2.0 * np.exp(0.3 * x)
noise = np.random.normal(loc=0, scale=0.5 * x)  # Variance increases with x
y = np.maximum(true_signal + noise, 0.1)        # Ensure positive y for log transform

X_ols = sm.add_constant(x)
model_ols = sm.OLS(y, X_ols).fit()

residuals_ols = model_ols.resid
fitted_ols = model_ols.fittedvalues
std_residuals_ols = model_ols.get_influence().resid_studentized_internal

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Chapter 4: Initial OLS Model Diagnostics', fontsize=16, fontweight='bold')

axes[0, 0].scatter(fitted_ols, residuals_ols, alpha=0.7, color='crimson')
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_title('1. Residuals vs. Fitted Values (Funnel/U-Shape)')
axes[0, 0].set_xlabel('Fitted Values ($\hat{y}$)')
axes[0, 0].set_ylabel('Residuals ($e$)')

stats.probplot(residuals_ols, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('2. Normal Q-Q Plot of Residuals')

axes[1, 0].scatter(fitted_ols, np.sqrt(np.abs(std_residuals_ols)), alpha=0.7, color='darkorange')
axes[1, 0].set_title('3. Scale-Location Plot ($\sqrt{|Std. Resid|}$ vs. $\hat{y}$)')
axes[1, 0].set_xlabel('Fitted Values ($\hat{y}$)')
axes[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

axes[1, 1].plot(residuals_ols, marker='o', linestyle='-', color='teal', alpha=0.7)
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_title('4. Residuals vs. Observation Order')
axes[1, 1].set_xlabel('Observation Index')
axes[1, 1].set_ylabel('Residuals ($e$)')

plt.tight_layout()
plt.show()


y_trans = np.log(y)
model_trans = sm.OLS(y_trans, X_ols).fit()

residuals_trans = model_trans.resid
fitted_trans = model_trans.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chapter 5: Model Remediation via Log Transformation $y^* = \ln(y)$', fontsize=16, fontweight='bold')

axes[0].scatter(x, y, label='Observed Data', color='navy', alpha=0.6)
axes[0].plot(x, np.exp(fitted_trans), color='red', linewidth=2, label='Back-transformed Fitted Curve')
axes[0].set_title('Linearized Fit Back-Transformed to Original Scale')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()

axes[1].scatter(fitted_trans, residuals_trans, color='forestgreen', alpha=0.7)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Transformed Residuals vs. Fitted Values (Stabilized Variance)')
axes[1].set_xlabel('Transformed Fitted Values $\ln(\hat{y})$')
axes[1].set_ylabel('Transformed Residuals ($e^*$)')

plt.tight_layout()
plt.show(
