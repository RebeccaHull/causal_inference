# This is Rebecca Hull Simulation Data for Causal Inference
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Set the parameter values.
beta0 = 3
beta1 = 7
n = 100

sim_data = (
# Simulate predictors using appropriate np.random distributions.
pl.DataFrame({
'x': np.random.uniform(0, 7, size = n)
})
# Use predictors and parameter values to simulate the outome.
.with_columns([
(beta0 + beta1 * pl.col('x') + np.random.normal(0, 3, size = n)).alias('y')
])
)

sim_data

sns.scatterplot(data=sim_data, x='x', y='y')
sns.lmplot(data=sim_data, x='x', y='y', height=6, aspect=1, scatter_kws={'s': 10}, line_kws={'color': 'red'})

# Specify the X matrix and y vector.
X = sim_data[['x']]
y = sim_data['y']

# Create a linear regression model.
model = LinearRegression(fit_intercept=True)

# Train the model.
model.fit(X, y)

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')

# Have you recovered the parameters?