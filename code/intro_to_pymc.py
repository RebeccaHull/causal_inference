import numpy as np
import polars as pl
import pymc as pm
import arviz as az

np.random.seed(42)

# Set the parameter values.
beta0 = 3
beta1 = 7
sigma = 3
n = 100

# Simulate Data
x = np.random.uniform(0, 7, size = n)
y = beta0 + beta1 * x + np.random.normal(0, sigma, size = n)

# Create a Model object
basic_model = pm.Model()

# Specify the Model
with basic_model:
    # prior
    beta = pm.Normal('beta', mu = 0, sigma = 10, shape = 2) #shape is 2 because both betas will use this normal dist for their prior
    # beta0 = pm.Normal('beta', mu = 0, sigma = 10)
    # beta1 = pm.Normal('beta', mu = 0, sigma = 10) #these two lines do the same as above
    sigma = pm.HalfNormal('sigma', sigma = 1)

    # Likelihood
    mu = beta[0] + beta[1] * x
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y)

# Create an inference object
with basic_model:
    # Draw 1000 posterior samples
    idata = pm.sample()

# Have we recovered parameters?
az.summary(idata, round_to = 2)

# Visualize the marginal posteriors
az.plot_trace(idata, combined = True)


# Import foxes data
foxes = pl.read_csv('../data/foxes.csv')

# Separate predictors and the outcome
X = foxes.select(pl.col(['avgfood', 'groupsize'])).to_numpy()
y = foxes.select(pl.col('weight')).to_numpy().flatten() #Flattens it from 2D matrix to 1D

# Estimate the direct causal effect of avgfood on weight
with pm.Model() as foxes_model:
    # Data
    X_data = pm.Data('X_data', X)
    y_data = pm.Data('y_data', y)

    # Priors
    alpha = pm.Normal('alpha', mu = 0, sigma = 0.2)
    beta = pm.Normal('beta', mu = 0, sigma = 0.5, shape = 2)
    sigma = pm.Exponential('sigma', lam = 1)

    # Likelihood
    mu = alpha + X_data @ beta
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed= y_data)

# Sample
with foxes_model:
    draw = pm.sample()



