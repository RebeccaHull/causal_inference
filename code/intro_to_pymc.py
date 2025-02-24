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