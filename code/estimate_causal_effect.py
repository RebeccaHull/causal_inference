'''
What this code does:
- Defines a Bayesian regression model where CLV is predicted using the generated features.
- Uses normal priors for the intercept and regression coefficients.
- Uses a half-normal prior for the noise (sigma).
- Estimates the posterior distribution of the model parameters using MCMC sampling (via pymc.sample).
- Summarizes the posterior distributions of parameters.
- Plots the marginal posterior distributions for the parameters.

Causality:
This code does not estimate causal effects directly. This is a Bayesian regression model, but it only captures associations, not causal effects.

How do I estimate causal effect?

'''

import numpy as np
import polars as pl
import pymc as pm
import arviz as az
import seaborn as sns

np.random.seed(42)

# Set the parameter values.
beta0 = 1000
income = 70000
flight_dist = 3500
travel_freq = 5
cust_marketing_strat = 1
cust_engagement = 3
loyalty_card_status = 1000
n = 100

# Simulate Data
sim_data = (
    pl.DataFrame({
        'x': np.random.uniform(0, 7, size=n),
        'flight_dist': np.random.uniform(0, 10000, size=n),
        'travel_freq': np.random.uniform(1, 10, size=n),
        'cust_marketing_strat': np.random.uniform(0, 5, size=n),
        'cust_engagement': np.random.uniform(1, 10, size=n),
        'loyalty_card_status': np.random.choice([0, 1], size=n)
    })
    .with_columns([ 
        (
            beta0 + income * pl.col('x') + flight_dist * pl.col('flight_dist') +
            travel_freq * pl.col('travel_freq') +
            cust_marketing_strat * pl.col('cust_marketing_strat') +
            cust_engagement * pl.col('cust_engagement') +
            loyalty_card_status * pl.col('loyalty_card_status') +
            np.random.normal(0, 3, size=n)
        ).alias('CLV')
    ])
)

# Separate predictors and outcome
X = sim_data[['x', 'flight_dist', 'travel_freq', 'cust_marketing_strat', 'cust_engagement', 'loyalty_card_status']].to_numpy()
CLV = sim_data['CLV'].to_numpy()

# Bayesian Linear Regression Model
with pm.Model() as clv_model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    beta = pm.Normal('beta', mu=0, sigma=50000, shape=X.shape[1])  # One coefficient per feature
    sigma = pm.HalfNormal('sigma', sigma=100)

    # Likelihood
    mu = alpha + X @ beta  # Matrix multiplication of predictors and coefficients
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=CLV)

    # Posterior Sampling
    trace = pm.sample(1000, return_inferencedata=True)

# Summary of results
summary = az.summary(trace, round_to=2)
print(summary)

# Visualizing the marginal posteriors
az.plot_trace(trace, combined=True)


# Save the figure as a file
plt.savefig("trace_plot.png", dpi=300, bbox_inches="tight")

# Show the plot (optional)
plt.show()