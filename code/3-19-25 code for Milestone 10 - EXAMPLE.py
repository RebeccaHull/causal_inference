#3-19-25 code for Milestone 10 - EXAMPLE

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

#Outcomes

def outcome(t, control_intercept, treat_intercept_delta, trend, Δ, group, treated):
    return control_intercept + (treat_intercept_delta * group) + (trend * t) + (Δ * treated * group)

def is_treated(t, intervention_time, group):
    return (t > intervention_time) * group

#True parameters
control_intercept = 1
treat_intercept_delta = 0.25
trend = 1 #trend is the slope
Δ = 0.5
intervention_time = 0.5 #when the treatment was implemented in time

#Generating our Synthetic Dataset
df = pd.DataFrame(
    {
        "group": [0, 0, 1, 1] * 10,
        "t": [0.0, 1.0, 0.0, 1.0] * 10,
        "unit": np.concatenate([[i] * 2 for i in range(20)]),
    }
)

df["treated"] = is_treated(df["t"], intervention_time, df["group"])

df["y"] = outcome(
    df["t"],
    control_intercept,
    treat_intercept_delta,
    trend,
    Δ,
    df["group"],
    df["treated"],
)
df["y"] += np.random.normal(0, 0.1, df.shape[0])
df.head()


# calculate a point estimate of the difference in differences
diff_control = (
    df.loc[(df["t"] == 1) & (df["group"] == 0)]["y"].mean()
    - df.loc[(df["t"] == 0) & (df["group"] == 0)]["y"].mean()
)
print(f"Pre/post difference in control group = {diff_control:.2f}")

diff_treat = (
    df.loc[(df["t"] == 1) & (df["group"] == 1)]["y"].mean()
    - df.loc[(df["t"] == 0) & (df["group"] == 1)]["y"].mean()
)

print(f"Pre/post difference in treatment group = {diff_treat:.2f}")

diff_in_diff = diff_treat - diff_control
print(f"Difference in differences = {diff_in_diff:.2f}")


# BAYESIAN APPROACH
with pm.Model() as model:
    # data
    t = pm.MutableData("t", df["t"].values, dims="obs_idx")
    treated = pm.MutableData("treated", df["treated"].values, dims="obs_idx")
    group = pm.MutableData("group", df["group"].values, dims="obs_idx")
    # priors
    _control_intercept = pm.Normal("control_intercept", 0, 5)
    _treat_intercept_delta = pm.Normal("treat_intercept_delta", 0, 1)
    _trend = pm.Normal("trend", 0, 5)
    _Δ = pm.Normal("Δ", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    # expectation
    mu = pm.Deterministic(
        "mu",
        outcome(t, _control_intercept, _treat_intercept_delta, _trend, _Δ, group, treated),
        dims="obs_idx",
    )
    # likelihood
    pm.Normal("obs", mu, sigma, observed=df["y"].values, dims="obs_idx")


