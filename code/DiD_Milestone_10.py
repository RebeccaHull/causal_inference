# Difference-in-Differences for Milestone 10 - Rebecca Hull
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

# Outcomes

def outcome(t, control_intercept, treat_intercept_delta, trend, Δ, group, treated):
    return control_intercept + (treat_intercept_delta * group) + (trend * t) + (Δ * treated * group)

def is_treated(t, intervention_time, group):
    return (t > intervention_time) * group

# True parameters
control_intercept = 5000  # CLV for non-members
baseline_treat_intercept = 6000  # CLV for Aurora before upgrade
trend = 3000  # General CLV increase over time
Δ = 2000  # Treatment effect: additional CLV boost from upgrading to Star
intervention_time = 0.5  # When the upgrade happens

# Generating Synthetic Dataset
df = pd.DataFrame(
    {
        "group": [0, 0, 1, 1] * 10,  # 0 = No Loyalty Card, 1 = Aurora
        "t": [0.0, 1.0, 0.0, 1.0] * 10,  # Time periods (Pre/Post)
        "unit": np.concatenate([[i] * 2 for i in range(20)]),
    }
)

df["treated"] = is_treated(df["t"], intervention_time, df["group"])

df["y"] = outcome(
    df["t"],
    control_intercept,
    baseline_treat_intercept - control_intercept,  # Initial difference in CLV
    trend,
    Δ,
    df["group"],
    df["treated"],
)
df["y"] += np.random.normal(0, 500, df.shape[0])  # Add noise

# Frequentist Diff-in-Diff Calculation
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

# Bayesian Approach
with pm.Model() as model:
    # Data
    t = pm.MutableData("t", df["t"].values, dims="obs_idx")
    treated = pm.MutableData("treated", df["treated"].values, dims="obs_idx")
    group = pm.MutableData("group", df["group"].values, dims="obs_idx")
    # Priors
    _control_intercept = pm.Normal("control_intercept", 5000, 1000)
    _treat_intercept_delta = pm.Normal("treat_intercept_delta", 1000, 1000)
    _trend = pm.Normal("trend", 3000, 1000)
    _Δ = pm.Normal("Δ", 2000, 1000)
    sigma = pm.HalfNormal("sigma", 500)
    # Expectation
    mu = pm.Deterministic(
        "mu",
        outcome(t, _control_intercept, _treat_intercept_delta, _trend, _Δ, group, treated),
        dims="obs_idx",
    )
    # Likelihood
    pm.Normal("obs", mu, sigma, observed=df["y"].values, dims="obs_idx")

with model:
    idata = pm.sample()

az.plot_trace(idata, var_names="~mu");



# Results
# Pre/post difference in control group = 3006.32
# Pre/post difference in treatment group = 4935.59
# Difference in differences = 1929.27