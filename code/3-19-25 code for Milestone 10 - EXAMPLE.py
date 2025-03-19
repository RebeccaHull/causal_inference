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