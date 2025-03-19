#3-19-25 code for Milestone 10 - EXAMPLE

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn sns

#Outcomes

def outcome(t, control_intercept, treat_intercept_delta, trend, Δ, group, treated):
    return control_intercept + (treat_intercept_delta * group) + (trend * t) + (Δ * treated * group)

