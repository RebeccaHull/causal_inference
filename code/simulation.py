# Rebecca Hull - Milestone 5

# Is all the parameters you include everything from the DAG or just everything from the adjustment set?? Because I have UP & LTE right now but they are not in my adjustment set.
# Do I need to relabel y as CLV?

# I have no idea if this code is right but the parameters I got back are very close, so that's good!

import numpy as np
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Set the parameter values.
beta0 = 1000
income = 70000
flight_dist = 3500
travel_freq = 5
cust_marketing_strat = 1
cust_engagement = 3
upgrades_perks = 6
length_time_enr = 12
loyalty_card_status = 1000
n = 100

sim_data = (
    # Simulate predictors using appropriate np.random distributions.
    pl.DataFrame({
        'x': np.random.uniform(0, 7, size=n),
        'flight_dist': np.random.uniform(0, 10000, size=n),
        'travel_freq': np.random.uniform(1, 10, size=n),
        'cust_marketing_strat': np.random.uniform(0, 5, size=n),
        'cust_engagement': np.random.uniform(1, 10, size=n),
        'upgrades_perks': np.random.uniform(0, 5, size=n),
        'length_time_enr': np.random.uniform(0, 15, size=n),
        'loyalty_card_status': np.random.choice([0, 1], size=n)  # New binary predictor
    })
    # Use predictors and parameter values to simulate the outcome.
    .with_columns([
        (
            beta0 + income * pl.col('x') + flight_dist * pl.col('flight_dist') +
            travel_freq * pl.col('travel_freq') +
            cust_marketing_strat * pl.col('cust_marketing_strat') +
            cust_engagement * pl.col('cust_engagement') +
            upgrades_perks * pl.col('upgrades_perks') +
            length_time_enr * pl.col('length_time_enr') +
            loyalty_card_status * pl.col('loyalty_card_status') +  # Adding loyalty_card_status with a coefficient (e.g., 1000)
            np.random.normal(0, 3, size=n)
        ).alias('y')
    ])
)

# Display the data
sim_data

# Visualize the data
sns.scatterplot(data=sim_data, x='x', y='y')
sns.lmplot(data=sim_data, x='x', y='y', height=6, aspect=1, scatter_kws={'s': 10}, line_kws={'color': 'red'})

# Specify the X matrix and y vector.
X = sim_data[['x', 'flight_dist', 'travel_freq', 'cust_marketing_strat', 'cust_engagement', 'upgrades_perks', 'length_time_enr', 'loyalty_card_status']]
y = sim_data['y']

# Create a linear regression model.
model = LinearRegression(fit_intercept=True)

# Train the model.
model.fit(X, y)

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope for x: {model.coef_[0]}')
print(f'Slope for flight_dist: {model.coef_[1]}')
print(f'Slope for travel_freq: {model.coef_[2]}')
print(f'Slope for cust_marketing_strat: {model.coef_[3]}')
print(f'Slope for cust_engagement: {model.coef_[4]}')
print(f'Slope for upgrades_perks: {model.coef_[5]}')
print(f'Slope for length_time_enr: {model.coef_[6]}')
print(f'Slope for loyalty_card_status: {model.coef_[7]}')

# Have you recovered the parameters?
# Yes