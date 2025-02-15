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
        'loyalty_card_status': np.random.choice([0, 1], size=n)  # New binary predictor
    })
    # Use predictors and parameter values to simulate the outcome.
    .with_columns([ 
        (
            beta0 + income * pl.col('x') + flight_dist * pl.col('flight_dist') +
            travel_freq * pl.col('travel_freq') +
            cust_marketing_strat * pl.col('cust_marketing_strat') +
            cust_engagement * pl.col('cust_engagement') +
            loyalty_card_status * pl.col('loyalty_card_status') +  # Adding loyalty_card_status with a coefficient (e.g., 1000)
            np.random.normal(0, 3, size=n)
        ).alias('CLV')  # Renaming y to CLV
    ])
)

# Display the data
sim_data

# Visualize the data
sns.scatterplot(data=sim_data, x='x', y='CLV')
sns.lmplot(data=sim_data, x='x', y='CLV', height=6, aspect=1, scatter_kws={'s': 10}, line_kws={'color': 'red'})

# Specify the X matrix and CLV vector.
X = sim_data[['x', 'flight_dist', 'travel_freq', 'cust_marketing_strat', 'cust_engagement', 'loyalty_card_status']]
CLV = sim_data['CLV']

# Create a linear regression model.
model = LinearRegression(fit_intercept=True)
# Train the model.
model.fit(X, CLV)

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope for x: {model.coef_[0]}')
print(f'Slope for flight_dist: {model.coef_[1]}')
print(f'Slope for travel_freq: {model.coef_[2]}')
print(f'Slope for cust_marketing_strat: {model.coef_[3]}')
print(f'Slope for cust_engagement: {model.coef_[4]}')
print(f'Slope for loyalty_card_status: {model.coef_[5]}')

# Have you recovered the parameters?
# Yes