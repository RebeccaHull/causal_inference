import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load your data
df = pd.read_csv("CLV.csv")

# For now, let’s filter out all cancellations for simplicity
df = df[df['Cancellation Year'].isna()]

# Step 2: Define treatment variable
df['treatment'] = df['Loyalty Card'].apply(lambda x: 1 if x in ['Nova', 'Star'] else 0)

# Step 3: Select covariates
covariates = ['Country', 'Province', 'City', 'Postal Code', 'Gender', 'Education', 
              'Marital Status', 'Enrollment Type', 'Enrollment Year', 'Enrollment Month']

# Handle missing salary with median imputation
imputer = SimpleImputer(strategy='median')
df['Salary'] = imputer.fit_transform(df[['Salary']])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[covariates], drop_first=True)

# Step 4: Estimate propensity scores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_scaled, df['treatment'])
df['propensity_score'] = log_reg.predict_proba(X_scaled)[:, 1]

# Step 5: Matching using Nearest Neighbors
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]

# Match each treated unit to the nearest control unit
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])

matched_control = control.iloc[indices.flatten()].copy()
matched_treated = treated.reset_index(drop=True)

# Step 6: Compare outcomes
matched_df = pd.concat([matched_treated, matched_control])
print("Average CLV - Treated:", matched_treated['CLV'].mean())
print("Average CLV - Matched Control:", matched_control['CLV'].mean())
print("Estimated treatment effect on CLV:", matched_treated['CLV'].mean() - matched_control['CLV'].mean())


# Average CLV - Treated: 7282.511146278872
# Average CLV - Matched Control: 10571.655396065013
# Estimated treatment effect on CLV: -3289.144249786141