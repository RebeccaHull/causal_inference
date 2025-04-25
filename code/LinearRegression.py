# Import Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# === STEP 1: Load your data ===
df = pd.read_csv('data/CLV.csv')  # <-- Your file path looks good

# === STEP 2: Map loyalty status to ordered numeric values ===
loyalty_order = ['Aurora', 'Nova', 'Star']
loyalty_map = {level: i for i, level in enumerate(loyalty_order)}
df['loyalty_numeric'] = df['Loyalty Card'].map(loyalty_map)

# === STEP 3: Drop rows with missing values in key columns ===
df = df.dropna(subset=['loyalty_numeric', 'Salary', 'Enrollment Type', 'CLV'])

# === STEP 4: Create dummy variables for enrollment type ===
df = pd.get_dummies(df, columns=['Enrollment Type'], drop_first=True)

# === STEP 5: Define your predictors ===
predictors = ['loyalty_numeric', 'Salary'] + [col for col in df.columns if col.startswith('Enrollment Type_')]
X = df[predictors]
y = df['CLV']

# === STEP 6: Fit the linear regression model ===
model = LinearRegression()
model.fit(X, y)

# === STEP 7: Print results ===
print(f'\nIntercept: {model.intercept_:.2f}')
for name, coef in zip(X.columns, model.coef_):
    print(f'Slope for {name}: {coef:.2f}')

# Results
# Intercept: 10652.88
# Slope for loyalty_numeric: -1796.82
# Slope for Salary: -0.00
# Slope for Enrollment Type_Standard: -12.88