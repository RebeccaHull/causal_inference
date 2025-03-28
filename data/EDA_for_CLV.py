# EDA for CLV Project

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the dataset (CSV format)
data = pd.read_csv('CLV.csv')

print(data.columns)

# If there are extra spaces in the column names, you can strip them
data.columns = data.columns.str.strip()

###############################################################

# Create a bar plot for loyalty_card_status
ax = sns.countplot(x='Loyalty Card', data=data)

# Set labels and title
plt.xlabel('Loyalty Card Status')
plt.ylabel('Count')
plt.title('Distribution of Loyalty Card Status')

# Add the count labels on top of the bars without decimals
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=12, color='black', 
                xytext=(0, 5), textcoords='offset points')

# Save the figure
plt.savefig('../figures/loyalty_card_distribution_bar_chart.png', dpi=300, bbox_inches='tight')

###############################################################

# Clean the data by removing null, empty, and negative salary values
data_clean = data[data['Salary'].notnull() & (data['Salary'] != '')]
data_clean = data_clean[data_clean['Salary'] >= 0]

# Plot the histogram
plt.hist(data_clean['Salary'], bins=8, color='skyblue', edgecolor='black')
plt.title('Histogram of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')

# Save the figure
plt.savefig('../figures/salary_histogram.png', dpi=300, bbox_inches='tight')

###############################################################

# Create a bar plot for Enrollment Type
ax = sns.countplot(x='Enrollment Type', data=data)

# Set labels and title
plt.xlabel('Enrollment Type')
plt.ylabel('Count')
plt.title('Distribution of Enrollment Type')

# Add the count labels on top of the bars without decimals
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=12, color='black', 
                xytext=(0, 5), textcoords='offset points')

# Save the figure
plt.savefig('../figures/enrollment_type_bar_chart.png', dpi=300, bbox_inches='tight')

###############################################################

# Clean the data and remove negative values
data_clean = data[data['CLV'].notnull() & (data['CLV'] != '')]
data_clean = data_clean[data_clean['CLV'] >= 0]  

# Plot the histogram
plt.hist(data_clean['CLV'], bins=6, edgecolor='black')

# Add titles and labels
plt.title('Distribution of Customer Lifetime Value (CLV)')
plt.xlabel('Customer Lifetime Value (CLV)')
plt.ylabel('Count')

# Save the figure
plt.savefig('../figures/clv_histogram.png', dpi=300, bbox_inches='tight')