import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the dataset (CSV format)
data = pd.read_csv('CLV.csv')

print(data.columns)

# If there are extra spaces in the column names, you can strip them
data.columns = data.columns.str.strip()

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

# Display the plot
plt.show()



###########################################

# Clean the data by removing null or empty values
data_clean = data[data['Salary'].notnull() & (data['Salary'] != '')]

# Plot the histogram
plt.hist(data_clean['Salary'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()



###############################

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

# Display the plot
plt.show()



#####################################

# Clean the data by removing null or empty values in the 'CLV' column
data_clean = data[data['CLV'].notnull() & (data['CLV'] != '')]

# Create 10 equal-width bins for CLV
bins = 6
data_clean['CLV_Binned'] = pd.cut(data_clean['CLV'], bins=bins)

# Create a bar plot for binned CLV values
ax = sns.countplot(x='CLV_Binned', data=data_clean)

# Set labels and title
plt.xlabel('Customer Lifetime Value (CLV) Range')
plt.ylabel('Count')
plt.title('Distribution of Customer Lifetime Value (CLV)')

# Display the plot without number labels
plt.show()