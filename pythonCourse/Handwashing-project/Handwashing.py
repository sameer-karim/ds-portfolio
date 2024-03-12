import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
yearly_df = pd.read_csv('Data/yearly_deaths_by_clinic.csv')
monthly_df = pd.read_csv('Data/monthly_deaths.csv')

# Group births & deaths by clinic
clinic_deaths_sum = yearly_df.groupby('clinic')['deaths'].sum()
clinic_births_sum = yearly_df.groupby('clinic')['births'].sum()

# Add a column showing proportion of deaths per clinic
yearly_df['Proportion of Deaths'] = yearly_df['deaths'] / yearly_df['births']

# Separate clinics into 2 datasets
clinic_1 = yearly_df[yearly_df['clinic'] == 'clinic 1']
clinic_2 = yearly_df[yearly_df['clinic'] == 'clinic 2']

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Clinic 1 deaths by year
axes[0].bar(clinic_1['year'], clinic_1['deaths'], width=0.6, color='red')
axes[0].set_title('Clinic 1 Deaths by Year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('# of Deaths')

# Clinic 2 deaths by year
axes[1].bar(clinic_2['year'], clinic_2['deaths'], width=0.6, color='blue')
axes[1].set_title('Clinic 2 Deaths by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('# of Deaths')

plt.tight_layout()
plt.show()

# Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(clinic_1.pivot_table(index='year', columns='clinic', values='Proportion of Deaths'),
            cmap='Reds', ax=axes[0])
axes[0].set_title('Proportion of Deaths in Clinic 1 over Years')

sns.heatmap(clinic_2.pivot_table(index='year', columns='clinic', values='Proportion of Deaths'),
            cmap='Blues', ax=axes[1])
axes[1].set_title('Proportion of Deaths in Clinic 2 over Years')

plt.tight_layout()
plt.show()

# Basic Machine Learning - Linear Regression
# Let's predict the proportion of deaths in Clinic 1 based on the year
X = clinic_1[['year']]
y = clinic_1['Proportion of Deaths']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.title('Linear Regression: Actual vs Predicted Proportion of Deaths in Clinic 1')
plt.xlabel('Year')
plt.ylabel('Proportion of Deaths')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
