import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

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

# Feature Engineering
monthly_df['date'] = pd.to_datetime(monthly_df['date'])
monthly_df['month'] = monthly_df['date'].dt.month
monthly_df['day'] = monthly_df['date'].dt.day

# Polynomial Regression
degree = 3  # Adjust the degree as needed
model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())

# Features and Target
X = monthly_df[['year', 'month', 'day']]
y = monthly_df['deaths']

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
avg_mse = -np.mean(cv_scores)

# Fit the model
model.fit(X, y)

# Predict on the full dataset for visualization
y_pred = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(monthly_df['date'], y, color='red', label='Actual')
plt.plot(monthly_df['date'], y_pred, color='blue', label='Predicted')
plt.title('Polynomial Regression: Actual vs Predicted Deaths')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.legend()
plt.show()

print('Average Cross-Validation MSE:', avg_mse)