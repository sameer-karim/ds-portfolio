import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read a sample of the dataset
uber_df = pd.read_csv("uber-raw-data-sep14.csv", nrows=10000)  # Sample 10,000 rows
print(uber_df.head(5))

# Convert 'date/time' column from string to datetime
uber_df['Date/Time'] = pd.to_datetime(uber_df['Date/Time'])

# Extract additional time features
uber_df['Day'] = uber_df['Date/Time'].dt.day
uber_df['Hour'] = uber_df['Date/Time'].dt.hour
uber_df['Weekday'] = uber_df['Date/Time'].dt.weekday

# Visualize the Density of rides per weekday
plt.figure(figsize=(12, 6))
sns.histplot(uber_df['Weekday'], bins=7, kde=True, color='green')
plt.title("Density of trips per Weekday", fontsize=16)
plt.xlabel("Weekday", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
plt.show()

# Visualize the Density of rides per hour
plt.figure(figsize=(12, 6))
sns.histplot(uber_df['Hour'], bins=24, kde=True, color='orange')
plt.title("Density of trips per Hour", fontsize=16)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Density of rides", fontsize=14)
plt.show()

# Heat Map - Density of trips per location
plt.figure(figsize=(12, 8))
sns.kdeplot(x=uber_df['Lon'], y=uber_df['Lat'], cmap='viridis', fill=True)
plt.title("Density of trips per Location", fontsize=16)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.show()

# Linear Regression
# Predicting number of rides per hour
hourly_counts = uber_df.groupby('Hour').size().reset_index(name='Counts')
X = hourly_counts['Hour'].values.reshape(-1, 1)
y = hourly_counts['Counts'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Visualizing the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X, model.predict(X), color='green', linewidth=3, label='Linear Regression')
plt.title('Linear Regression - Hourly Ride Prediction')
plt.xlabel('Hour')
plt.ylabel('Number of Rides')
plt.legend()
plt.show()
