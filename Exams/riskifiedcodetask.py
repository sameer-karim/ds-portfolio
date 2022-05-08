import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('new_dataset.csv')
# df.head(5)

# df.describe()

np.percentile(df['classification_score'], 10)
# df.count(df['classification_score'] > 0.999974139124037)

# plt.boxplot = df.boxplot(figsize= (20,15), by= 'classification_score')

plt.boxplot = df.boxplot(figsize= (10,7), column=['classification_score'])

