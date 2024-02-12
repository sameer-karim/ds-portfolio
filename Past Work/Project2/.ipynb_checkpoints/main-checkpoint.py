import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hateCrime1_df = pd.read_csv('table13.csv')
hateCrime2_df = pd.read_csv('table14.csv')

state = hateCrime1_df['State']
race = hateCrime1_df['Race']
religion = hateCrime1_df['Religion']
orientation = hateCrime1_df['Sexual orientation']
ethnicity = hateCrime1_df['Ethnicity']
disability = hateCrime1_df['Disability']
gender = hateCrime1_df['Gender']
genderIdentity = hateCrime1_df['Gender Identity']
q1 = hateCrime1_df['1st quarter']
q2 = hateCrime1_df['2nd quarter']
q3 = hateCrime1_df['3rd quarter']
q4 = hateCrime1_df['4th quarter']
population = hateCrime1_df['Population']


fig, ax = plt.subplots()
ax.bar(state, race)
ax.set_xlabel('State')
ax.set_ylabel('Race Motivated Crimes')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.bar(state, religion)
ax.set_xlabel('State')
ax.set_ylabel('Religion Motivated Crimes')
plt.xticks(rotation=90, fontsize=6)
plt.show()

fig, ax = plt.subplots()
ax.bar(state, orientation)
ax.set_xlabel('State')
ax.set_ylabel('Sexual Orientation Motivated Crimes')
plt.xticks(rotation=90, fontsize=6)
plt.show()

fig, ax = plt.subplots()
ax.bar(state, ethnicity)
ax.set_xlabel('State')
ax.set_ylabel('Ethnicity Motivated Crimes')
plt.xticks(rotation=90, fontsize=6)
plt.show()

fig, ax = plt.subplots()
ax.bar(state, disability)
ax.set_xlabel('State')
ax.set_ylabel('Disability Motivated Crimes')
plt.xticks(rotation=90, fontsize=6)
plt.show()

# fig, ax = plt.subplots()
# ax.bar(state, gender)
# ax.set_xlabel('State')
# ax.set_ylabel('Gender Motivated Crimes')
# plt.xticks(rotation=90, fontsize=6)
# plt.show()

fig, ax = plt.subplots()
ax.bar(state, genderIdentity)
ax.set_xlabel('State')
ax.set_ylabel('Gender Identity Motivated Crimes')
plt.xticks(rotation=90, fontsize=6)
plt.show()

cat1 = np.add(q1, 0)
cat2 = np.add(cat1, q2)
cat3 = np.add(cat2, q3)
cat4 = np.add(cat3, q4)

plt.bar(state, cat1, label='Q1')
plt.bar(state, cat2, label='Q2')
plt.bar(state, cat3, label='Q3')
plt.bar(state, cat4, label='Q4')
plt.xticks(rotation=90, fontsize=6)

plt.legend()
plt.show()