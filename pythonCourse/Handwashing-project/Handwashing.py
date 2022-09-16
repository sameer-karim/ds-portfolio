from cProfile import label
from calendar import month
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

yearly_df = pd.read_csv('Data/yearly_deaths_by_clinic.csv')
print(yearly_df)

# #Group births & deaths by clinic
print(yearly_df.groupby('clinic') ['deaths'].sum())
print(yearly_df.groupby('clinic') ['births'].sum())

#Add a column showing proportion of deaths per clinic
yearly_df['Proportion of Deaths'] = yearly_df['deaths'] / yearly_df['births']
print(yearly_df)

# #Separate 2 clinics into 2 datasets
clinic_1 = yearly_df[yearly_df['clinic'] == 'clinic 1']
clinic_2 = yearly_df[yearly_df['clinic'] == 'clinic 2']
print(clinic_1)
print(clinic_2)

fig,ax = plt.subplots(figsize = (10,4))
plt.bar(clinic_1.year, clinic_1.deaths, width= 0.6, color= 'red')
plt.title('Clinic 1 Deaths by Year')
plt.xlabel('Year')
plt.ylabel('# of Deaths')

fig,ax = plt.subplots(figsize = (10,4))
plt.bar(clinic_2.year, clinic_2.deaths, width= 0.6, color= 'blue')
plt.title('Clinic 2 Deaths by Year')
plt.xlabel('Year')
plt.ylabel('# of Deaths')

ax= clinic_1.plot(x= 'year', y= 'Proportion of Deaths', label= 'clinic_1', color= 'red')
clinic_2.plot(x= 'year', y= 'Proportion of Deaths', label= 'clinic_2', ax=ax, ylabel= 'Proportion of Deaths', color= 'blue')

plt.show()

monthly_df = pd.read_csv('Data/monthly_deaths.csv')

monthly_df['Proportion of Deaths'] = monthly_df['deaths'] / monthly_df['births']

#Dr Semmelweis ordered the doctors to wash their hands and made it obligatory in the summer of 1847 to 
# see if that will affect the number of deaths, and since we have the monthly data now, we can trace the 
# number of deaths before and after the handwashing started.

#change dates from string to datetime data type
monthly_df.dtypes
monthly_df['date'] = pd.to_datetime(monthly_df['date'])

#split dates before and after hadnwashing
start_handwashing = pd.to_datetime('1847-06-01')
before_handwashing = monthly_df[monthly_df['date'] < start_handwashing]
after_handwashing = monthly_df[monthly_df['date'] >= start_handwashing]

#before handwashing
fig,ax = plt.subplots(figsize = (10,4))
x = before_handwashing['date']
y = before_handwashing['deaths']
plt.plot(x, y, color= 'orange')
plt.xlabel('Date')
plt.ylabel('Proportion of Deaths')

# #after handwashing
fig,ax = plt.subplots(figsize = (10,4))
x = after_handwashing['date']
y = after_handwashing['deaths']
plt.plot(x, y, color= 'purple')
plt.xlabel('Date')
plt.ylabel('Proportion of Deaths')

#visualize before and after 
ax= before_handwashing.plot(x= "date", y= "Proportion of Deaths", label= "Before Handwashing", color="orange")
after_handwashing.plot(x= "date", y= "Proportion of Deaths", label= "After Handwashing", ax=ax, ylabel= "Proportion deaths", 
color="green") 

plt.show()