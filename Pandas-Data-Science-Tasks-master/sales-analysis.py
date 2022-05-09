from turtle import color
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

from sympy import rotations

# df = pd.read_csv('SalesAnalysis/Sales_Data/Sales_April_2019.csv')

# #list all csv files
# files = [file for file in os.listdir('SalesAnalysis/Sales_Data')]

##combine months together
# all_months_data = pd.DataFrame()

# for file in files:
#     df = pd.read_csv('SalesAnalysis/Sales_Data/'+file)
#     all_months_data = pd.concat([all_months_data, df])

# all_months_data.to_csv('all_data.csv', index=False)

all_data = pd.read_csv('all_data.csv')
# all_data.head()

# data cleaning to add a month column
#dropping NaN rows
nan_df = all_data[all_data.isna().any(axis=1)]
# nan_df.head()

all_data = all_data.dropna(how='all')
# all_data.head()

#dropping 'Or'
all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']

#add month column
all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')

#add sales column
#convert columns to correct type
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])

all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

#add city column
def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' ' + get_state(x))

#Add Hour and Minute columns
# all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
# all_data['Hour'] = all_data['Order Date'].dt.hour
# all_data['Minute'] = all_data['Order Date'].dt.minute

###############################

# Q1: Best month for sales? How much was earned?
# results = all_data.groupby('Month').sum()

# months = range(1,13)
# plt.bar(months, results['Sales'])
# plt.xticks(months)
# plt.ylabel('Sales ($)')
# plt.xlabel('Month #')
# plt.show()

###############################

# Q2: Which city had the highest number of sales? 
# results = all_data.groupby('City').sum()

# cities = [city for city, df in all_data.groupby('City')]
# plt.bar(cities, results['Sales'])
# plt.xticks(cities, rotation= 'vertical')
# plt.ylabel('Sales ($)')
# plt.xlabel('City')
# plt.show()

###############################

# # Q3: What time should we display ads to maximize likelihood of customer buying product?
# hours = [hour for hour, df in all_data.groupby('Hour')]

# plt.plot(hours, all_data.groupby(all_data['Hour']).count())
# plt.xticks(hours)
# plt.grid()
# plt.xlabel('Hour')
# plt.ylabel('# of Orders')
# plt.show()

###############################

# Q4: Which products were most often sold together?
# df = all_data[all_data['Order ID'].duplicated(keep=False)]
# df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# df = df[['Order ID','Grouped']].drop_duplicates()

# count = Counter()

# for row in df['Grouped']:
#     row_list = row.split(',')
#     count.update(Counter(combinations(row_list,2)))

# count.most_common()

###############################

# Q5: Which product sold the most and why? 
product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product, df in product_group]
# plt.bar(products, quantity_ordered)
# plt.xticks(products, rotation='vertical')

prices = all_data.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered)
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='green')
ax2.set_ylabel('Price ($)', color='blue')
ax1.set_xticklabels(products, rotation='vertical')

plt.show()