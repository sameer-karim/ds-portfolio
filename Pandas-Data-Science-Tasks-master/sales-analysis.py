import pandas as pd
import os
import matplotlib.pyplot as plt

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
#add city column
def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + '()' + get_state(x))
all_data.head()

