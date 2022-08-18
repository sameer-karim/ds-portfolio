import math
from ntpath import join
from statistics import mean
from turtle import circle, shape
from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read dataset
shape_df = pd.read_csv('Analyst_Coding_Test_(1)_(2).csv')
# shape_df.head(5)

#################################################################

# 1) boxplot - grouped by shape
plt.boxplot = shape_df.boxplot(figsize= (10,7), by= 'shape')

#################################################################

# 2) mean, max, st dev of area of each color
#group by color
area_mean = shape_df.groupby(['color']).mean()
area_max = shape_df.groupby(['color']).max()
area_std = shape_df.groupby(['color']).std()

#print values
print(f'Mean: \n', area_mean, '\n')
print(f'Max: \n', area_max, '\n')
print(f'Standard Deviation: \n', area_std, '\n')

#################################################################

# 3) Mean area of yellow square

yellow_square = shape_df[(shape_df['color'] == 'yellow') & (shape_df['shape'] == 'square')]
print(yellow_square.mean())

#################################################################

# 4) Shape mostly likely to be green
# create data frames for each green shape
green_circle = shape_df[(shape_df['color'] == 'green') & (shape_df['shape'] == 'circle')]
green_square = shape_df[(shape_df['color'] == 'green') & (shape_df['shape'] == 'square')]
green_triangle = shape_df[(shape_df['color'] == 'green') & (shape_df['shape'] == 'triangle')]

# create variables that store chances
chance_circle = (green_circle.count() / shape_df.count()) * 100
chance_square = (green_square.count() / shape_df.count()) * 100
chance_triangle = (green_triangle.count() / shape_df.count()) * 100

# store variables in dictionary
max_chance = {float(chance_circle['shape']):'circle', float(chance_square['shape']):'square', float(chance_triangle['shape']):'triangle'}

# print the max value from dictionary
print(max_chance.get(max(max_chance)))

#################################################################

# 5) Red shape with area > 3000, chances of each shape?
# create dataframes for total red shapes and individual red shapes
red_shapes = shape_df[(shape_df['color'] == 'red') & (shape_df['area'] > 3000)]
red_circle = shape_df[(shape_df['color'] == 'red') & (shape_df['area'] > 3000) & (shape_df['shape'] == 'circle')]
red_square = shape_df[(shape_df['color'] == 'red') & (shape_df['area'] > 3000) & (shape_df['shape'] == 'square')]
red_triangle = shape_df[(shape_df['color'] == 'red') & (shape_df['area'] > 3000) & (shape_df['shape'] == 'triangle')]

# store chances in variables 
chance_red_circle = (red_circle.count() / red_shapes.count()) * 100
chance_red_square = (red_square.count() / red_shapes.count()) * 100
chance_red_triangle = (red_triangle.count() / red_shapes.count()) * 100

#print chances 
print(f'Circle: ',float(chance_red_circle['shape']), '%')
print(f'Square: ',float(chance_red_square['shape']), '%')
print(f'Triangle: ',float(chance_red_triangle['shape']), '%')

################################################################

# 6 and 7) Function that calculates the side or radius of an object and placed in new column 

shape_df['side'] = ''

def calculateSide():
    shapes = [
        (shape_df['shape'] == 'circle'),
        (shape_df['shape'] == 'square'),
        (shape_df['shape'] == 'triangle')
    ]
    
    calculations = [(np.sqrt(shape_df['area']) / np.pi),(np.sqrt(shape_df['area'])),(2 * np.sqrt(shape_df['area'] / np.sqrt(3)))]
    shape_df['side'] = np.select(shapes, calculations)

    print(shape_df.head())

calculateSide()

################################################################

# 8) Boxplot showing side size distribution
# Shows that the circles generally have the smallest side/radius length, followed by square and then triangle.
plt.boxplot = shape_df.boxplot(figsize= (10,7), by= 'shape',column=['side'])

################################################################

# 9) Scatter plot of side length by shape
fig, ax = plt.subplots()
colors = {'square':'red','circle':'green','triangle':'blue'}
ax.scatter(shape_df['side'],shape_df['area'],c=shape_df['shape'].map(colors))
plt.show()

################################################################

# 10a) Dataframe of total red objects within shape
red_objects = shape_df[shape_df['color'] == 'red']

################################################################

# 10b) Proportion of blue area out of total shape area
blue_area_total = shape_df[shape_df['color'] == 'blue']
blue_area_triangle = shape_df[(shape_df['color'] == 'blue') & (shape_df['shape'] == 'triangle')]
blue_area_circle = shape_df[(shape_df['color'] == 'blue') & (shape_df['shape'] == 'circle')]
blue_area_square = shape_df[(shape_df['color'] == 'blue') & (shape_df['shape'] == 'square')]

print(f'Circle: ', (blue_area_circle['area'].sum() / blue_area_total['area'].sum()) * 100, '%')
print(f'Square: ', (blue_area_square['area'].sum() / blue_area_total['area'].sum()) * 100, '%')
print(f'Triangle: ', (blue_area_triangle['area'].sum() / blue_area_total['area'].sum()) * 100, '%')

################################################################

# 11) Function calculating proportion of area out of total shape area

def areaProportion(shape, color):
   
    #create dataframes from each shape and color
    square_df = shape_df[shape_df['shape'] == 'square']
    circle_df = shape_df[shape_df['shape'] == 'circle']
    triangle_df = shape_df[shape_df['shape'] == 'triangle']
    
    red_df = shape_df[shape_df['color'] == 'red']
    green_df = shape_df[shape_df['color'] == 'green']
    blue_df = shape_df[shape_df['color'] == 'blue']
    yellow_df = shape_df[shape_df['color'] == 'yellow']
   
    #if-else statements to assign the two variables 
    if shape == 'square':
        shape = square_df
    elif shape == 'circle':
        shape = circle_df
    elif shape == 'triangle':
        shape = triangle_df
    else:
        print('Invalid shape input')
    
    if color == 'blue':
        color = blue_df
    elif color == 'red':
        color = red_df
    elif color == 'green':
        color = green_df
    elif color == 'yellow':
        color = yellow_df
    else:
        print('Invalid color input')

    #combine shape and color into dataframe
    final_shape = shape_df[(shape['shape'] == shape & color['color'] == color)]
   
    #print result
    print(f'Proportional area is: ', (final_shape['area'] / shape['area']) * 100, '%')

areaProportion('square','yellow')