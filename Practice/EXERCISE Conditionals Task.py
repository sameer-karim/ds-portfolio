#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:04:18 2019

@author: Giles
"""

'''
Question 1
Write code that asks the user to input a number between 1 and 5 inclusive.
The code will take the integer value and print out the string value. So for
example if the user inputs 2 the code will print two. Reject any input that
is not a number in that range
'''

# user_int = input('Enter a number between 1 and 5 here -> ')

# if user_int >= 1 and user_int <= 5:
#     print(user_int)
# else:
#     print('This is not a number between 1 and 5. Try again.')


'''
Question 2
Repeat the previous task but this time the user will input a string and the
code will ouput the integer value. Convert the string to lowercase first.
'''
# user_input = input('Please enter an string between One and five:> ')
# user_input = user_input.lower()
# if user_input == 'one':
#     print(1)
# elif user_input == 'two':
#     print(2)
# elif user_input == 'three':
#     print(3)
# elif user_input == 'four':
#     print(4)
# elif user_input == 'five':
#     print(5)
# else:
#     print('Out of range')


'''
Question 3
Create a variable containing an integer between 1 and 10 inclusive. Ask the
user to guess the number. If they guess too high or too low, tell them they
have not won. Tell them they win if they guess the correct number.
'''
# x = 6
# user_input2 = int(input('Guess the lucky number between 1 and 10 -> '))

# if user_input2 == x:
#     print('You win! Congratulations.')
# else:
#     print('That\'s not it. Try again.')


'''
Question 4
Ask the user to input their name. Check the length of the name. If it is
greater than 5 characters long, write a message telling them how many characters
otherwise write a message saying the length of their name is a secret
'''
# user_input = input('Enter your name -> ')

# if len(user_input) > 5:
#     print('Your name is ' + str(len(user_input)) + ' characters long.')
# else:
#     print('The length of your name is a secret!')


'''
Question 5
Ask the user for two integers between 1 and 20. If they are both greater than
15 return their product. If only one is greater than 15 return their sum, if
neither are greater than 15 return zero
'''
# user_x = int(input('Enter 2 integers between 1 and 20. \n Integer 1 -> '))
# user_y = int(input('Integer 2 -> '))

# if user_x > 15 and user_y > 15:
#     print(user_x * user_y)
# elif user_x <= 15 and user_y > 15:
#     print(user_x + user_y)
# elif user_x > 15 and user_y <= 15:
#     print(user_x + user_y)
# elif user_x <= 15 and user_y <= 15:
#     print(0)
# else:
#     print('Error inputting integers')
    
'''
Question 6
Ask the user for two integers, then swap the contents of the variables. So if
var_1 = 1 and var_2 = 2 initially, once the code has run var_1 should equal 2
and var_2 should equal 1.
'''
# user_x = int(input('Enter two integers.\nInteger 1 -> '))
# user_y = int(input('Integer 2 -> '))

# user_x, user_y = user_y, user_x

# print(user_x)
# print(user_y)

False or not (7 < 3)