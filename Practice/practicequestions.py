# Question 1
# Can you write a short program that will print out the version of Python
# that you are using?

# import sys

# py = sys.version
# print('Using ',py)

################################

# Question 2
# Write a program that requests five names separated by commas and create a
# list containing those names. Print your answer.
# For example James,Alison,Fred,Sally,Matthew
# should return ['James','Alison','Fred','Sally','Matthew']

# names = input('Enter 5 names here -> ')
# name_list = names.split(',')
# print(name_list)

################################

# Question 3
# Write a program to determine whether a given number is within 10 of 100 or 200.

# def Number():
#     a = int(input('Enter an integer -> '))

#     if 90 <= a <= 110 or 190 <= a <= 210:
#         print(f'Your integer, {a}, is within 10 of 100 or 200')
#     else:
#         print(f'Your integer, {a}, is not within 10 of 100 or 200')

# Number()

################################

# Question 4
# Write a program that takes a list of non-negative integers and prints each integer
# to the screen the same number of times as the value of the integer, each new value
# on a new line. For example
# [2,3,4,1] would print:
# 22
# 333
# 4444
# 1

# def print_nums(number_list):

#     while item > 0:
#         for item in number_list:
#             for i in len(number_list):
#                 number_list[i+1]
#             value = str(item) * item
#             print(value)


# print_nums([9,3,6])

################################

# Question 5
# Write some code that will return the number of CPUs in the system.

# import multiprocessing

# print(f'The amount of CPUs in this computer is {multiprocessing.cpu_count()}')

################################

# Question 6
# Write a program that will return the sum of the digits of an integer.

# def getSum(a):

#     sum = 0
#     for digit in str(a):
#         sum += int(digit)
#     print(f'The sum of the digits is {sum}')

# getSum(84848)

################################

# Question 7
# Write a program that converts text into pig latin. Pig latin works as follows:
# All letters before initial vowel are placed at the end of the word and then 'ay'
# is added (explanation adapted from Wikipedia), so pig becomes igpay, cat becomes
# atcay, potential becomes otentialpay etc.


# s = input('Enter a sentence -> ').lower()
# words = s.split()

# for i, word in enumerate(words):
#     if word[0] in 'aeiou':
#         words[i] = words[i] + 'ay'
#     else:
#         has_vowel = False
#         for j, letter in enumerate(word):
#             if letter in 'aeiou':
#                 words[i] = word[j:] + word[:j] + 'ay'
#                 has_vowel = True
#                 break

#         if(has_vowel == False):
#             words[i] = words[i] + 'ay'

# pig_latin = ' '.join(words)
# print('Pig Latin: ',pig_latin)

################################

# Question 8
# Write a function that will check for the occurrence of double letters in
# a string. If the string contains double letters next to each other it
# will return True, otherwise it will return False.

# def checkLetters(word):
#     for i in range(len(word) - 1):
#         if word[i] == word[i+1]:
#             return True
#     return False

# print(checkLetters('aa'))

################################

# Question 9
# Write a function that will check if a string is a palindrome.

# def is_palindrome(word):
#     if word == word[::-1]:
#         print('Yes')
#     print('No')

# print(is_palindrome('hello'))

################################

# Question 10
# Write a function def add_commas(numbers) that will add commas to an integer and return it as a string.
# For example add_commas(1000000) will return 1,000,000 Do it first without using string fomratting
# or f strings.

# def add_commas():
#     numbers = "{:,}".format(20000000)
#     print(numbers)

# add_commas()

################################

# Question 11
# Write a function that will convert an integer into binary.

# def convertToBinary(number):
#     binary = bin(number)
#     print(f'{number} in binary is {binary}')

# convertToBinary(19382732)

################################

# Question 12
# Write a function that calculates the sum of all integers up to n. Use the iterative method
# and the formula and compare the results. (sum of n integers given by S = (n(n+1))/2)

# def check_sums(number):
#     sum_1 = 0
#     for i,v in enumerate(range(number+1)):
#         sum_1 = sum_1 + v
#     sum_2 = (number * (number + 1))/2

#     return sum_1, sum_2

# print(check_sums(3867))

################################

# Question 13/14
# Implement the TwoSum solution without referring to the solution.

# def twoSum(nums, target):
#     d = {}

#     for i in range(len(nums)):
#         if target - nums[i] in d:
#             print(d)
#             return [d[target - nums[i]], i]
#         d[nums[i]] = i

#     return -1

# L = [12,2,54,66,4,8]
# print(twoSum(L,12))

################################

# Question 15
# Write a function that takes a positive integer n and converts it into hours and minutes.
# 45 would return 0h:45mins 135 would return 2h:15mins

# from turtle import pd

# def getTime(n):
#     h = n / 60
#     m = n % 60
#     print(f'{int(h)}h:{m}mins')

# getTime(360)

################################

# Question 16
# Write a function to determine whether all numbers in a list are unique.

# import numpy as np

# def uniqueList(list):
#     if(len(set(list)) == len(list)):
#         print('List is unique')
#     else:
#         print('List is NOT unique')

# list1 = [3,4,6,7,1]
# print(uniqueList(list1))

################################

# Question 17
# Write a function to add two positive integers together without using the + operator.
# (Note, this will require some research - start here https://en.wikipedia.org/wiki/Bitwise_operation)

# def getSum(a,b):
#     return a+b

# print(getSum(2,6))

################################


