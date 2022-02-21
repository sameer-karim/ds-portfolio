# from turtle import color
# import pandas as pd
# import matplotlib.pyplot as plt

# x = []
# y = []

# for data in open('dot-to-dot.txt', 'r'):
#     values = [float(s) for s in data.split()]
#     x.append(values[0])
#     y.append(values[1])


# plt.plot(x, y, color = 'blue')

# plt.show()

# l = [0, 1, 4, 5, 7, 7, 3, 4, 2]

# print(l[-4:])

import random
random.seed = 365

# Mapping of numbers to rock, paper & scissors
int2choice = {0:'R', 1:'P', 2:'S'}

# Class to play the game
class RPS_Game(object):
    
    # Initialize the game
    def __init__(self, num_rounds):
        self.num_rounds = num_rounds # record the total number of rounds
        self.counter_pc_wins = 0     # how many rounds are won by the PC
        self.draws = 0               # how many rounds end in a draw
        print(f"Start of Rock-Paper-Scissors Game - {num_rounds} rounds.")


