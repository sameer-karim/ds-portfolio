#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:48:47 2022

@author: sameerkarim
"""

class Card(object):
    
    suits = {'d':'Diamonds', 'c':'Clubs', 'h':'Hearts', 's':'Spades'}
    value = {1:'Ace', 11:'Jack', 12:'Queen', 13:'King'}
    
    def __init__(self,value,suit):
          self.value = value
          self.suit = suit
        
    def get_value(self):
        return self.value
    
    def get_suit(self):
        return self.suit
    
    def __str__(self):
        
        my_card = str(self.value) + ' of ' + str(self.suit)
        return my_card