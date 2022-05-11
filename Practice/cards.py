#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:48:47 2022

@author: sameerkarim
"""

class Card(object):
    
    
    def __init__(self,value,suit):
          self.value = value
          self.suit = suit
        
    def get_value(self):
        return self.value
    
    def get_suit(self):
        return self.suit
    
    def __str__(self):
        
        suits = {'d':'Diamonds', 'c':'Clubs', 'h':'Hearts', 's':'Spades'}
        value = {1:'Ace', 11:'Jack', 12:'Queen', 13:'King'}
        
        my_card = str(value[self.value]) + ' of ' + str(suits[self.suit])
        return my_card