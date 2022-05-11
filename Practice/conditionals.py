#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:29:54 2022

@author: sameerkarim
"""

temp = int(input('input a temperature in Celsius, between 0-40 >>> '))

if temp > 30 :
    print('Wear shorts!')
elif temp <= 30 and  temp > 20:
    print('It\'s warm, but not quite shorts weather.')
elif temp <= 20 and temp > 10:
    print('You\'ll probably need a vest today.')
else:
    print('Too cold to go out!26')