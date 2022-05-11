#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:54:33 2022

@author: sameerkarim
"""

class Patient(object):
    '''Medical center patient'''
    
    status = 'patient'
    
    def __init__(self,name,age):
                
        self.name = name
        self.age = age
        self.conditions = []
        
    def get_details(self):
        print(f'Patient record: {self.name}, {self.age} years.' \
              f' Current information: {self.conditions}')
        
    def add_info(self,information):
        self.conditions.append(information)
        
sameer = Patient('Sameer Karim',27)
amna = Patient('Amna Anwar',28)

class Infant(Patient):
    '''Patient under 2 years'''
    
    def __init__(self, name, age):
        self.vaccinations = []
        super().__init__(name, age) 
        
    def add_vac(self, vaccine):
        self.vaccinations.append(vaccine)
        
    def get_details(self):
        print(f'Patient record: {self.name}, {self.age} years.' \
              f' Current information: {self.conditions}' \
                  f' Patient has had {self.vaccinations} vaccines.' \
                      f' \n{self.name} is an infant, have they had all of their checks?')
            
archie = Infant('Archie Fittleworth',0)
archie.add_vac('MMR')

