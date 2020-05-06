#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:19:42 2020

@author: aimachine
"""
import numpy as np

import csv

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile

def ConvertTZXYtoNEAT(csv_file, savedir, Name):
    
    
    time, z, x, y =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)  
    
    Eventlist = np.column_stack([time, x, y, z])
    
    Event_data = [] 
    Event_data = [['Time', 'X', 'Y', 'Yes(1)No(0)']]
     
    for line in Eventlist:
        Event_data.append(line)
        writer = csv.writer(open(savedir + "/" + "Location" + Name  +".csv", "w"))
        writer.writerows(Event_data)
    
def ConvertXYTZtoNEAT(csv_file, savedir, Name):
    
    
    x, y, time, z =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)  
    
    Eventlist = np.column_stack([time, x, y, z])
    Event_data = [] 
    Event_data = [['Time', 'X', 'Y', 'Yes(1)No(0)']]
     
    for line in Eventlist:
        Event_data.append(line)
        writer = csv.writer(open(savedir + "/" + "Location" + Name  +".csv", "w"))
        writer.writerows(Event_data)    
        

def main(csv_file, savedir, Name):
    
    ConvertTZXYtoNEAT(csv_file, savedir, Name)
        
if __name__ == "__main__":

    csv_file = '/Users/aimachine/csvfiles/cell_events/MatureP1Event.csv'
    savedir = '/Users/aimachine/NEATcsvfiles'
    Path(savedir).mkdir(exist_ok = True)
    
    
    Name = 'MatureP1EventMovie2'
    
    main(csv_file,savedir,Name)