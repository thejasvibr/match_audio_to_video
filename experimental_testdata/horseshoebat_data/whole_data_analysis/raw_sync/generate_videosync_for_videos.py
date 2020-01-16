#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate video sync data for both cameras on 2018-08-16 22-24 hours
Created on Sun Nov  3 17:32:25 2019

@author: tbeleyur
"""
module_folder = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/bin/'
import glob
import sys 
sys.path.append(module_folder)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from generate_data_from_video import generate_videodata_from_videofiles, default_ROI_processing


annotation_file = '../annotations/whole_video_annotations.csv'
annotations = pd.read_csv(annotation_file)

generate_videodata_from_videofiles(annotations, 
                                   custom_processing=default_ROI_processing)
#
#sync_files = glob.glob('*.csv')
#print(sync_files)
#df = pd.read_csv(sync_files[0])
#
#plt.figure()
#plt.plot(df['led_intensity'])
#
#
#print(df['timestamp'])