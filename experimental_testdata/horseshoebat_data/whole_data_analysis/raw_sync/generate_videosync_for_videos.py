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
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import pandas as pd
import scipy.signal as signal 


from generate_data_from_video import generate_videodata_from_videofiles, segment_numbers_and_resize

video_annotation_files = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/whole_data_analysis/'
annotation_files = glob.glob(video_annotation_files+'annotations/'+'*.csv')
print(annotation_files)
file_index = 0
annotations = pd.read_csv(annotation_files[file_index])
print('processing...', annotation_files[file_index][-40:])   

STARTFRAME = 32000

generate_videodata_from_videofiles(annotations, 
                                   custom_processing=segment_numbers_and_resize,
                                   final_dims=(1400,200),
                                   numeric_pixel_threshold=225,
                                   read_timestamp_and_lamp=(False, True),
                                   start_frame=STARTFRAME,
                                   end_frame=STARTFRAME+100)
#
sync_files = glob.glob('*.csv')
print(sync_files)
df = pd.read_csv(sync_files[0])
#
#standard_led = df_npsum['led_intensity']
#
plt.figure()
plt.plot(df['led_intensity'])
plt.grid()
plt.show()
print(df['timestamp'])