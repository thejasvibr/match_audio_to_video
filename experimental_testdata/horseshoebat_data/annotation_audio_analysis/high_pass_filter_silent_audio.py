#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:42:25 2020

@author: tbeleyur
"""

import soundfile as sf
import glob
import os 
import scipy.signal as signal 

all_files_in_folder = glob.glob('silence_audio/'+'*.WAV')

# check if a hp_silence_audio folder exists already

# the same highpass filter used in the single call analysis (See 'Making high-pass copy of...')
b,a = signal.butter(2, 70000/125000, 'highpass') 

hp_audio_folder = 'hp_silence_audio/'

if os.path.exists(hp_audio_folder):
    pass
else:
    os.mkdir(hp_audio_folder)
    
for each in all_files_in_folder:
    try:
        audio, fs = sf.read(each)
        hp_audio = signal.filtfilt(b,a, audio)
        only_filename = os.path.split(each)[-1]
        file_name, file_format =  only_filename.split('.')
        hp_file_name = file_name + '_hp'
        new_file_name = hp_file_name +'.'+file_format
        final_file_path = os.path.join(hp_audio_folder, new_file_name)
        sf.write(final_file_path, hp_audio, fs)
    except:
        print(f'Could not process: {each}')
        
        
# There are some silent regions which could not be filtered properly because 
# of erroneous data entry (start/stop times)

# silence_matching_annotaudio_Aditya_2018-08-17_23_173.WAV : stop time before start
# silence_matching_annotaudio_Aditya_2018-08-16_21502300_9.WAV : start / stop time beyond current file length. 
#   Aditya also notes that the silent region is taken from another file. 
# silence_matching_annotaudio_Aditya_2018-08-19_0120-0200_90.WAV : silent 
#           interval taken from another file (see above)
# silence_matching_annotaudio_Aditya_2018-08-17_23_70.WAV : silent 
#           interval taken from another file (see above)
# silence_matching_annotaudio_Aditya_2018-08-19_0120-0200_117.WAV : typo in start time. Start time is > length of audio