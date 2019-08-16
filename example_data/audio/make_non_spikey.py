#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process huge audio files and make the sync channel for them 
Created on Tue Aug 13 09:05:24 2019

@author: tbeleyur
"""
import os
import sys 
sys.path.append('../../')
import glob
import numpy as np 
import soundfile as sf

import av_sync

raw_audio_folder = '../../../usbdrive_point/'

audiofiles = glob.glob(raw_audio_folder+'*07.WAV')

for each_file in audiofiles:
    print(each_file)
    audio, fs = sf.read(each_file)
    sync = audio[:,-1]
    reconstr_audio = av_sync.get_audio_sync_signal(sync, parallel=True,
                                           min_distance=int(fs*0.07*2))
    print('Done with reconstruction')
    final_audio = np.column_stack((audio[:,0],sync, reconstr_audio))
    file_name = os.path.split(each_file)[-1]
    sf.write('non_spikey_' + file_name, final_audio,
             samplerate=fs)