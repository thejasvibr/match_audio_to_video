#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process huge audio files and make the sync channel for them 
Created on Tue Aug 13 09:05:24 2019

@author: tbeleyur
"""
import pdb
import os
import resource
import sys 
sys.path.append('../../bin/')
import glob
import numpy as np 
import soundfile as sf

import av_sync
from audio_for_videoannotation import get_file_series

raw_audio_folder ='/media/tbeleyur/THEJASVI_DATA_BACKUP_3/fieldwork_2018_002/horseshoe_bat/audio/2018-08-16/'
#raw_audio_folder = '/home/thejasvi/audio_mtpt/'
all_wav_files = glob.glob(raw_audio_folder+'*.WAV')

audiofiles = get_file_series(('684','744'), all_wav_files)

for each_file in audiofiles:
    print(each_file)
    audio, fs = sf.read(each_file)
    sync = audio[:,-1]
    print('Memory usage: %s (kB)'%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    reconstr_audio = av_sync.get_audio_sync_signal(sync, parallel=True,
                                           min_distance=int(fs*0.07*2))
    print('Memory usage: %s (kB)'%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('Done with reconstruction')
    #final_audio = np.column_stack((audio, reconstr_audio))
    file_name = os.path.split(each_file)[-1]
    sf.write('non_spikey_' + file_name, reconstr_audio,
             samplerate=fs)
    del audio, sync, reconstr_audio
    print('Memory usage: %s (kB)'%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
