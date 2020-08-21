#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:42:22 2020

@author: tbeleyur
"""
import sys 
sys.path.append('/home/tbeleyur/Documents/packages_dev/correct_call_annotations/')
import correct_call_annotations.correct_call_annotations as cca
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm

valid_annots = pd.read_csv('valid_annotations.csv')
valid_annots


checking_folder = 'checking_valid_annot_audio/' # the folder which will hold the spectrograms of the annotation audio

if not os.path.exists(checking_folder):
    os.mkdir(checking_folder)

    
source_audio_folder = '../individual_call_analysis/annotation_audio/'
for each in tqdm.tqdm(valid_annots['valid_annotations']):
    filepath = cca.find_file_in_folder(each+'.WAV', source_audio_folder)
    audio, fs = sf.read(filepath[0])
    channel1 = audio[:,0]
    t = np.linspace(0, channel1.size/fs, channel1.size)
    
    plt.figure()
    ax = plt.subplot(211)
    plt.specgram(channel1, Fs=fs, NFFT=256, noverlap=128);
    only_filename = os.path.split(filepath[0])[-1]
    plt.title(only_filename)
    plt.subplot(212, sharex=ax)
    plt.plot(t, channel1)
    destination_path = os.path.join(checking_folder, only_filename[:-4]+'_check.png')
    plt.savefig(destination_path)
    plt.close()
