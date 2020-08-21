#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:27:58 2020

@author: tbeleyur
"""
import sys 
sys.path.append('/home/tbeleyur/Documents/packages_dev/correct_call_annotations/')
import correct_call_annotations.correct_call_annotations as cca
import pandas as pd

source_folder = '../individual_call_analysis/annotation_audio/'
csv_file = 'silent_startstop.csv'
destination_folder = './silence_audio/'
df = pd.read_csv(csv_file)


for i in range(df.shape[0]):
    cca.write_annotation_segment_to_file(df.iloc[i,:], source_folder, get_calls=False,
                                save_channels=0,
                                file_prefix='silence_',
                                save_to=destination_folder)
