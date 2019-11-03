#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate video sync data for both cameras on 2018-08-16 22-24 hours
Created on Sun Nov  3 17:32:25 2019

@author: tbeleyur
"""
module_folder = '/home/tbeleyur/Documents/packages_dev/match_audio_to_video/bin/'
import sys 
sys.path.append(module_folder)
import pandas as pd

from generate_data_from_video import generate_videodata_from_videofiles

annotation_file = '../annotations/22-24h.csv'
annotations = pd.read_csv(annotation_file)

generate_videodata_from_videofiles(annotations)
