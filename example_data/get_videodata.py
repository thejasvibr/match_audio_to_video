#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Read videos of all kinds 
Created on Tue Aug 13 10:20:56 2019

@author: tbeleyur
"""
import sys
sys.path.append('..//')
import pandas as pd
from generate_data_from_video import generate_videodata_from_videofiles

annotations_df = pd.read_csv('eg_annotations.csv')
# Since it's a small video with only 1370 frames - we'll run the whole thing! This could take a couple of minutes
generate_videodata_from_videofiles(annotations_df)


