#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Tests for process_video_annotations 
Created on Wed Aug 14 12:47:58 2019

@author: tbeleyur
"""
from __future__ import division

import sys 
sys.path.append('..//bin//')
import unittest
import pandas as pd
import soundfile as sf
import numpy as np 
import datetime as dt
from datetime import timedelta
from process_video_annotations import *


class TestingAttachAnnotation(unittest.TestCase):
    
    def setUp(self):
        # create the videosync data
        # attach_annotation_start_and_end(video_sync_data, target_timestamps,
                                        #videosync_commonfps,
                                        #**kwargs):
        self.video_sync_data = pd.DataFrame(data=[], columns=['timestamp_verified',
                                                   'led_intensity'])
        start = '2019-01-01 10:00:00' 
        stop = '2019-01-01 10:01:00'
        self.video_sync_data['timestamp_verified'] = self.make_timestamps_for_testcase(start,
                                                       stop,
                                                       timestamp_pattern='%Y-%m-%d %H:%M:%S')
        
        self.video_sync_data['led_intensity'] = np.random.normal(0,0.1,self.video_sync_data.shape[0])

        self.target_timestamps = {}
        self.target_timestamps['annotation_block'] = pd.DataFrame(data={'start_timestamp':['2019-01-01 10:00:03'],
                                                     'start_framenumber':[5], 
                                                     'end_timestamp':['2019-01-01 10:00:03'],
                                                     'end_framenumber':[20]}).iloc[0,:]
        self.video_commonfps = pd.DataFrame(data=[], columns=['timestamp_verified',
                                                   'led_intensity'])
    
        sync_timestamps = ['2019-01-01 10:00:03','2019-01-01 10:00:04',
                           '2019-01-01 10:00:05','2019-01-01 10:00:06']    
        self.video_commonfps['timestamp_verified'] = [each for each in sync_timestamps for i in range(25)]
        

    def make_timestamps_for_testcase(self,start,stop, **kwargs):
        '''
        '''
        posix_times = make_timestamps_in_between(start,stop,
                                                **kwargs)
        human_readable = [datetime_from_posix(each, **kwargs) for each in posix_times]
        
        frame_rate_variation = np.tile([20,21,22],20)
        repeated_timestamps = []
        for eachframerate, timestamp in zip(frame_rate_variation, human_readable):
            repeated_timestamps.append(np.tile(timestamp, eachframerate))
        
        repeated_timestamps = np.concatenate(repeated_timestamps)
        return(repeated_timestamps)
        

    
    def test_checkproper_annotation_start(self):
        '''Check if the annotation block calculations work as they should 
        '''
        attach_annotation_start_and_end(self.video_sync_data, 
                                        self.target_timestamps,
                                        self.video_commonfps,
                                        timestamp_pattern='%Y-%m-%d %H:%M:%S',
                                        common_fps=25)
        start_relative_position = self.target_timestamps['annotation_block']['start_framenumber']/20
        expected_start_position = int(np.floor(start_relative_position*25))
        obstained_start_rel_pos= int(min(np.argwhere(self.video_commonfps['annotation_block']==True)))

        self.assertEqual(expected_start_position, obstained_start_rel_pos)
    
    def test_checkproper_annotation_end(self):
        '''Check if the annotation block calculations work as they should 
        '''
        self.target_timestamps['annotation_block'] = pd.DataFrame(data={'start_timestamp':['2019-01-01 10:00:03'],
                                                     'start_framenumber':[1], 
                                                     'end_timestamp':['2019-01-01 10:00:03'],
                                                     'end_framenumber':[19]}).iloc[0,:]
        attach_annotation_start_and_end(self.video_sync_data, 
                                        self.target_timestamps,
                                        self.video_commonfps,
                                        timestamp_pattern='%Y-%m-%d %H:%M:%S',
                                        common_fps=25)
        end_relative_position = self.target_timestamps['annotation_block']['end_framenumber']/20
        expected_end_position = int(np.floor(end_relative_position*25))-1
        obstained_end_rel_pos= int(np.max(np.argwhere(self.video_commonfps['annotation_block']==True)))
        self.assertEqual(expected_end_position, obstained_end_rel_pos)
    
    def test_check_bigger_annotation(self):
        '''a longer annotation 
        '''
        self.target_timestamps['annotation_block'] = pd.DataFrame(data={'start_timestamp':['2019-01-01 10:00:03'],
                                                     'start_framenumber':[5], 
                                                     'end_timestamp':['2019-01-01 10:00:05'],
                                                     'end_framenumber':[19]}).iloc[0,:]
        number_of_frames_in_annotation_per_second = np.array([15/20.0, 21/21.0,
                                                              19/22.0])*25
        exp_frames_in_annotation = int(np.floor(sum(number_of_frames_in_annotation_per_second)))

        attach_annotation_start_and_end(self.video_sync_data, 
                                        self.target_timestamps,
                                        self.video_commonfps,
                                        timestamp_pattern='%Y-%m-%d %H:%M:%S',
                                        common_fps=25)
        
        obtainednumframes_in_annotation = np.sum(self.video_commonfps['annotation_block'])
        self.assertEqual(exp_frames_in_annotation,obtainednumframes_in_annotation)
        
        
        



class TestCalcRelativeFramePosition(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.df = pd.read_csv('videosync_test_data.csv')
        self.frame_num = 1
        self.ts_of_interest = '2019-07-04 16:51:40'
        self.all_timestamps = self.df['timestamp_verified']
        
        self.num_ts = np.sum(self.ts_of_interest==self.all_timestamps)
        
    
    def test_basic(self):
        '''
        '''
        
        expected_relative_position = self.frame_num/float(self.num_ts)
        
        obtained_relative_position = calculate_relative_frame_position(self.frame_num, 
                                                                       self.ts_of_interest,
                                                                       self.all_timestamps)

        self.assertEqual(obtained_relative_position, expected_relative_position)

    def test_zeroframe(self):
        self.frame_num = 0
        with self.assertRaises(InvalidFramenumber) as context:
            calculate_relative_frame_position(self.frame_num, self.ts_of_interest,
                                              self.all_timestamps)

        self.assertTrue('Framenumbers must be >=1' in context.exception[0])

    def test_toomanyframes(self):
        self.frame_num = 90
        with self.assertRaises(InvalidFramenumber) as context:
            calculate_relative_frame_position(self.frame_num, self.ts_of_interest,
                                              self.all_timestamps)

        self.assertTrue('Given frame number is more than the' in context.exception[0])
    
    def test_foreigntimestamps(self):
        self.ts_of_interest = '2018-02-05 12:02:10'
        with self.assertRaises(NoMatchFound) as context:
            calculate_relative_frame_position(self.frame_num, self.ts_of_interest,
                                              self.all_timestamps)

        self.assertTrue('Unable to find a matching timestamp for' in context.exception[0])

class CheckBadAnnotations(unittest.TestCase):
    def setUp(self):
            # create the videosync data
            # attach_annotation_start_and_end(video_sync_data, target_timestamps,
                                            #videosync_commonfps,
                                            #**kwargs):
            self.video_sync_data = pd.DataFrame(data=[], columns=['timestamp_verified',
                                                       'led_intensity'])
            start = '2019-01-01 10:00:00' 
            stop = '2019-01-01 10:01:00'
            self.video_sync_data['timestamp_verified'] = self.make_timestamps_for_testcase(start,
                                                           stop,
                                                           timestamp_pattern='%Y-%m-%d %H:%M:%S')
            
            self.video_sync_data['led_intensity'] = np.random.normal(0,0.1,self.video_sync_data.shape[0])
    
            self.annotation = pd.DataFrame(data={'start_timestamp':['2019-01-01 10:00:00'],
                                                 'start_framenumber':[10],
                                                 'end_timestamp':['2019-01-01 10:00:01'],
                                                 'end_framenumber':[15],
                                                 'annotation_id':[2201]})
    def make_timestamps_for_testcase(self,start,stop, **kwargs):
        '''
        '''
        posix_times = make_timestamps_in_between(start,stop,
                                                **kwargs)
        human_readable = [datetime_from_posix(each, **kwargs) for each in posix_times]
        
        frame_rate_variation = np.tile([20,21,22],20)
        repeated_timestamps = []
        for eachframerate, timestamp in zip(frame_rate_variation, human_readable):
            repeated_timestamps.append(np.tile(timestamp, eachframerate))
        
        repeated_timestamps = np.concatenate(repeated_timestamps)
        return(repeated_timestamps)
        
    
    def test_poor_starttimestamp(self):
        '''
        '''
        self.annotation['start_timestamp'][0] = '2019-01-01 9:00:00'
        video_sync_over_annotation_block(self.annotation.loc[0,:], self.video_sync_data,
                                         timestamp_pattern = '%Y-%m-%d %H:%M:%S',
                                         min_durn = 10)
        
    
    
    


if __name__ == '__main__':
    unittest.main()
        
        
