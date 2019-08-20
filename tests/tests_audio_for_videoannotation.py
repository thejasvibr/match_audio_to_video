#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Tests for the audio_for_videoannotation module
Created on Mon Aug 12 14:26:34 2019

@author: tbeleyur
"""
import sys 
sys.path.append('..//bin//')
import unittest
import string
import pandas as pd
import soundfile as sf
import scipy.signal  as signal 
import numpy as np 
import datetime as dt
from datetime import timedelta
import time 

from audio_for_videoannotation import *

class test_match_video_sync_to_audio(unittest.TestCase):
    
    def setUp(self):
        '''
        '''
        np.random.seed(82319)
        self.videosync_df = pd.DataFrame(data=[],
                                         columns=['led_intensity',
                                                  'timestamp_verified',
                                                  'annotation_block'])
        self.audio_folder = './'
        self.kwargs= {}
        self.kwargs['audio_fs'] = 250000
        self.kwargs['audio_fileformat'] = '*testing.wav'
        self.kwargs['contiguous_files'] = True
        self.kwargs['audio_sync_spikey'] = False
        self.kwargs['common_fps'] = 25.0
        
        
        
        #create the video sync signal first : 
        fs = self.kwargs['audio_fs']
        
        durns = np.arange(0.2,0.5,0.050)
        
        onoff_signal = []
        total_durn = 0
        while total_durn <= 90.0:
            on = np.random.choice(durns,1)
            on_samples = int(fs*on)
            off = np.random.choice(durns,1)
            off_samples =int(fs*off)
            onoff_signal.append(np.concatenate((np.ones(on_samples),
                                                np.zeros(off_samples))))
            total_durn += on + off
        
        audio_onoff_signal = np.concatenate(onoff_signal)
        audio_onoff_signal *= 0.8
        just_noise = np.random.normal(0,10**(-40/20.0), audio_onoff_signal.size)
        just_noise /= np.max(just_noise)
        just_noise *= 0.5

        two_channel_audio = np.column_stack((just_noise,
                                             audio_onoff_signal))
        # write the audio signal to a file : 
        sf.write('audio_rec'+self.kwargs['audio_fileformat'][1:],
                 two_channel_audio,samplerate=fs)
        
        # recreate the video signal that corresponds to it by downsampling 
        # the first 5 seconds 
        vid_fps = 25
        downsampling_factor = int(fs/float(vid_fps))
        every_10000th_value = np.arange(0,audio_onoff_signal.shape[0],
                                        downsampling_factor)
        video_sync_durn = 20.0
        video_sync = audio_onoff_signal[every_10000th_value][int(vid_fps*video_sync_durn):int(vid_fps*video_sync_durn)*2]
        
        self.videosync_df['led_intensity'] = video_sync
        self.videosync_df['annotation_block'] = True
        self.videosync_df['timestamp_verified'] = self.make_timestamps(video_sync_durn,vid_fps )
        
        self.videosync_df.to_csv('common_fps_TESTING.csv')
        print('setup complete')

    def make_timestamps(self, duration, fps=25):
        '''
        Parameters
        ----------
        duration : duration over which to make timestmaps in seconds.
                  
        fps : video frames per seoncs

        Returns
        ------
        timestamps : timestamps in YYYY-mm-dd HH:MM:SS format which 
                     match the duration according the fps. 
        '''
        rounded_duration = np.ceil(duration)
        start_time = '2018-09-04 23:00:00'
        start = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        rounded_end = start + timedelta(seconds=rounded_duration)
        rounded_end_time = rounded_end.strftime('%Y-%m-%d %H:%M:%S')
        kw = {}
        kw['timestamp_pattern'] = '%Y-%m-%d %H:%M:%S'
        time_between = self.make_timestamps_in_between(start_time, 
                                                       rounded_end_time,
                                                       **kw)
        time_between_human_readable = [ self.datetime_from_posix(each, **kw) for each in time_between]
        
        # repeat the timestmaps 
        all_timestamps_repeated = []
        for each in time_between_human_readable:
            all_timestamps_repeated.append(np.tile(each, fps))

        all_timestamps = np.concatenate(all_timestamps_repeated)
        # take only the ones that are relevant to the duration
        num_frames = int(fps*duration)
        timestamps_durnrelevant = all_timestamps[:num_frames]
        return(timestamps_durnrelevant)
            


    
    def make_timestamps_in_between(self, start, end, **kwargs):
        '''Get timestamps in one second granularity and output POSIX tiemstamps
        for nice matching and sequence generation. 
    
        ACHTUNG : THIS ONLY HANDLES ONE SECOND GRANULARITY RIGHT NOW !!!!
        '''
        # generate posix timestamps between start and end times
        posix_start = self.make_posix_time(start, **kwargs)
        posix_end = self.make_posix_time(end, **kwargs)
        posix_values = np.arange(posix_start, posix_end+1)
        return(posix_values)
            
    
    
    def make_posix_time(self,timestamp, **kwargs):
        '''
        '''
        dt_timstamp = dt.datetime.strptime(timestamp, kwargs['timestamp_pattern'])
        posix_time = time.mktime(dt_timstamp.timetuple())
        return(posix_time)
    
    def datetime_from_posix(self, posix_ts, **kwargs):
        '''thanks https://stackoverflow.com/a/3682808/4955732
        '''
        readable_datetime = dt.datetime.fromtimestamp(posix_ts).strftime(kwargs['timestamp_pattern'])
        return(readable_datetime)




    def test_check_number_samples(self):
        '''        
        '''
        matched_audio, _, _ = match_video_sync_to_audio(self.videosync_df, 
                                  './', **self.kwargs)
        exp_duration = sum(self.videosync_df['annotation_block'])/self.kwargs['common_fps']
        exp_samples = int(self.kwargs['audio_fs']*exp_duration)
        
        obtained_samples, _ = matched_audio.shape
        self.assertEqual(exp_samples, obtained_samples)
        
        sf.write('test_results_1.WAV',matched_audio,samplerate=self.kwargs['audio_fs'])
    
    def test_number2_check_number_samples(self):
        ''' set annotation block to only 2 seconds length
        
        '''
        self.videosync_df['annotation_block']= False
        self.videosync_df.loc[20:45,'annotation_block'] = True
        
        matched_audio, _, _ = match_video_sync_to_audio(self.videosync_df, 
                                  './', **self.kwargs)
        exp_duration = sum(self.videosync_df['annotation_block'])/self.kwargs['common_fps']
        exp_samples = int(self.kwargs['audio_fs']*exp_duration)
        
        obtained_samples, _ = matched_audio.shape
        self.assertEqual(exp_samples, obtained_samples)
        
        sf.write('test_results_2.WAV',matched_audio,samplerate=self.kwargs['audio_fs'])
        

class CheckingIfFileSubsettingWorks(unittest.TestCase):
    '''
    '''
    def setUp(self):
        '''
        '''
        self.all_files_numeric = [ 'T0000'+str(i)+'.WAV'   for i in range(100,200)]
        self.all_files_alphabetic = ['T0000'+each+'.WAV' for each in string.ascii_lowercase]
    
    def test_simple_numeric(self):
        fname_range = ('100','180')
        matched_files  = get_file_series(fname_range, self.all_files_numeric)
        print(matched_files,'MIAOW')
        self.assertEqual(len(matched_files), 81)
        
        self.assertEqual(matched_files[0],self.all_files_numeric[0])
    
    def test_simple_alphabets(self):
        fname_range = ('a','z')
        matched_files  = get_file_series(fname_range, self.all_files_alphabetic)
        
        self.assertEqual(len(matched_files), 26)
        self.assertEqual(matched_files[0],self.all_files_alphabetic[0])

    def test_wrong_entry(self):
        fname_range = ('!@#','!%^')
        with self.assertRaises(CheckFileRangeID) as context:
            get_file_series(fname_range, self.all_files_numeric)
        
        self.assertTrue('Unable to match' in context.exception[0])
                       
    def test_overgeneral_entry(self):
        fname_range = ('74','9')
        with self.assertRaises(CheckFileRangeID) as context:
            get_file_series(fname_range, self.all_files_numeric)
        
        
        self.assertTrue('Multiple matching entries found' in context.exception[0])
        
        
            
            
if __name__ =='__main__':
    unittest.main()
        
        
                
            

