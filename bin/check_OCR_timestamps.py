#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Timestamp OCR error checking code.
Display the timestamps and their occurence frequency. 
IFFF the OCR is good:
Timestamps that occur very few times are likely to be
    >well-recognised start/end timestamps
    >badly recognised timestamps
Timestamps that occur unusually often are likely to be 
    >genuine repeats of the video frames
    >badly recognised timestamps


Created on Mon Nov  4 06:44:32 2019

@author: tbeleyur
"""

import datetime as dt
import os
import time

import easygui as eg
import numpy as np
import pandas as pd 

from process_video_annotations import make_posix_time

def detect_unusual_timestamps(raw_timestamps,datetime_format, 
                                  **kwargs):
    '''
    Identifies the bad and wrong reads of a series of timestamps 
    meant to be in chronlogical order. The timestamps 
    
    The function identifies two types of misreads:
        unparsable timestamps : those that cannot be parsed because 
            they are very badly recognised during OCR.
            eg.'PAW 03efgg:20:12' or '2018-09-03 2220:20:12'
            instead of '2018-09-03 22:20:12' 

        abrupt timestamps : those that appear abruptly in 
            and do not match the chronological
            order. These are typically reads 
            which are mostly correct except for
            one or two wrongly read characters.
            eg. '2018-08-09 22:00:09' suddenly appearing 
            after a series of '2018-08-09 22:00:06'
    
    An output is given that identifies candidate bad and wrong timestamps.
    
    PLEASE NOTE: this function only uses basic heuristics, it is always 
    important to check the raw data as sometimes bad/wrong reads may actually
    be in the data itself (eg. frame drops, garbled images)
    
    Parameters
    ----------
    raw_timestamps : column of a pd.DataFrame with strings. 
                     EAch entry is a timestamp eg. 
                     '2019-08-07 22:00;10'
                     or
                     '10:00 31-12-2017'

    datetime_format : str.
                    The format in which the timestamps are set in. 

    Keyword Arguments
    -----------------        
    timedelta : time_unit and step size. 
                       eg. if the timestamps change each second, the entry would be
                       seconds=1
                       OR
                       if the timestamp changes every 3 minutes
                       minutes=3
                         
               
    
    Returns
    --------
    odd_and_bad_timestamps : pd.Series with strings.
                            The index location of an odd/bad entry is labelled
                            with 'VERIFY'. All other entries that seem fine 
                            are labelled with 'seemsfine'.

    '''
    number_of_timestamps = len(raw_timestamps)
    odd_and_bad_timestamps = pd.DataFrame(index=xrange(number_of_timestamps),
                                          columns=['user_suggestion'])
    odd_and_bad_timestamps['user_suggestion'] = 'maybeokay'
   
    # identify all un-parsable timestamps (nonsense timestamps)
    posix_timestamps = np.zeros(number_of_timestamps)
    print('..........Checking for bad timestamps..........')
    for i, each in enumerate(raw_timestamps):
        try:
            posix_timestamps[i] = make_posix_time(each, 
                                            timestamp_pattern=datetime_format)
        except:
            posix_timestamps[i] = np.nan
  
    unparsable_timestamps = np.isnan(posix_timestamps)
    odd_and_bad_timestamps.loc[unparsable_timestamps,'user_suggestion'] = 'VERIFY'

    # detect all odd-jump timestamps (parsable but wrong reads)
    print('..........Checking for odd-jumps between timestamps..........')
    odd_jumps = check_for_odd_jumps(posix_timestamps, **kwargs)
    odd_and_bad_timestamps.loc[odd_jumps,'user_suggestion'] = 'VERIFY'
    return(odd_and_bad_timestamps)    

def check_for_odd_jumps(posix_timestamps, **kwargs):
    '''
    '''
    num_timestamps = len(posix_timestamps)
    odd_jumps = np.zeros(num_timestamps, dtype=bool)

    valid_posix_timestamps = get_one_valid_posix_timestamp(posix_timestamps)

    expected_time_difference = get_expected_posix_jump(valid_posix_timestamps, **kwargs)
    
    # a timestamp can be followed by itself or a new one
    expected_timedifferences = set([0,expected_time_difference])
    
    for i, (one_timestamp, next_timestamp) in enumerate(zip(posix_timestamps[:-1],
                                                     posix_timestamps[1:])):
        timegap = next_timestamp - one_timestamp

        if timegap in expected_timedifferences:
            pass
        else:
            odd_jumps[i] = True
            odd_jumps[i+1] = True
    return(odd_jumps)

def get_one_valid_posix_timestamp(many_timestamps):
    '''
    Extracts one single valid timestamp. 
    Parameters
    ----------
    many_timestamps : (Ntimestamps,) np.array 
                     The entries are either np.nan or POSIX timestamps

    '''
    invalid_entries = np.isnan(many_timestamps)
    valid_indices = np.invert(invalid_entries)
    valid_entries = many_timestamps[valid_indices]
    one_valid_entry = valid_entries[0]
    return(one_valid_entry)

def get_expected_posix_jump(one_posix_timestamp, **kwargs):
    '''
    '''
    one_timestamp = dt.datetime.fromtimestamp(one_posix_timestamp)
    next_timestamp = one_timestamp + dt.timedelta(**kwargs)
    # convert to POSIX
    next_posix_timestamp = time.mktime(next_timestamp.timetuple())
    # get expected difference in timestamp on POSIX scale
    posix_timestamp_difference = next_posix_timestamp - one_posix_timestamp
    return(posix_timestamp_difference)

if __name__ =='__main__':
    video_sync_raw_path = eg.fileopenbox('Which file needs  timestamp-checking?',
                                       title='Choose video_sync file')
    raw_video_sync = pd.read_csv(video_sync_raw_path)
    
    timestamp_column = eg.choicebox('Which column to summarise?', 
                                        choices=['timestamp',
                                                 'timestamp_verified'])
    
    #timestamp_format = eg.textbox('Enter the timestamp format')
    timestamp_format = '%Y-%m-%d %H:%M:%S'
    unusual_timestamps = detect_unusual_timestamps(raw_video_sync[timestamp_column],
                                                   timestamp_format,seconds=1)
    
    video_sync_w_suggestions = pd.concat((raw_video_sync, unusual_timestamps),
                                         axis=1)
    video_sync_path = os.path.split(video_sync_raw_path)
    video_sync_folder, video_sync_file = video_sync_path[:-1][0], video_sync_path[-1]
    
    actual_file_name = video_sync_file[:-4]
    suffix='w_suggestions.csv'
    final_path = os.path.join(video_sync_folder, actual_file_name+suffix)
    video_sync_w_suggestions.to_csv(final_path)
    #    
    #    raw_timestamps, occurence = np.unique(raw_video_sync[timestamp_column], 
    #                                           return_counts=True)
    #    # return read timestamps in ascending order of occurence
    #    ascending_occurence_index = np.argsort(occurence)
    #    occurence_ascending_order = occurence[ascending_occurence_index]
    #    timestamps_by_occurence_frequency = raw_timestamps[ascending_occurence_index]
    #    
    #    timestamp_and_occurence = pd.DataFrame(data={timestamp_column:timestamps_by_occurence_frequency,
    #                                                 'occurence':occurence_ascending_order})
    #    
    #    
    #    
    #    summary_timestamp_file = 'occurence_summary' + '-' + video_sync_file
    #    summary_timestamp_path = os.path.join(video_sync_folder,summary_timestamp_file)
    #    timestamp_and_occurence.to_csv(summary_timestamp_path)

    




    
    
    
    
    