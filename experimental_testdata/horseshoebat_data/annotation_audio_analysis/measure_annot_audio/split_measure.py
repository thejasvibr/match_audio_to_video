#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module that splits audio into segments of user-defined length and performs various measurements
on these segments. 

The output measuremets are then stored into a Pandas dataframe and can be saved into a csv file.


Created on Wed Jul  8 15:41:00 2020

@author: tbeleyur
"""
import os
import soundfile as sf
import numpy as np 
import pandas as pd 
from measure_annot_audio.inbuilt_measurement_functions import rms, peak_amplitude, lower_minusXdB_peakfrequency, dominant_frequencies

def split_measure_audio(audio_path, **kwargs):
    '''
    

    Parameters
    ----------
    audio_path : str/path
        Path to the single audio file that needs to be analysed.
    segment_size : float > 0, optional
        The whole audio will be split into segments of this length. 
        Defaults to 50ms.
    measurement_functions : list with functions, optional
        The measurements to be performed on each audio segment. 
        Defaults to the functions ehich measure the following:
            rms, peak amplitude, peak frequency, -10dB frequency,
            dominant frequencies
    channel_num : int, optional
        Which channel is to be analysed. The numbering starts from 0. 
        Defaults to the 0th (first) channel.

    Returns
    -------
    per_window_measurements : pd.Dataframe
        A dataframe with all the measurements in one Dataframe.
        This is a four-column Dataframe with the first column indicating
        the audio file name, the second column indicating the segment number
        , the third column indicating the measurement performed, and the fourth 
        column indicating the actual value obtained.
        
    Note
    ----
    1. If the audio duration cannot be split into a whole number of windows given the
    window_size and the length of the audio itself, then any remaining segment is
    fist checked if it is >= 90% of the user-defined window_size. If the remaining
    segment is >=90% of the defined segment_size, then it is treated as a valid window 
    and measurements are performed on it. Otherwise, it is discarded, and no measurements
    are performed on it. 
    
    2. All measurement functions need to have one compulsory input, which is the audio np.array
    and accept keyword arguments (even if they are not used in the function itself). 
    
    '''
    measurement_functions = kwargs.get('measurement_functions', 
                                                               default_measurement_functions)
    
    full_audio, fs = sf.read(audio_path)
    kwargs['fs'] = fs
    try:
        audio = full_audio[:,kwargs.get('channel_num',0)]
    except IndexError: 
        audio = full_audio
    audio_segments = split_audio(audio, fs, kwargs.get('segment_size', 0.050))
    segment_measurements = []
    for each in audio_segments:
        each_segment_measures = perform_measurements(each,
                                                     measurement_functions=measurement_functions, 
                                                     **kwargs
                                                     ) 
        segment_measurements.append(each_segment_measures)
    
    per_window_measurements = format_all_measures(segment_measurements)
    audio_file_name = os.path.split(audio_path)[-1]
    per_window_measurements['file_name'] = audio_file_name
    return per_window_measurements    

default_measurement_functions = [rms, peak_amplitude, 
                                 lower_minusXdB_peakfrequency,
                                 dominant_frequencies]

def split_audio(audio, samplerate, segment_size):
    '''
    

    Parameters
    ----------
    audio : np.array
        
    samplerate : float>0
        Sampling rate in Hz
    segment_size : float>0
        The intended size of each segment in seconds

    Returns
    -------
    audio_segments : list
        List with np.arrays in it. Each np.array is an audio segment from the
        input audio.

    Note
    ----
    If the audio duration cannot be split into a whole number of windows given the
    window_size and the length of the audio itself, then any remaining segment is
    fist checked if it is within >=90% of the user-defined window_size. If the remaining
    segment is >=90% of the defined segment_size, then it is treated as a valid window 
    and returned. Otherwise, it is discarded. 

    '''
    possible_num_segs = audio.size/(samplerate*segment_size)
    integer_num_segs = int(possible_num_segs) 
    if integer_num_segs == possible_num_segs:
        audio_segments = np.split(audio, integer_num_segs)
    else:
        last_seg_fraction = possible_num_segs - integer_num_segs
        main_audio_samples = int(samplerate*segment_size*integer_num_segs)            
        audio_segments = np.split(audio[:main_audio_samples],
                                                       integer_num_segs)
      
        if last_seg_fraction >= 0.9:
            # keep the last segment 
            audio_segments.append(audio[main_audio_samples:])
    return audio_segments

    
def perform_measurements(audio_segment, measurement_functions, **kwargs):
    '''
    

    Parameters
    ----------
    audio_segment : np.array
        Audio to be analysed
    measurement_functions : list with functions
        Each measurement function needs to have two inputs, the audio segment and 
        the sampling rate. It must output its measurements in a list.


    Returns
    -------
    one_segment_measurements : list
        List with all measurements for a single segment. 
        Each measurement is stored in its unique key.

    '''
    one_segment_measurements = []
    for each_function in measurement_functions:
        this_measure = each_function(audio_segment, **kwargs)
        one_segment_measurements.append(this_measure)
    return one_segment_measurements
        
def format_all_measures(all_segment_measures):
    '''
    

    Parameters
    ----------
    all_segment_measures : list with sublists
        Each sublist contains a series of dictionaries. Each dictionary corresponds
        to a measurement made on that particular audio segment. 

    Returns
    -------
    formatted_measurements : pd.DataFrame
        A long format DataFrame with one row per measurement value obtained from each 
        segment and each measurement. The DataFrame has three columns:
            1) segment_number
            2) measurement
            3) value

    '''
    all_measurements = []
    for segment_number, each_segment in enumerate(all_segment_measures):
        for each_measure in each_segment:
            
            measurement_values = extract_values(each_measure)
            num_values = len(measurement_values)

            one_row = pd.DataFrame(index=range(num_values))
            one_row['value'] = measurement_values
            one_row['segment_number'] = int(segment_number)
            measurement_name = list(each_measure.keys())[0]
            one_row['measurement'] = measurement_name
            all_measurements.append(one_row)
    formated_measurements = pd.concat(all_measurements)
    return formated_measurements 

def extract_values(measurement):
    '''
    Parameters
    ----------
    measurement : dictionary
        Dictionary with one measurement. 
        The entries may be in the form of 
        a single float, or a list with one or more values in it. 

    Returns
    -------
    measurement_values : list
        List with the values extracted from the measurement dictionary.
    '''
    values = list(measurement.values())[0]
    if np.logical_or(type(values) == list, type(values) == tuple):
        return values
    else:
        return [values]

if __name__ == '__main__':
    main_audio_path = '../../individual_call_analysis/annotation_audio'
    rec_hour = '2018-08-16_2300-2400_annotation_audio'
    rec_file = 'matching_annotaudio_Aditya_2018-08-16_2324_211.WAV'
    
    audio_path = os.path.join(main_audio_path, rec_hour, rec_file)

    z = split_measure_audio(audio_path, spectrum_smoothing_width=50, inter_peak_difference=500,
                            peak_range=20)

    
    
    
    
    
    
    

