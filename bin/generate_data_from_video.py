# -*- coding: utf-8 -*-
"""Module that handles getting video sync signal data
Created on Wed Jul 24 09:41:06 2019

@author: tbeleyur
"""
import os
import cv2
import glob
import time
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.filters import threshold_local
from tqdm import tqdm
from tqdm import trange

def generate_videodata_from_videofiles(annotations_df, **kwargs):
    '''
    
    Parameters
    ----------
    annotations_df : pandas DataFrame with at elast the following columns
                    'video_path'
    '''
    
    #generate the timestamps and led signals for each video file that 
    # has annotations

    processed_videos = []
    for i, annotation in tqdm(annotations_df.iterrows()):
        print('Now generating dat from row', i)
        video_name = os.path.split(annotation['video_path'])[-1]
        if video_name in processed_videos:
            print(video_name, ' is already processed moving to next video')
            pass
        else:
            kwargs['video_name'] = video_name
            kwargs['led_border'] = parse_borders_in_annotation(annotation['led_border'])
            kwargs['timestamp_border'] = parse_borders_in_annotation(annotation['timestamp_border'])
            print('gettin raw video data from '+video_name+'  now....')
            get_syncdata_for_a_videofile(annotation['video_path'], **kwargs)
            print('doen w getting raw video data ')
            processed_videos.append(video_name)
    print('All of the videos have been processed...')

def get_syncdata_for_a_videofile(video_path,**kwargs):
    '''
    Parameters
    ----------
    video_annotation : pandas DataFrame row with at least the following 
                       columns. 
                       video_path : full file path to the video file

    Returns
    --------
    None 
    
    A side effect of this function is the csv which follows the naming
    convention:
        'videosync_{video_name_here}_.csv'
                 
    '''
    
    timestamps, intensity = get_data_from_video(video_path, 
                                                           **kwargs)

    df = pd.DataFrame(data=[], index=range(1,len(timestamps)+1), 
                      columns=['frame_number','led_intensity',
                               'timestamp','timestamp_verified'])
    print(df.shape, len(intensity))
    df['led_intensity'] = intensity
    df['timestamp'] = timestamps
    df['frame_number'] = range(1,len(timestamps)+1)
    df.to_csv('videosync_'+kwargs['video_name']+'_.csv',
              encoding='utf-8')

def parse_borders_in_annotation(border):
    '''
    '''
    try:
        text_between_brackets = border[1:-1]
        text_as_entries = text_between_brackets.split(',')
        text_as_numbers  = tuple([ float(each) for each in text_as_entries])
        return(text_as_numbers)
    except:
        raise Exception('Unable to parse borders - please check their format!')

def get_data_from_video(video_path, **kwargs):
    '''
    
    Keyword Argument
    
    timestamp_border = (550, 50, 70, 990) # left, up, right, bottom
    led_border = (867,1020,40,30)
    end_frame : optional. End frame for reading the timestamp + LED signal
    start_frame : optional. can get the timestamp reading to start from any arbitrary points
    '''
    video = cv2.VideoCapture(video_path)

    print('starting frame reading')
    timestamps = []
    led_intensities = []
    
    
    start_frame = kwargs.get('start_frame',0) 
    end_frame = kwargs.get('end_frame', int(video.get(cv2.CAP_PROP_FRAME_COUNT)))    
    
    # check if led_border and timestamp border are given - else throw error message!
    check_if_borders_are_given(**kwargs)

    video.set(1, start_frame)

    for i in  trange(start_frame, end_frame, desc='Frames read', 
                     position=0, leave=True):
        successful, frame = video.read()
        #        if np.remainder(i,50)==0:
        #            print('reading '+str(i)+'th frame')
        if not successful:
            frame = np.zeros((1080,944,3))
            print('Couldnt read frame number' + str(i))

        timestamp, intensity = get_lamp_and_timestamp(frame ,**kwargs)
        timestamps.append(timestamp)
        try:
            led_intensities.append(float(intensity))
        except ValueError:
            print('Unable to read LED intensity at :', i)

    print('Done with frame conversion')
    return(timestamps, led_intensities)

def  check_if_borders_are_given(**kwargs):
    '''
    '''
    if kwargs.get('led_border') is None:
        raise ValueError('The borders for the blinking light have not been defined...')
    
    if kwargs.get('timestamp_border') is None:
        print('No timestamp border detected..are you sure you want to proceed?')



def get_lamp_and_timestamp(each_img, **kwargs):
    '''
    
    Keyword Arguments
    ------------------
    timestamp_border, led_border : tuple with 4 entries
                       Defines the border area where the timestamp/led data
                       can be extracted from.

                       The number of pixels to crop in the following order:
                       to the left of, above, to the right of and below. 

    read_timestamp : Boolean.
                      Whether timestamps need to be read or not. 
                      Defaults to True. 

    measure_led : function
                  A custom function to measure the led intensity of
                  the cropped patch. 
                  Defaults to np.max if not given. 
                  eg. if there are saturated patches and very dark patches in 
                  and the led intensity tends to be somewhere in between 
                  tracking the median value with np.median
                  could show when the led goes on and 
                  off. 

    bw_threshold : 1 > float >0
                  Sets the threshold for binarisation after a color image is turned to 
                  grayscale. Defaults to 0.65.
    '''
    try:
        im = Image.fromarray(each_img)
        
        if kwargs.get('read_timestamp', True):
        
            timestamp_region = kwargs.get('timestamp_border')
            cropped_img = ImageOps.crop(im, timestamp_region).resize((1600,200))
            P = np.array(cropped_img)
            P_mono = rgb2gray(P)
            
            block_size = 11
            P_bw = threshold_local(P_mono, block_size,
                                                 method='mean')
            thresh = kwargs.get('bw_threshold', 0.65)
            P_bw[P_bw>=thresh] = 1
            P_bw[P_bw<thresh] = 0
            input_im = np.uint8(P_bw*255)
            
            text = pytesseract.image_to_string(Image.fromarray(input_im),
                                               config='digits')
        else:
            text = np.nan
        # calculate LED buld intensity:
        measure_led_intensity = kwargs.get('measure_led', np.max)
        led_intensity = measure_led_intensity(ImageOps.crop(im,kwargs['led_border']))
        return(text, led_intensity)
    except:
         print('Failed reading' + 'file:')
         return(np.nan, np.nan)

if __name__ == '__main__':
    
    annotations_df = pd.read_csv('example_data/eg_annotations.csv')
    # Since it's a small video with only 1370 frames - we'll run the whole thing! This could take a couple of minutes
    generate_videodata_from_videofiles(annotations_df, end_frame=50)


