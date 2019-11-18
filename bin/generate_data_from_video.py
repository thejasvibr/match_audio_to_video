# -*- coding: utf-8 -*-
"""Module that handles getting video sync signal data
Created on Wed Jul 24 09:41:06 2019

@author: tbeleyur
"""
from collections import Counter
import os
import cv2
import glob
import time
import pdb
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.filters import threshold_local, threshold_otsu, threshold_li
from tqdm import tqdm   
from tqdm import trange

from check_OCR_timestamps import make_posix_time

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
            try:
                kwargs['timestamp_border'] = parse_borders_in_annotation(annotation['timestamp_border'])
            except:
                print('Unable to parse timestamp border ')

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
    
    Keyword Arguments
    -----------------    
    timestamp_border = (550, 50, 70, 990) # left, up, right, bottom
    
    led_border = (867,1020,40,30)
    
    end_frame : optional. End frame for reading the timestamp + LED signal
    
    start_frame : optional. can get the timestamp reading to start from any arbitrary points
    
    '''
    print(video_path)
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
                  Defaults to np.sum if not given. 
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
        text = read_timestamp(im, **kwargs)
        
        measure_led_intensity = kwargs.get('measure_led', np.sum)
        cropped_led_ROI = ImageOps.crop(im,kwargs['led_border'])
        led_intensity = measure_led_intensity(cropped_led_ROI)
        return(text, led_intensity)

    except:
         print('Failed reading' + 'file:')
         return(np.nan, np.nan)

def read_timestamp(full_image, **kwargs):
    '''
    Parameters
    -----------
    full_image : 2D ImageOps image with the whole video frame
    
    Keyword Arguments
    ----------------
    timestamp_border : tuple with 4 entries
                       The borders follow ImageOps.crop syntax:
                           "The box is a 4-tuple defining the left, upper,
                           right, and lower pixel coordinate."

    custom_processing : function that accepts the cropped 2D image
                        and other possible keywords
                        Defaults to the 'default_ROI_processing'
                        function in this module.

    Returns
    -------
    text : str or np.nan
    
    '''
        
    if kwargs.get('read_timestamp', True):   
        timestamp_region = kwargs.get('timestamp_border')
        cropped_image = ImageOps.crop(full_image, timestamp_region)

        processing_function = kwargs.get('custom_processing', 
                                             do_nothing_processing)
        
        processed_image = processing_function(cropped_image,**kwargs)
        text = pytesseract.image_to_string(processed_image,
                                           config='digits')
    else:
        text = np.nan

    return(text)


def re_try_bad_reads(video_path, frames_to_be_verified,**kwargs):
    '''
    
    Parameters
    -----------
    video_path : str/os.path object pointing to the original video file

    
    frames_to_be_verified : array-like with integers.
                           Frame numbers of a video file
                           that need to be re-processed 
                           and fed through the OCR.
                           
                           NOTE: The first frame starts with 1 !!!! 
                           
    Keyword Arguments
    -----------------
    candidate_functions : list with functions. 
                          A set of one or more image processing functions
                          which will be applied onto the cropped image
                          having the ROI with timestamp. 

    timestamp_border : tuple with 4 entries.
                        The borders follow ImageOps.crop syntax:
                           "The box is a 4-tuple defining the left, upper,
                           right, and lower pixel coordinate."
    '''
    print('.......re-doing OCR for weirdly read frames....')
    # load video
    cap = cv2.VideoCapture(video_path)

    all_consensus_timestamps = pd.DataFrame(index=frames_to_be_verified,
                                            columns=['consensus_timestamp'])
    all_consensus_timestamps.index.name = 'frame_number'
    # load the frames that need verifying 
    for each_frame in tqdm(frames_to_be_verified):
        index = each_frame-1
        cap.set(1, index)
        success, frame = cap.read()
        full_image = Image.fromarray(frame)
        consensus_timestamp = get_consensus_timestamp(full_image, 
                                                      **kwargs)
        all_consensus_timestamps.loc[each_frame,'consensus_timestamp'] = consensus_timestamp
    

    return(all_consensus_timestamps)



def get_consensus_timestamp(full_image, **kwargs):
    '''If the timestamp is not parsable or un expected, try reading the 
    timestamp after applying different processing functions.
    

    
    The unparsable timestamps are discarded and not considered.
    
    Parameters
    ----------
    full_image : 2D ImageOps image with the whole video frame
    
    
    Keyword Arguments
    -----------------
    candidate_functions : list with functions. 
                          A set of one or more image processing functions
                          which will be applied onto the cropped image
                          of the ROI with timestamp. 


    timestamp_border : timestamp_border : tuple with 4 entries.
                        The borders follow ImageOps.crop syntax:
                           "The box is a 4-tuple defining the left, upper,
                           right, and lower pixel coordinate."
    datetime_format : str.
                    The format in which the timestamps are set in.

    Returns
    -------
    re_processed_timestamp : str or np.nan
                            A single prediction chosen from the
                            multiple predictions arising from
                            each candidate processing function. 
    
    '''
    candidate_functions = kwargs['candidate_functions']
    all_timestamp_predictions = list(range(len(candidate_functions)))

    for i, processing_function in enumerate(candidate_functions):
        kwargs['custom_processing'] = processing_function
        all_timestamp_predictions[i] = read_timestamp(full_image, **kwargs)

    #print('all obtained outputs',all_timestamp_predictions)
    re_processed_timestamp = get_majority_prediction(all_timestamp_predictions
                                                              ,**kwargs)
    return(re_processed_timestamp)
    


def get_majority_prediction(timestamp_predictions, **kwargs):
    '''
    Given a group of timestamp predictions, the ensemble answer refers
    to the most common prediction for each character position. 

        The final prediction is the most common result for each string position. 
    ie. if the format is %Y-%m-%d %H:%M:S
    and the outputs are 
    2018-09-11 10:05:05
    2018-09-11 10:03:05
    2018-09-11 11:03:03
    
    Then the final prediction will be :
    2018-09-11 10:03:05
    Parameters
    ----------
    timestamp_predictions : array-like with strings
                            
    Keyword Arguments
    -----------------
    datetime_format : str.
                      The format of the timestamp in the 
                      datetime package notation. 
                      eg. YYYY-mm-DD HH:MM:SS 
                      is '%Y-%m-%d %H:%M:%S'
    
    Returns
    -------
    majority_prediction_per_character : str. or np.nan
                                        
    Thanks to FogleBird @ https://stackoverflow.com/a/20038135/4955732
    for the Counter idea to get the majority value
    
    '''
    parsable_timestamps = []
    for each_prediction in timestamp_predictions:
        #print('timestamp and type', each_prediction, type(each_prediction))
        parsable = is_it_a_parsable_timestamp(each_prediction, **kwargs)
        if parsable:
            parsable_timestamps.append(each_prediction)
           
    #print('candidate parsable timestamps', parsable_timestamps,
    #      'len parsable_tiemstamp',len(parsable_timestamps))
    if len(parsable_timestamps) > 0:
        
        majority_character_predictions = []
        for character_position in range(len(parsable_timestamps[0])):
            character_predictions = []
            for prediction in parsable_timestamps:
                character_predictions.append(prediction[character_position])
            all_character_predictions = Counter(character_predictions)
            character, counts = all_character_predictions.most_common()[0]
            majority_character_predictions.append(character)
            
        majority_prediction_per_character = ''.join(majority_character_predictions)
    else:
        majority_prediction_per_character = np.nan

    return(majority_prediction_per_character)
                

def is_it_a_parsable_timestamp(text_output, **kwargs):
    '''
    '''
    try:
        posix_time = make_posix_time(text_output, 
                                  timestamp_pattern=kwargs['datetime_format']) 
    except:
        posix_time = np.nan

    if np.isnan(posix_time):
        return(False)
    else:
        return(True)

def default_ROI_processing(image, **kwargs):
    '''
    An older processing function that seemed to work 
    Parameters
    ----------
    image : ImageOps image

    
    Returns
    ---------
    output_im : ImageOps image post-processing
    '''
    cropped_img = image.resize((1600,200))
    P = np.array(cropped_img)
    P_mono = rgb2gray(P)
    
    block_size = 11
    P_bw = threshold_local(P_mono, block_size,
                                         method='mean')
    thresh = kwargs.get('bw_threshold', 0.65)
    P_bw[P_bw>=thresh] = 1
    P_bw[P_bw<thresh] = 0
    output_im = np.uint8(P_bw*255)
    output_imageops = Image.fromarray(output_im)
    return(output_imageops)


def blur_processing(image, **kwargs):
    '''
    '''
    
    
    return(image)

def simple_thresholding(image, **kwargs):
    '''
    The brightest part of the image is typically the 
    text itself. Use this to do a thresholding. 
    
    Parameters
    -----------
    image : Image object 
            cropped section with timestamp

    Keyword Arguments
    ------------------
    simple_threshold : 0<=float<=100
                       The percentile threshold between 0-100%ile of 
                       all pixel values. 

                       All pixels below the threshold are set to 0
                       and all those >= are set to 255. 
                       Defaults to the 90th percentile value. 

    Returns
    -------
    simple_thresholded_image : Image object.
                               same size as input 'image' variable
    '''
    grayscale = ImageOps.grayscale(image)
    image_as_array = np.array(grayscale)
    simple_threshold = kwargs.get('simple_threshold',90)
    brightest_pixel_values = np.percentile(image_as_array, simple_threshold)
     
    simple_thresholded_image = np.ones(image_as_array.shape)
    simple_thresholded_image[image_as_array<brightest_pixel_values] = 0
    simple_thresholded_image[image_as_array>=brightest_pixel_values] = 255
    simple_thresholded_image = np.int8(simple_thresholded_image)
     
    simple_thresholded_image = Image.fromarray(simple_thresholded_image)
    return(simple_thresholded_image)
 
def image_processing_sequence(original_image, 
                              functions, **kwargs):
    '''
    '''
    new_image = original_image.copy()
    for each_function in functions:
        new_image = each_function(new_image, **kwargs)
    
    return(new_image)
        
    
    

def resize(image, **kwargs):
    '''
    Incrase number of rows in image
    
    Parameters
    -----------
    image : Image object 
            cropped section with timestamp

    Keyword Arguments
    -----------------
    taller_factor : 0<float
                    The X times change from the current image height.
                    Defaults to 1.0 times. 

    wider_factor : 0<float
                    The X times change from the current image width.
                    Defaults to 1.0 times. 
    
    Returns
    -------
    taller_image : Image object
                   The taller version of the input 'image' 
    '''
    width, height = image.size
    
    height_magnification = kwargs.get('taller_factor',1.0)
    width_magnification = kwargs.get('wider_factor',1.0)
    target_height = int(np.around(height*height_magnification))
    target_width = int(np.around(width*width_magnification)) 
    
    taller_image = image.resize( (target_width, target_height))
    
    return(taller_image)

def resize_wider(image, **kwargs):
    '''
    '''
    return(image)

  

def do_nothing_processing(image, **kwargs):
    '''
    '''
    return(image)
    
    

def threshold_and_resize(image, **kwargs):
    '''
    '''
    thresholded_and_resized = image_processing_sequence(image,
                                                       [simple_thresholding,
                                                        resize],
                                                       **kwargs)
    return(thresholded_and_resized)

def adaptive_gaussian_thresholding(image,**kwargs):
    '''
    '''
    image_as_array = np.asarray(image)
    thresholded = cv2.adaptiveThreshold(image_as_array,255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY,15,2)
    return(thresholded)

def li_thresholding(image,**kwargs):
    '''
    '''
    image = ImageOps.grayscale(image)
    image_as_array = np.asarray(image)
    li_threshold = threshold_li(image_as_array)
    
    output_image = np.int8(np.zeros(image_as_array.shape))
    output_image[image_as_array>li_threshold] = 255
    
    output = Image.fromarray(output_image)    
    return(output)
#
#
if __name__ == '__main__':
#    
#    annotations_df = pd.read_csv('../example_data/eg_annotations.csv')
#    generate_videodata_from_videofiles(annotations_df, start_frame=2000,
#                                       end_frame=2100,
#                                       measure_led=np.sum)
    import matplotlib.pyplot as plt
    full_path = '/media/tbeleyur/THEJASVI_DATA_BACKUP_3/fieldwork_2018_002/horseshoe_bat/video/Horseshoe_bat_2018-08/2018-08-16/cam01/OrlovaChukaDome_01_20180816_21.50.31-23.00.00[R][@afc][0].avi'
    video = cv2.VideoCapture(full_path)
    frame_number = 549
    video.set(1,frame_number-1)
    success, frame = video.read()
    frame
    success
    ts_border = (590.4865253498716, 55.153357631379635,
                 112.86677643519215, 992.9494364196597)

    #ts_border2 =(587.0752513615416, 58.13242185916317, 117.66595953636647, 999.7975850161156)
    full_image = Image.fromarray(frame)
    timestamp = ImageOps.crop(full_image, ts_border)
    
    print('Basic')
    print(read_timestamp(full_image, timestamp_border=ts_border))
    print('Getting consensus picture')
    
    kwargs = {'candidate_functions':[do_nothing_processing, 
                                     simple_thresholding],
                              'timestamp_border':ts_border,
                              'datetime_format':'%Y-%m-%d %H:%M:%S',
                              'simple_threshold':90,
                              'wider_factor':1.5}   
    print(get_consensus_timestamp(full_image, **kwargs))
    
    #### testing the repeated readings of the bad ones : 
    wsuggestions = pd.read_csv('/home/tbeleyur/Documents/packages_dev/match_audio_to_video/experimental_testdata/horseshoebat_data/2018-08-16_2200-2400h/video_sync/videosync_OrlovaChukaDome_01_20180816_21.50.31-23.00.00[R][@afc][0].avi_w_suggestions.csv')
    odd = wsuggestions[wsuggestions['user_suggestion']=='ODDJUMP']
    odd_frame_numbers = odd['frame_number']

    frames_to_reread = odd_frame_numbers.to_list()[:100]
    P = re_try_bad_reads(full_path, frames_to_reread, **kwargs)
    wsuggestions.loc[frames_to_reread,'timestamp_reread1'] = P
    wsuggestions.loc[odd_frame_numbers.to_list(), 'timestamp_reread1'] = P
    wsuggestions.to_csv('cam_01_@afc[0]_w_re-reading.csv')
