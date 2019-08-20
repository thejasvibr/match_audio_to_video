#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Matches the video annotations to their corresponding audio segments

TODO : 
    1) Figure what a decent cross correlation is for a good fit - and set the 
       the threshold accordingly. 

Created on Thu Jul 25 17:59:23 2019

@author: tbeleyur
"""
import pdb
import glob
import os
import warnings
import numpy as np 
import pandas as pd
import scipy.signal as signal
import soundfile as sf
from tqdm import tqdm
#import matplotlib.pyplot as plt
#plt.rcParams['agg.path.chunksize'] = 10000

from av_sync import get_audio_sync_signal  as make_onoff_from_spikey


def match_video_sync_to_audio(video_sync, 
                                    audio_folder, **kwargs):
    '''Matches the video sync signal around an annotation to the audio sync 
    signal present in the given audio folder. 


    This file assumes that the video frame rate is lower than the 
    audio frame rate !!! 
    
    TODO :
        0) allow user to define a particular set of files in the audio_folder
           if not specified - then let it run through *ALL* files in the 
           folder
        1) what happens when the audio files are smaller than the 
           video sync ? 
        2) write case to handle contiguous_file = False 
        
         

    Parameters
    ----------
    video_sync :  pd.DataFrame.
                  The DataFrame has led intensities and assumes that these
                  are from a uniformly sampled video file. 

                  led_intensity : float.
                                  Representing led/bulb intensity over
                                  each frame

                  timestamp_verified: string.
                                     Timestamp on each frame.

                  annotation_block : Boolean.
                                     True indicates the frame belongs to
                                     the annotation block. False indicates 
                                     the frame belongs to the 
                                     overall sync block.

    

    audio_folder : string. 
                 Path to the folder containing the audio files to be matched.

    Keyword Arguments
    -----------------
    audio_fs : int. 
               Sampling rate of the audio files in Hz. 
               If not specified, then the sampling rate of the first file
               encountered with the given file formats found in the
               audio_folder is used. 

    audio_fileformat : str.
                      The file format of the audio files to be searched.   
                      eg. if the audio files are .flac files then 
                      the entry should be '*.flac'

                      Defaults to '*.WAV'
    
    audiosync_channel : int. 
                        Channel index starting from 0. This channel has
                        the audio recording of the common sync channel. 
                        Defaults to the last channel (-1)
    
    contiguous_files : Boolean. 
                       Whether all files in the audio folder correspond
                       to a common span of time - ie. if the end of one file
                       follows onto the beginning of the next file. 
                       IMPORTANT:
                       The file names *must* sort into an experimentally
                       sensible order. 
                       Defaults to True

    audio_sync_spikey : Boolean.
                        Whether the audio sync channel is in the form of
                        'up' and 'down' spikes. These spikes occur
                        because not all sound cards can record DC -type
                        signals well. In these cases the spikes occur
                        at the edges of start and end of the rectangular
                        ON/OFf waves - or capacitors can also be added 
                        to make the spikes more obvious. 
                        Defaults to True. 

    crosscorr_threshold : -1 <= float <= 1                      
                        An index that provides an idea of how well the 
                        audio and video are synced given the data available. 
                        When the cross correlation coefficient of the 
                        audio and video sync signals falls below the 
                        threshold a warning is issued. 
                        Defaults to 0.5. 

    Returns
    -------
    matched_audio : N_channels + 1 x Msamples np.array if succesfule. None if not.
                    The matched audio is an array with
                    the matched audio and the upsampled video sync signal 
                    attached along with it. 
                    When a prop

    
                    
    '''
    # load video sync signal and annotation block
    video_sync_signal, annotation_block, video_fps = get_videosync(video_sync)
    print('video_fps obtained is :', video_fps)
    print('videosync_size')
    # get audio sampling rate 
    audio_fs = kwargs.get('audio_fs', get_fs_from_audiofiles(audio_folder, 
                                                          **kwargs))

    resampling_factor = int(audio_fs/float(video_fps))
    if resampling_factor < 1:
        raise ValueError('Audio frame rate is lower than video frame rate --not implemented')

    # upsample video sync signal 
    upsampled_video_sync = upsample_signal(video_sync_signal, resampling_factor)

    upsampled_annotation_block = upsample_signal(annotation_block, resampling_factor,
                                                     annotation=True)
    # cross-correlate video sync signal with 
    best_match_to_syncblock  = get_best_audio_match(upsampled_video_sync, audio_folder, 
                                 **kwargs)
    if best_match_to_syncblock is not None:
        # extract the audio only relevant to the annotation
        samples_in_annotation = upsampled_annotation_block.astype('bool')
        annotation_audio = best_match_to_syncblock[samples_in_annotation, :]
        matched_audio = add_video_sync_channel(annotation_audio, 
                                             upsampled_video_sync,
                                             samples_in_annotation)
        
        syncblock_audio = np.column_stack((best_match_to_syncblock, 
                                           set_between_pm1(upsampled_video_sync)))
        
        audio_video_match = calculate_AV_match(syncblock_audio)
        
        av_match_threshold = kwargs.get('crosscorr_threshold', 0.5)
        if audio_video_match <= av_match_threshold:   
            warning_msg = 'The AV sync may not be very great - please check again. The value was :' + str(audio_video_match)
            warnings.warn(warning_msg, stacklevel=1)
        else:
            print('AV Sync was above threshold: ', audio_video_match)
        
        return(matched_audio, syncblock_audio, 
                       audio_video_match)
    else:
        warnings.warn('Proper macthing audio segment not found, moving to next')
        return(None, np.nan, np.nan)



def get_videosync(videosync_df):
    '''
    '''
    
    timestamps, frames = np.unique(videosync_df['timestamp_verified'],
                                   return_counts=True)
    # check for uniform frame rates 
    video_fps_values = np.unique(frames)
    if  len(video_fps_values)>1:
        raise ValueError('Frame rate varies in video sync - cannot proceed')
    else:
        video_fps = int(video_fps_values)
        
    video_sync_signal = videosync_df['led_intensity']
    annotation_block = np.int8(np.array(videosync_df['annotation_block']))
    return(video_sync_signal, annotation_block, video_fps)


def calculate_AV_match(matched_audio):
    '''
    Parameters
    ----------
    matched_audio : Nchannels x nsamples np.array where Nchannels >=2.
                    The last two channels are assumed to have the
                    audio sync[-2 index] and video sync channels[-1]
                    repectively. 
    
    Returns
    -------
    correlation_coefficient : -1 <=float<=1.
                            The correlation coefficient between the
                            audio and video signals.
    '''
    corr_coef = np.corrcoef(matched_audio[:,-1], matched_audio[:,-2])[0,1]
    return(corr_coef)
    



def get_fs_from_audiofiles(audio_folder,**kwargs):
    '''
    '''
    # find the first file matching the format
    matching_files = look_for_matching_audio_files(audio_folder, **kwargs)
    print('Did not find user-provided sample rate - getting it from first file that matches format!')
    try:
        audio, fs = sf.read(matching_files[0], frames=10)
        print('sampling rate is : ', fs)
    except:
        raise Exception('Could not read audio file for sampling rate to get sampling rate - please check format or folder path')
    return(fs)

def look_for_matching_audio_files(audio_folder, **kwargs):
    '''
    
    Keyword Arguments
    -----------------
    
    audio_fileformat : 
    
    file_subset : tuple.
                  Allows user to provide a custom range of files 
                  within which to look for the matching audio. 
                  See get_file_series
                    
    
    '''
    file_format = kwargs.get('audio_fileformat', '*.WAV')
    search_pattern = os.path.join(audio_folder,file_format)
    
    
    subset_range = kwargs.get('file_subset', False)
    if subset_range:
        all_files = glob.glob(search_pattern)
        matching_files = get_file_series(subset_range, all_files)
    
    else:
        matching_files = glob.glob(search_pattern)
        
    
    return(matching_files)


def get_file_series(fname_range, all_filenames):
    '''
    Parameters
    ---------
    fname_range : tuple with 2 strings.
                  Filename or parts of file name that 
                  allow sufficiently unique identification. 


    
    '''
    start, stop = fname_range
    sorted_files = sorted(all_filenames)
    
    start_index = get_matching_index(sorted_files, start)
    stop_index =  get_matching_index(sorted_files, stop)
    
    if np.logical_or(start_index.size>1, stop_index.size>1):
        raise CheckFileRangeID('Multiple matching entries found - please provide a more unique file name identifier')
    
    try:
        full_file_list = sorted_files[int(start_index):int(stop_index)+1]
    except:
        print(start_index, stop_index,'WAAAS')
        raise CheckFileRangeID('Unable to match given substrings to file names in folder - please check again!')
    
    return(full_file_list)


def get_matching_index(all_names, substring):
    '''
    '''
    present = map(lambda X: substring in X, all_names)
    if sum(present)>0:
        return(np.where(present)[0])
    else:
        return(np.array([]))



def upsample_signal(input_signal, resampling_factor, 
                    annotation=False):
    '''upsamples by repetition -- I guess its good mainly when 
    the video frame rate is *much* slower than the audio.

    Parameters
    ----------
    input_signal : np.array. 
                   array to be upsampled

    resampling_factor :int. 
                    The number of samples that each input_signal 
                    sample will be expanded into.
                    ie. if the resampling_factor is 100
                    each sample from the input_signal will
                    be repeated a 100 times. 

    annotation : Boolean. 
                Whether the input_signal is an annotation marker column.
                Defaults to False.


    Returns
    -------
    upsampled_signal : np.array. 

    '''
    if annotation:
        input_signal = np.array(1*input_signal)
    current_size = input_signal.size
    upsampled_2d = np.ones((current_size, resampling_factor))
    for row, sample_value in enumerate(input_signal):
        upsampled_2d[row,:] *= sample_value
    upsampled_signal = upsampled_2d.reshape(-1)
    return(upsampled_signal)


def get_best_audio_match(upsampled_video_sync, audio_folder, 
                                 **kwargs):
    '''Does a brute force search across the sync channel 
    of all files in the audio folder.
    

    
    Parameters
    ----------
    upsampled_video_sync : np.array. 

    audio_folder : str/path.
                   Path to the folder with all the audio files
                   

    Keyword Arguments
    -----------------
    
    Returns
    -------
    best_audio_fit : np.array
    
    '''
    matching_audio_files = sorted(look_for_matching_audio_files(audio_folder, 
                                                         **kwargs))
    
    contiguous= kwargs.get('contiguous_files', True)
    if contiguous:
        # find the matching audio 
        try:
            best_audio_match  = search_for_best_fit(upsampled_video_sync,
                                                            matching_audio_files,
                                                            **kwargs)
        except:
            print('Unable to get proper audio match for video segment!')
            best_audio_match = None
            
    else:
        raise NotYetImplemented('Non contiguous file cross correlation not yet implemented!!')
    
    return(best_audio_match)
                

def search_for_best_fit(upsampled_video_sync, audio_files_to_search,
                                                            **kwargs):
    '''
    Every audio file is first split into video_sync sized ish chunks.
    From the start of the audio file to its end - pairs of 
    chunks are joint into doublechunks. These audio doublechunks are then 
    cross-correlated with video sync and the max value is stored for each
    double chunk. 
    
    When there are multiple files the last chunk of the previous audio file
    is attached to the first chunk of the next to make the first doublechunk. 
    
    Parameters
    ------------
    upsampled_video_sync : 1x Msamples np.array. 
                           The upsampled video sync corresponding 
                           to the sync block of the audio stream. 

    Keyword Arguments
    -----------------
    audio_sync_spikey : Boolean 

    Returns
    ----------
    best_matching_audio : Nchannels x Msamples np.array. 
    '''
    ## get a broad area of the audio that matches
    best_doublechunk_id = find_best_matching_doublechunk(audio_files_to_search,
                                                         upsampled_video_sync,
                                                         **kwargs)

    print('Best doublechunk segment is:' + best_doublechunk_id)
    print('\n Starting refined search within best doublechunk segment...')
    # refine the search and now get the actual corresponding 
    best_matching_audio = do_refined_search(audio_files_to_search,
                                            upsampled_video_sync,
                                            best_doublechunk_id, **kwargs)
    print('Done with refined search - best matching audio returned')
    return(best_matching_audio)

def add_video_sync_channel(annotation_audio, 
                           whole_video_sync, 
                           annotation_marker):
    '''
    Parameters
    ----------
    annotation_audio : Nchannels x Msamples np.array
                       single/multichannel
                       audio segment corresponding strictly to the annotation
                       segment

    whole_video_sync : 1 x Xsamples np.array
                       Video sync signal

    annotation_marker : 1 x X samples x boolean np.array 
    
    Returns
    --------
    annotation_audio_w_videosync : N+1 channels x Msamples np.array
                                   Annotation_audio with a range-normalised
                                   video-sync signal 
    '''
    # check if the annotation marker makes sense 
    numsamples_tobechosen = np.sum(annotation_marker)
    if numsamples_tobechosen != annotation_audio.shape[0]:
        raise ValueError('Number of samples to be chosen from annotation marker do not match with annotation audio samples!')
    
    annotation_aligned_vsync = whole_video_sync[annotation_marker]
    # normalise the video sync to lie between +1 and -1 
    annotation_aligned_vsync -= np.mean(annotation_aligned_vsync)
    annotation_aligned_vsync /= np.max(np.abs(annotation_aligned_vsync))
    # just to prevent audio programs from showing red/clip indicators
    annotation_aligned_vsync *= 0.9 
    
    # join the video sync signal onto the annotation audio
    audio_w_vsync = np.column_stack((annotation_audio, annotation_aligned_vsync))
    return(audio_w_vsync)
    
    
    

def find_best_matching_doublechunk(audio_files_to_search, 
                                   upsampled_video_sync, **kwargs):
    '''
    
    Returns
    -------
    best_doublechunk_id : string.
                          The best_doublechunk_id is  
                          a composite of two file chunks from the same file 
                          or different files. 
                          Each chunk is described by its chunk_id with the
                          following pattern:
                              {filename with format}_{chunk number}
                              where chunk_number starts from 0.
                              eg. T0001.WAV_0 describes the first chunk 
                                 in the audio file T0001.WAV

                          A doublechunk is the combined chunk_ids of
                          two adjacent chunks:
                          eg1. 'T0002.WAV_2-T0002.WAV_3' means 
                          the audio obtained by concatenating
                          the second and third chunks of
                          T0002.WAV. 
                          
                          eg2. 'T0058.WAV_10-T0059.WAV_0'
                          describes the doublechunk obtained
                          by concatenating the 10th chunk 
                          of T0058.WAV and the 0th chunk of T0059.WAV
                           
    '''
    all_max_of_cc = []
    all_doublechunk_ids = []
    sync_channel  = kwargs.get('audiosync_channel',-1)
    for i, recording in enumerate(audio_files_to_search):
        file_name = os.path.split(recording)[-1]
        audio= read_syncaudio(recording, sync_channel)
        audio_chunks, chunk_ids = split_audio_to_chunks(file_name, 
                                                        audio,
                                                        upsampled_video_sync)
        if i>0:
            audio_chunks.insert(0,lastchunk)
            chunk_ids.insert(0,lastchunk_id)
        
        print('..transitioning to next file',chunk_ids[:2])
        maxcc, doublechunk_ids, lastchunk_id, lastchunk  = cross_correlate_chunks_contiguously(audio_chunks,
                                                                                            chunk_ids, 
                                                                                            upsampled_video_sync,
                                                                                            **kwargs)
        all_max_of_cc.append(maxcc)
        all_doublechunk_ids.append(doublechunk_ids)
    cc_maxes = np.concatenate(all_max_of_cc)
    best_doublechunk_index = np.argmax(cc_maxes)
    best_doublechunk_id = np.concatenate(all_doublechunk_ids)[best_doublechunk_index]
    
    return(best_doublechunk_id)


def do_refined_search(audio_files_to_search, upsampled_video_sync,
                                            best_doublechunk_id, **kwargs):
    
    '''
    Parameters
    ----------
    upsampled_video_sync : np.array
    
    audio_files_to_search : list
                            File paths to relevant audio files to load
    
    best_doublechunk_id : string.
                          Broad region of the audio file/s that match
                          the video sync. The two parts of the 
                          doublechunk are separated by a '-'.
                          see
                          find_best_matching_doublechunk documentation

    Keyword Arguments 
    ----------------
    audiosync_channel s
    
    Returns
    --------
    best_matching_audio : np.array
                            
    '''
    double_chunk_audio = parse_doublechunk_ids(best_doublechunk_id, 
                                              audio_files_to_search,
                                              upsampled_video_sync)

    nsamples, nchannels = double_chunk_audio.shape

    sync_channel  = kwargs.get('audiosync_channel',-1)
    spikey = kwargs.get('audio_sync_spikey', True)
    if spikey:
        onoff = make_onoff_from_spikey(double_chunk_audio[:,sync_channel], **kwargs)
        cc = signal.correlate(onoff, 
                          upsampled_video_sync,
                                 'same')
    else:
        norm_sync_chunk = set_between_pm1(double_chunk_audio[:,sync_channel])
        
        cc = signal.correlate(norm_sync_chunk, 
                              upsampled_video_sync,
                                 'same')
    ind = np.argmax(cc)
    numsamples, numchannels = double_chunk_audio.shape
    start =  int(ind - upsampled_video_sync.size/2.0)
    end = start + upsampled_video_sync.size
    check_for_proper_indices([start, end], double_chunk_audio)
    
    best_matching_audio = double_chunk_audio[start:end, :]
    
    return(best_matching_audio)

def parse_doublechunk_ids(doublechunk_ids, full_audio_filepaths, 
                          video_sync_ups):
    '''Takes in a doublechunk and returns an audio segment
    that is combination of both 
    
    Parameters
    ------------
    doublechunk_ids : string. 
                      two chunks with a '^' separating their names.
    
    
    '''
    chunk_names = doublechunk_ids.split('^')
    filenames = [parse_chunkid(name,'filename') for name in chunk_names]
    chunk_numbers = [parse_chunkid(name,'chunknumber') for name in chunk_names]         

    relevant_chunks = []
    for chunk_index, each_file in zip(chunk_numbers, filenames):
        full_file_path = get_fullfilepath(each_file, full_audio_filepaths)
        audio, fs = sf.read(full_file_path)
        audio_chunks, _ = split_audio_to_chunks(each_file, audio, 
                                                video_sync_ups, multichannel=True)
        relevant_chunks.append(audio_chunks[int(chunk_index)])
        del audio, audio_chunks
    doublechunk_audio = np.concatenate(relevant_chunks)
    return(doublechunk_audio)


def parse_chunkid(chunkid, req_output='filename'):
    '''
    Parameters
    ----------
    chunkid : str. 
              The ID for a particular sub-segment of a file following the 
              pattern :
              <FILENAME>.<EXTENSION>_<CHUNKNUMBER>
              for example 
              small_snippet_T-2019-08-02_15-12-30.WAV_0

    req_output : str. 
                req_output can either be:
                    'filename'
                    or 
                    'chunknumber'

    Returns
    --------
    req_output : str.
                Either a file name or the chunk number 
        
    '''
    filename, extension_and_chunknumber = chunkid.split('.')
    fileextension, chunknumber = extension_and_chunknumber.split('_')
    full_filename = filename + '.'+fileextension

    if req_output == 'filename':
        return(full_filename)
    elif req_output == 'chunknumber':
        return(chunknumber)

def get_fullfilepath(justfilename, full_filepaths):
    '''
    '''
    present = [justfilename in each_path for each_path in full_filepaths]
    if sum(present)<1:
        raise Exception('Could not find a matching file path for the given  \
                        file name!!')
    elif sum(present)>1:
        raise Exception('Found multiple file paths matching given file name!!')
    else:
        matching_filepath = full_filepaths[np.argmax(present)]
        return(matching_filepath)
        
    

def read_syncaudio(recording_path, sync_channel):
    '''
    '''
    audio, fs = sf.read(recording_path)
    return(audio[:,sync_channel])

def split_audio_to_chunks(recording_name, sync_audio, video_sync_signal, multichannel=False):
    '''
    
    Returns
    -------
    chunks : list with np.arrays
    chunk_ids: list with strings
    '''
    number_chunks = int(np.ceil(sync_audio.shape[0]/float(video_sync_signal.size)))
    if number_chunks <= 1 :
        raise NotYetImplemented('The video sync signal is longer than the audio file - this has not been implemented yet!')

    if not  multichannel:
        chunks = np.array_split(sync_audio, number_chunks)
    else:
        channels_chunked = []
        samples, num_channels = sync_audio.shape
        channels_chunked = [np.array_split(sync_audio[:,channel], number_chunks) for channel in range(num_channels)]
        #combine the channel wise chunks into     
        chunks = []
        for each_chunk in range(number_chunks):
            chunks.append(np.column_stack([channel[each_chunk] for channel in channels_chunked ]))
        
    chunk_ids = [recording_name+'_'+str(number) for number, chunklet in enumerate(chunks)]
    return(chunks, chunk_ids)
    
    



def extract_best_matching_snippet(file_name, startstop):
    '''
    '''
    audio, fs = sf.read(file_name, start=startstop[0], stop=startstop[1]+1)
    
    return(audio)

                
def cross_correlate_chunks_contiguously(audio_chunks_to_cc, chunk_ids, 
                                        upsampled_video_sync,
                                                       **kwargs):
    '''
    Parameters
    -----------
    audio_chunks_to_cc : 
        
    chunk_ids :
        
    upsampled_video_sync 
    
    Keyword Arguments
    ------------------
    
    
    
    '''
    maxcross_correlations = []
    doublechunk_ids = []
    spikey = kwargs.get('audio_sync_spikey', True)
    for one_chunk, next_chunk, one_id, next_id in tqdm(zip(audio_chunks_to_cc[:-1],
                                     audio_chunks_to_cc[1:],
                                     chunk_ids[:-1],
                                     chunk_ids[1:]), position=0, leave=True):

        doublechunk_id = '^'.join([one_id,next_id])
        if spikey:
            raw_audio = np.concatenate((one_chunk, next_chunk))
            audio_sync_chunk = make_onoff_from_spikey(raw_audio,
                                           **kwargs)
        else:
            audio_sync_chunk = np.concatenate((one_chunk, next_chunk))

        cc = signal.correlate(audio_sync_chunk, upsampled_video_sync, 'full')
        maxcross_correlations.append(np.max(cc))
        
        doublechunk_ids.append(doublechunk_id)
    return(maxcross_correlations, doublechunk_ids, next_id, next_chunk)


def set_between_pm1(X):
    '''Sets the values in X between +1 and -1
    '''
    highest, lowest, mean = np.max(X), np.min(X), np.mean(X)
    X_range = highest - lowest
    conversion = 2.0/X_range
    X_pm1 = (X-mean)*conversion
    return(X_pm1)

def check_for_proper_indices(inds, X):
    '''
    '''
    numsamples, _ = X.shape
    if np.any(np.array(inds)<0):
        raise IndexError('There are negative indices - please check if audio falls within relevant video sync')
    if np.any(np.array(inds)>= numsamples):
        raise IndexError('Calculated indices extend beyond the current audio chunk - please check if the video sync falls within audio')

class NotYetImplemented(ValueError):
    pass

class CheckFileRangeID(ValueError):
    pass

if __name__ == '__main__':
        
#    all_commonfps = glob.glob('example_data/common_fps*') # get all the relevant common_fps_sync files
#    audio_folder = 'example_data/audio/' # the current folder
#    kwargs= {}
#    kwargs['audio_fileformat'] = '*.wav'
#    kwargs['contiguous_files'] = True
#    kwargs['audio_sync_spikey'] = False
#    # generate the 
#    each_commonfps = all_commonfps[0]
#    video_sync = pd.read_csv(each_commonfps)
#    best_audio, cc = match_video_sync_to_audio(video_sync, audio_folder,
#                                           **kwargs)
#    sf.write('example_data/audio/matching_sync_'+str(5693)+'.WAV', best_audio, 
#             samplerate =kwargs.get('audio_fs',100000))
#   plotsamples = 5000000
#    plt.figure()
#    plt.plot(video[:plotsamples], label='sync block video')
#    plt.plot(audio[:plotsamples,-1], label='sync block audio')
#    plt.legend()
    matches = get_file_series(fname_range, all_files_numeric)