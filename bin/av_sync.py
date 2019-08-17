# -*- coding: utf-8 -*-
"""Systematic attempts at getting the AV sync in place

Created on Wed Jun 26 13:50:37 2019

@author: tbeleyur
"""
import glob
import warnings
import multiprocessing
import time
#import matplotlib.pyplot as plt
#plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import peakutils as peak
import scipy.signal as signal 
import soundfile as sf

def get_audio_sync_signal(sync_channel, **kwargs):
    '''Creates a square type wave that takes on 3 values: -1,0,1.
    The pulse durations (ON/OFF) is calculated by the time gap between 
    adjacent peaks. If the pulse duration is < min_durn, then this pulse
    is set to 0. If the pulse duration is >= min_durn then it is set to 
    either +1/-1 depending on which type of pulse it is. 

    TODO:
        1) If consecutive +ve/+ve pulses or -ve/-ve pulses are encountered
        throw a warning and proceed by filling it with an OFF state ? 
       
        2) accept min threshold and min distance for peak detection through kwargs
          instead of the current get_peaks_parallelised

    Parameters
    ----------
    
    sync_channel : array-like. 1x Nsamples


    Keyword Arguments
    -----------------

    parallel : Boolean. 
               If True then the peak finding is done with pool.map 
               , else it is done serially with plain map syntax. 
    
    min_distance : see get_peaks
    min_level : see get_peaks

    Returns
    --------
    reassembled_signal : 1 x Nsamples np.array. 
                        The reassembled set of ON/OFF pulses with 
                        0 being the 'suppressed' pulses. 
                        Valid ON pulses have +1 value, 
                        and valid OFF pulses have -1 value. 

    *Note*
    ----
    When running the script in a visual editor like spyder it's best to 
    disable the parallel processing as everything gets stuck...
    '''
    sync = sync_channel.copy()
    # normalise the sync channel
    sync *= 1/np.max(sync)
    
    # identify positive and negative peaks 
    # isolate the peaks into positive and negative peaks 
    pos_sync = sync.copy()
    pos_sync[sync<0] = 0
    neg_sync = sync.copy()
    neg_sync[sync>0] = 0

    # get indices of positive and negative peaks 

    print('begin peak  processing')
    
    doit_parallely = kwargs.get('parallel', False)  
    map_inputs = [(pos_sync,kwargs), (neg_sync,kwargs)]

    if doit_parallely:
        pool = multiprocessing.Pool(2)
        pos_peaks, neg_peaks = pool.map(get_peaks, 
                                            map_inputs)
    else:
        pos_peaks, neg_peaks = map(get_peaks, 
                                            map_inputs)
        
    print('peak finding processing done')

    all_peaks = np.concatenate((neg_peaks, pos_peaks))
    
    all_peaks_argsort = np.argsort(all_peaks)
    sorted_peaks = all_peaks[all_peaks_argsort]
    multiply_by_minus1 = np.isin(sorted_peaks, neg_peaks)
    sorted_peaks[multiply_by_minus1] *= -1 # tag the off with a -1 

    # now go peak to peak and figure out if its on or off:
    first_peak_is_rising = sorted_peaks[0] > 0
    if first_peak_is_rising:
        first_segment = np.zeros(abs(sorted_peaks[0]))
    else:
        first_segment = np.ones(abs(sorted_peaks[0]))

    many_segments = []
    many_segments.append(first_segment)
    for one_peak, adjacent_peak in zip(sorted_peaks[:-1],sorted_peaks[1:]):
        pulse_is_on, length = get_pulse_state(one_peak, adjacent_peak)
        pulse_segment = make_onoff_segment(pulse_is_on, length)
        many_segments.append(pulse_segment)
        
    # check if the last peak:
    last_peak_is_rising = sorted_peaks[-1] > 0
    nsamples_lastsegment = int(sync.size - np.abs(sorted_peaks[-1]))
    if last_peak_is_rising:
        last_segment = np.ones(nsamples_lastsegment)
    else:
        last_segment = np.zeros(nsamples_lastsegment)
    many_segments.append(last_segment)
    reassembled_signal = np.concatenate(many_segments)
    
    return(reassembled_signal)

def get_pulse_state(index1, index2, **kwargs):
    '''index1 and index2 are two integers
    
    Parameters
    ---------
    index1, index2 : integers
                     index1 and index2 may be +ve or -ve depending on whether 
                     they are rising peaks (+ve) - or dropping peaks (-ve)
    Returns
    ------
    pulse_state: Boolean
    
    length : integer. 
             the length of the pulse state
    '''
    length = np.diff(np.abs([index1,index2]))
    
    peak_types = tuple([each_peak >0 for each_peak in [index1, index2]])
    pulse_state = pulse_state_dictionary.get(peak_types)
    if pulse_state is None:
        warnings.warn('The consecutive peaks of same type occured at samples:'+str(index1)+' ,'+str(index2))
        warnings.warn(sametype_spike1 +'\n '+sametype_spike2)
        return(False, length)        
    else:
        return(pulse_state, length)


pulse_state_dictionary = {(True, False): True, (False,True): False}
sametype_spike1 = 'There are two consecutive pulses that are rising or dropping!'
sametype_spike2 = 'The signal between this sametype spike region has been treated as an OFF'

def make_onoff_segment(pulse_state_ison, length):
    '''
    '''
    if pulse_state_ison:
        return(np.ones(length))
    else:
        return(np.zeros(length))


def get_peaks(X, **kwargs):
    ''' Gets the peaks by taking the |X| and doing peak finding.
    Parameters
    -----------
        X  : tuple with 2 entries
             entry 1 is the input signal, which is a 1D array-like object
             entry 2 is the keyword argument dictionary 

    Keyword Arguments
    ------------------
    min_distance : integer
                   Number of  samples between two consecutive peaks
                   Defauls to #samples equivalent to 140ms gap @
                   250 kHz (0.14 * 250000 samples)
                   See peakutils.peak.indexes

    min_level : 1>= float >0
                Minimum level to be reached for a sample to be considered a peak. 
                See peakutils.peak.indexes. 
                Defaults to 0.5

    Returns
    -------
    det_peaks : array-like
                Samples at which peaks were detected. 
    '''
    input_signal, kwargs = X
    min_distance = kwargs.get('min_distance',int(250000*0.07*2))
    min_level = kwargs.get('min_level', 0.5)
    det_peaks = peak.indexes(np.abs(input_signal), 
                             min_level, min_distance)
    return(det_peaks)


def multiplyby(peaktype):
    if peaktype > 0 :
        return(1)
    else:
        return(0)


def make_time_axis(X,fs=250000):
    t = np.linspace(0, X.size/float(fs), X.size)
    return(t)



if __name__ == '__main__':
    #    durns = np.arange(0.2, 2.0, 0.08)
    #    ons=[];offs=[];
    #    for  i in range(50):
    #        ons.append(np.random.choice(durns,1))
    #        offs.append(np.random.choice(durns,1))
    #    
    #    actual_LED_onoff = []
    #    fs = 250000
    #    for on,off in zip(ons,offs):
    #        actual_LED_onoff.append(np.ones(int(on*fs)))
    #        actual_LED_onoff.append(np.zeros(int(off*fs)))
    #    
    #    full_onoff = np.concatenate(actual_LED_onoff)
    #    just_peaks = np.diff(full_onoff)
    #    reconstr = get_audio_sync_signal(just_peaks, parallel=True)
    audiofs = 100000
    fname = 'T2019-07-04_15-26-53_0000001.wav'
    audio, fs = sf.read('example_data/audio/'+fname)
    sync = audio[:,-1]
    reconstr_audio = get_audio_sync_signal(sync, parallel=True,
                                           min_distance=int(audiofs*0.07*2))
    final_audio = np.column_stack((audio,reconstr_audio))
    sf.write('example_data/audio/non_spikey_' + fname,final_audio,
             samplerate=fs)
   #np.save('1sec_Reconstr_audio',reconstr_audio[:250000])
    
    
  
