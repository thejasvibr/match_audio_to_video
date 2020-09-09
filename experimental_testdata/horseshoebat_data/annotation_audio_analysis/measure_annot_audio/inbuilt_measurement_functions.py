#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inbuilt functions that perform various measurements on a given audio segment.

All measurement functions need to have one compulsory input, and accept keyword
arguments (they can also be unused).

The output of a measurement function is in the form of a dictionary with 
the key indicating the measurement output and the entry being in the form of 
a list with one or more values. 


@author: tbeleyur
"""
import numpy as np 
import scipy.signal as signal 

def dB(X):
    return 20*np.log10(abs(X))

def rms(audio, **kwargs):
    return {'rms': np.sqrt(np.mean(audio**2.0))}

def peak_amplitude(audio, **kwargs):
    return {'peak_amplitude':  np.max(np.abs(audio))}

def make_smoothened_spectrum(audio, **kwargs):
    '''
    Makes a smoothed power spectrum with power in dB

    Parameters
    ----------
    audio 
    fs 
    spectrum_smoothing_width


    Returns
    -------
    None.

    '''
    fs = kwargs['fs']
    spectrum_smoothing_width = kwargs['spectrum_smoothing_width']  
    
    power_spectrum_audio = 20*np.log10(np.abs(np.fft.rfft(audio)))
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    
    # make the smoothed spectrum
    smoothing_window = int(spectrum_smoothing_width/freq_resolution)
    smoothed_spectrum = np.convolve(power_spectrum_audio,
                                    np.ones(smoothing_window)/smoothing_window, 
                                    'same')
    return smoothed_spectrum
    

def lower_minusXdB_peakfrequency(audio, **kwargs):
    '''
    Returns the lowest frequency that is -X dB of the peak frequency.
    First the smoothened spectrum is generatd, and then the -X dB
    
    Parameters
    ----------
    audio : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    db_range : float, optional
        The X dB range. Defaults to 20 dB
    spectrum_smoothing_width : 

    Returns
    -------
    dictionary with key "minus_XdB_frequency" and entry with a list with one float
    '''
    fs = kwargs['fs']
    db_range = kwargs.get('db_range',20)
    #identify the peak frequency of the audio clip
    smooth_spectrum = make_smoothened_spectrum(audio, **kwargs)
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    
    peak_f = peak_f = freqs_audio[np.argmax(smooth_spectrum)]# peak_frequency(audio, fs=fs)

    below_threshold = smooth_spectrum <= np.max(smooth_spectrum)-db_range
    freqs_below_threshold = freqs_audio[below_threshold]
    # choose only those frequencies that are below the peak frequencies.
    freqs_below_peak = freqs_below_threshold[freqs_below_threshold<peak_f]
    if len(freqs_below_peak) <1:
        return {"minus_XdB_frequency": np.nan}
    else:
        minus_XdB_frequency = np.max(freqs_below_peak)
        return {"minus_XdB_frequency": minus_XdB_frequency}

def dominant_frequencies(audio, **kwargs):
    '''
    Identify multiple dominant frequencies in the audio. This works by considering
    all frequencies within -X dB of the peak frequency. 
    
    The 'dominant frequency' is identified as the central point of a region which 
    is continuously above the threshold. 
    
    The power spectrum is normally quite spikey, making it hard to discern individual 
    peaks with a fixed threshold. To overcome this, the entire spectrum is 
    mean averaged using a running mean filter that is 3 frequency bands long.

    
    Parameters
    ----------
    audio : np.array
    fs : float 
        sampling rate
    spectrum_smoothing_width : float
        The extent of spectral smoothing to perform in Hz. 
        This corresponds to an equivalent number of centre frequencies that will 
        be used to run a smoothing filter over the raw power spectrum. The
        number of center frequencies used to smooth in turn depends on the frequency resolution 
        of the power spectrum itself.         
        See Notes for more.
    inter_peak_difference : float 
        The minimum distance between one peak and the other in the smoothed power spectrum in
        Hz.
    peak_range : float
        The range in which the peaks lie in dB with reference to the maximum spectrum value. 

    Returns
    -------
    dictionary with key "dominant_frequencies" and a  List with dominant frequencies in Hz. 

    Notes
    -----
    The spectrum_smoothing_width is calculated so. If the spectrum frequency resolution is
     3 Hz, and the given spectrum_smoothing_width is set to 300 Hz, then the number
     of center frequencies used for the smoothing is 100. 
     
     The peak_range parameter is important in determining how wide the peak detection
     range is. If the peak_range parameter is very large, then there is a greater chance
     of picking up irrelevant peaks. 
     
    '''
    fs = kwargs['fs']
    inter_peak_difference = kwargs['inter_peak_difference']
    peak_range = kwargs['peak_range']

    
    smoothed_spectrum = make_smoothened_spectrum(audio, **kwargs)
    # get the dominant peaks 
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    freq_resolution = np.max(np.diff(freqs_audio))
    inter_peak_distance = int(inter_peak_difference/freq_resolution)
    peak_heights = (np.max(smoothed_spectrum)-peak_range, np.max(smoothed_spectrum))
    peaks, _ = signal.find_peaks(smoothed_spectrum,
                                 distance=inter_peak_distance, 
                                 height=peak_heights)
    dominant_frequencies = freqs_audio[peaks].tolist()
    return {"dominant_frequencies": dominant_frequencies}

def peak_frequency(audio, **kwargs):
    fs = kwargs['fs']
    spectrum_audio = np.abs(np.fft.rfft(audio))
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    peak_ind = np.argmax(spectrum_audio)
    peak_freq = freqs_audio[peak_ind]
    return peak_freq

if __name__=='__main__':
    # randomly sample 50ms audio segments from a randomly chosen file in a folder each time.
    import glob
    import os
    import soundfile as sf
    import matplotlib.pyplot as plt
    plt.rcParams['agg.path.chunksize'] = 10000

    main_audio_path = '../../individual_call_analysis/annotation_audio'
    rec_hour = '2018-08-16_2300-2400_annotation_audio'
    all_files = glob.glob(os.path.join(main_audio_path,rec_hour,'*.WAV'))
    rec_file = np.random.choice(all_files,1)[0]
    
    #audio_path = os.path.join(main_audio_path, rec_hour, rec_file)
    
    #multi_bat_path = os.path.join('/home/tbeleyur/Desktop','multi_batwav.wav')
    raw_audio, fs = sf.read(rec_file)
    b,a = signal.butter(4, 80000/fs*0.5, 'highpass')
    start_time = np.random.choice(np.arange(0,sf.info(rec_file).duration, 0.001)-0.05, 1)
    stop_time = start_time + 0.05
    start, stop = int(fs*start_time), int(fs*stop_time)
    audio = signal.filtfilt(b,a, raw_audio[start:stop,0])
    
    kwargs = {'inter_peak_difference':250, 
              'spectrum_smoothing_width': 100,
              'peak_range': 14,
              'fs':fs,
              'db_range':46}
    
    dom_freqs = dominant_frequencies(audio, **kwargs)

    power_spectrum_audio = 20*np.log10(np.abs(np.fft.rfft(audio)))
    smooth = make_smoothened_spectrum(audio, **kwargs)
    freqs_audio = np.fft.rfftfreq(audio.size, 1.0/fs)
    
    plt.figure()
    plt.plot(freqs_audio, power_spectrum_audio)
    plt.plot(freqs_audio, smooth)
    inds = [int(np.where(each==freqs_audio)[0]) for each in list(dom_freqs.values())[0]]
    plt.plot(list(dom_freqs.values())[0], power_spectrum_audio[inds],'*')
    
    
    lower = lower_minusXdB_peakfrequency(audio, **kwargs)
    plt.vlines(lower['minus_XdB_frequency'], 0, np.max(power_spectrum_audio))
    
    plt.figure()
    plt.specgram(audio, Fs=fs)
    plt.hlines(dom_freqs['dominant_frequencies'], 0,audio.size/fs)
    plt.hlines(lower['minus_XdB_frequency'], 0,audio.size/fs,'r')
    


