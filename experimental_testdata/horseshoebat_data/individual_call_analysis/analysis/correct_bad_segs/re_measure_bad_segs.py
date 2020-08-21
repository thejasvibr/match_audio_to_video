import pandas as pd
import numpy as np 
import soundfile as sf
import sys 
sys.path.append('/home/tbeleyur/Documents/packages_dev/correct_call_annotations')
import correct_call_annotations 
from correct_call_annotations import correct_call_annotations

def fuse_old_to_new_regions(call_regions, **kwargs):
    '''
    Allows the 'fusion' of multiple regions to a single new reassigned region
    
    Parameters
    ----------
    call_regions : pd.DataFrame
        Dataframe with the following compulsory columns 
        'audio_file', 'video_annot_id', 'start', 'stop'
    old_regions : list
        List with strings of region names
    reassigned_region : str
        String with name of newly reassigned region
    
    Returns
    -------
    corrected_call_regions : pd.DataFrame
        DataFrame similar to call_regions but with new region ids, and some cells with nans.

    Example
    -------
    # non-working example
    # call which has fm1, cf1, fm2, cf2 
    # the fm2 and cf2 regions need to be reassigned to fm2
    >>> correction_old_to_new_regions(call_regions, ['fm2','cf2'], 'fm2')
    
    '''
    old_regions, reassigned_region = kwargs['old_regions'], kwargs['reassigned_region']
   
    corrected_region = generate_corrected_fused_region(call_regions, old_regions, reassigned_region)
    valid_regions = call_regions.loc[~call_regions['region_id'].isin(np.array(old_regions))]
    corrected_call_regions = pd.concat([valid_regions, corrected_region.to_frame().transpose()], axis=0).reset_index(drop=True)
    return corrected_call_regions


def multi_fuse_old_to_new_regions(call_regions, **kwargs):
    '''
    
    Parameters
    ----------
    call_regions : pd.DataFrame
    
    Keyword Arguments
    -----------------
    old_regions_list : list with lists
    reassigned_regions_list : list with strings
    '''
    corrected = call_regions.copy()
    for old_regions, reassigned_region  in zip(kwargs['old_regions_list'], kwargs['reassigned_regions_list']):
        corrected = fuse_old_to_new_regions(corrected, old_regions=old_regions, reassigned_region=reassigned_region)
    return corrected




def generate_corrected_fused_region(call_regions, regions, reassigned_region):
    
    problem_regions = call_regions.loc[call_regions['region_id'].isin(np.array(regions))].reset_index(drop=True)
    new_start, new_stop = np.min(problem_regions['start']), np.max(problem_regions['stop'])
    
    # make a new row with all important columns intact and the rest with NAs
    corrected_region = problem_regions.loc[0,:]
    corrected_region['region_id'] = reassigned_region
    corrected_region['start'] = new_start
    corrected_region['stop'] = new_stop
    return corrected_region


def make_boolean_mask_for_call_regions(call_regions, audio_folder):
    '''
    Parameters
    ----------
    call_regions : pd.DataFrame
        for more check out See Also 
    audio_folder : str/path
        Folder with files in them. Can also be a folder with sub-folders.

    Returns 
    -------
    cf, fm :np.array 
        Boolean arrays with CF and FM regions indicated by samples that are True.
    
    See Also 
    --------
    create_correct_boolean_masks
    '''
    audio, fs = load_audio_from_call_region(call_regions, audio_folder)
    # initiate cf and fm
    cf = np.zeros(audio.size, dtype=np.bool)
    fm = np.zeros(audio.size, dtype=np.bool)
    
    regions = call_regions.groupby('region_id')
    is_cf = False; is_fm = False
    for regionid, region_df in regions:
        start_sample, end_sample = int(fs*region_df['start']), int(fs*region_df['stop'])
        if regionid[:2] == 'cf':
            cf[start_sample:end_sample] = True
        elif regionid[:2] == 'fm':
            fm[start_sample:end_sample] = True
    return cf, fm

def load_audio_from_call_region(call_regions, audio_folder):
    wav_file_name = call_regions['audio_file'][0]+'.WAV'
    wav_file = call_regions['audio_file'][0]+'.WAV'
    matched_audio  = correct_call_annotations.find_file_in_folder(wav_file_name,
                                                                  audio_folder)
    num_matches = len(matched_audio)
    if np.logical_or(num_matches>1, num_matches ==0):
        raise ValueError(f'Incorrect number of matches found: {num_matches}')
    
    # read audio
    audio, fs  = sf.read(matched_audio[0])
    return audio, fs

def create_correct_boolean_masks(call_regions, audio_folder, correction_to_be_done, **kwargs):
    '''
    Performs correction on a call region df, and then generates the correct Boolean CF and FM masks
    for the correctly segmented call. 
    
    Parameters
    ----------
    call_regions : pd.DataFrame 
        Dataframe with the following compulsory columns 
        'audio_file', 'video_annot_id', 'start', 'stop'
    correction_to_be_done : function
        The function that performs deletions or re-groupings of the call-regions in the 
        call. The function should accept the call_regions and other parameters need to be 
        keyword arguments. 
    audio_folder : str/path
        Folder with all audio files or further sub-folders.
    old_regions : list
        List with regions to be corrected
    reassigned_region : str
        String with name of newly reassigned region
        
    Returns 
    -------
    cf, fm : np.array 
        Boolean arrays with 
    corrected_call_regions : pd.DataFrame
        Version of call_regions with a >=1 rows deleted and renamed. 
        Aside from region_id, audio_file and video_annotation_id, cell entries may not be reliable!
    See Also
    --------
    correction_old_to_new_regions
    '''
    corrected_call_regions = correction_to_be_done(call_regions, **kwargs)
    cf, fm = make_boolean_mask_for_call_regions(corrected_call_regions, audio_folder)
    return cf, fm, corrected_call_regions

