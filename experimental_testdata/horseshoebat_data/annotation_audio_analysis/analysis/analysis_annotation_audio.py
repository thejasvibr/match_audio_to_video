#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to hold all the code related to overlap, shuffling and single window 
selection for the annotation audio analysis. 

"""
import numpy as np 
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import tqdm

def split_into_single_multi_and_virtual(df):
    '''
    Splits all split-measurements into single, multi and virtual multi bat 
    DataFrames.
    '''
    single_bat = df[df['num_bats']==1].reset_index(drop=True)
    
    real_multi_bat = np.logical_and(df['num_bats']>1, df['type']=='observed')
    multi_bat  = df[real_multi_bat].reset_index(drop=True)
    virtual_multi_bat = np.logical_and(df['num_bats']>1, df['type']=='virtual')
    virtual_multibat = df[virtual_multi_bat]
    return single_bat, multi_bat, virtual_multibat

def extract_one_measurement(df,measurement_name):
    '''
    '''
    one_measurement = df[df['measurement']==measurement_name].reset_index(drop=True)
    return one_measurement


def calc_group_type_summary(df, measurement_name, summary_fn, proc_fun:lambda X:X):
    '''
    Splits the df into three groups based on the column 'group_type'. 
    If there are more than two group types, then an error is raised
    
    '''
    
    df_bygrouptype = split_into_grouptypes(df)
    if len(df_bygrouptype)!=3:
        raise ValueError('There must be 3 group types for this analysis')
    df_grouptypes = [each[each['measurement']==measurement_name]  for each in df_bygrouptype]
    
    summary_grouptypes = [summary_fn(proc_fun(each_df['value'])) for each_df in df_grouptypes] 
    summary1, summary2, summary3 = summary_grouptypes
    return summary1, summary2, summary3

def split_into_grouptypes(df):
    '''
    splits the dataframes by group type
    '''
    group_types = np.unique(df['group_type'])
    return [df[df['group_type']==each] for each in group_types]
    

def extract_single_multi_virtualmulti_by_measurement(df, measurement_name):
    '''
    '''
    by_measurements = df[df['measurement']==measurement_name].reset_index(drop=True)
    single, multi, virtual_multi = split_into_single_multi_and_virtual(by_measurements)
    return single, multi, virtual_multi

def make_inspection_and_comparison_plot(df, measurement_name, process_fn=lambda X: X):
    '''
    '''
    single, multi, virtual_multi = extract_single_multi_virtualmulti_by_measurement(df, measurement_name)
    plt.figure(figsize=(8,4))
    plt.violinplot([process_fn(single['value'].to_numpy()),
                    process_fn(multi['value'].to_numpy()),
                    process_fn(virtual_multi['value'].to_numpy())],
                   [0, 1, 2], showmedians=True,
                  quantiles=[[0.25,0.75],[0.25,0.75],[0.25,0.75],]);    
    plt.xticks([0,1,2],['single','multi', 'virtual multi'])
    plt.ylabel(measurement_name, fontsize=12)
    
    
def resample_one_segment_from_each_audio(df):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the following compulsory columns
        file_name, segment_number
    Returns 
    -------
    one_seg_per_file : pd.DataFrame
        A subset of all rows, but with only one segment per file chosen. 
    '''
    by_filename = df.groupby(['file_name'])
    #resampled_data = [choose_one_segment(each) for each in by_filename]
    resampled_data = by_filename.apply(choose_one_segment)
    one_seg_per_file = resampled_data.reset_index(drop=True)
    return one_seg_per_file

def choose_one_segment(sub_df):
    #filename, sub_df = filename_and_subdf
    # if there are >1 valid segments in the audio file, then select just one randomly
    segments = sub_df['segment_number'].tolist()
    if len(segments) == 1:
        return sub_df
    else:
        one_segment = int(np.random.choice(segments, 1))
        chosen_rows = sub_df['segment_number']==one_segment
        return sub_df[chosen_rows]
    

def shuffle_values_between_grouptypes(df1, df2, measurement_name):
    '''
    '''
    df1_msmt, df2_msmt = [each[each['measurement']==measurement_name] for each in [df1, df2]]
    df1_rows= df1.shape[0]
    # swap values between single and multi bat annotations
    all_values = np.concatenate([df1_msmt['value'], df2_msmt['value']]).flatten()
    np.random.shuffle(all_values)
    df1['value'] = all_values[:df1_rows]
    df2['value'] = all_values[df1_rows:]
    return df1, df2
    
def just_return_input(X):
    return X

def calculate_observed_and_shuffled_overlap(df, measurement_name, summary_fn, proc_fn, num_shuffles=500):
    '''
    '''
    observed_shuffled_BC = Parallel(n_jobs=4)(delayed(get_observed_and_shuffled)(df, measurement_name, summary_fn, proc_fn) for i in tqdm.tqdm(range(num_shuffles)))
    observed = [each[0] for each in observed_shuffled_BC]
    shuffled =  [each[1] for each in observed_shuffled_BC]

    return observed, shuffled
        

def get_observed_and_shuffled(df, measurement_name, summary_fn, proc_fn):
    one_segment = resample_one_segment_from_each_audio(df)
    group_type1_measures, group_type2_measures = calc_group_type_summary(one_segment, 
                                                                          measurement_name, 
                                                                          summary_fn, proc_fn)
    this_run_BC, _ = shuffle_overlap.calculate_overlap(group_type1_measures, group_type2_measures)

    # now do the data shuffling
    shuffled_segment = make_shuffled_df(one_segment, measurement_name)
    grouptype1_shufmeasures, grouptype2_shufmeasures = calc_group_type_summary(shuffled_segment, 
                                                                          measurement_name, 
                                                                          summary_fn, proc_fn)
    this_run_shufBC, _ = shuffle_overlap.calculate_overlap(grouptype1_shufmeasures, grouptype2_shufmeasures)
    return this_run_BC, this_run_shufBC


def calculate_bootstrapped_median_across_grouptypes(df, measurement_name, summary_fn,proc_fn):
    '''
    A DataFrame with 2 group types is taken, bootstrapped and the delta median 
    between the two group types is repeatedly calculated
    
    
    Returns
    -------
    median_data : 
    '''
    one_segment = resample_one_segment_from_each_audio(df)
    group123_measures = calc_group_type_summary(one_segment, 
                                                          measurement_name, 
                                                      summary_fn, proc_fn)
    median_data = list(map(np.median, group123_measures))
    
    return median_data

def bootstrapped_median_distribution(df, measurement_name, summary_fn,proc_fn, Nruns=500, parallel=False):
    '''
    '''
    if parallel:
        all_median_data = Parallel(n_jobs=4)(delayed(calculate_bootstrapped_median_across_grouptypes)(df, 
                                                                                                  measurement_name, summary_fn,proc_fn,) for i in tqdm.trange(Nruns))
    else:
        all_median_data = [calculate_bootstrapped_median_across_grouptypes(df,   measurement_name, summary_fn,proc_fn,) for i in tqdm.trange(Nruns)]

    num_groups  = len(all_median_data[0])
    reshaped_median_data = np.concatenate(all_median_data).reshape(-1,num_groups)
    return reshaped_median_data
