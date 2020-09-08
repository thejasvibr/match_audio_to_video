# also write all the cell contents into a separate python file to help with reproducibility
# across notebook 



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
    median_data = [ np.median(each) for each in group123_measures]
    
    return median_data

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

def just_return_input(X):
    return X

dB = lambda X : 20*np.log10(X)

def  reshape_to_3_cols(X):
    return np.array(X).reshape(-1,3)

def extract_single_multi_virtualmulti_by_measurement(df, measurement_name):
    '''
    '''
    by_measurements = df[df['measurement']==measurement_name].reset_index(drop=True)
    single, multi, virtual_multi = split_into_single_multi_and_virtual(by_measurements)
    return single, multi, virtual_multi
    
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



def shuffle_values_between_grouptypes(df1, df2, measurement_name):
    '''
    '''
    df1_measure, df2_measure = [each[each['measurement']==measurement_name] for each in [df1, df2]]
    df1_rows, df2_rows = df1.shape[0], df2.shape[0]
    # swap values between single and multi bat annotations
    all_values = np.concatenate([df1_measure['value'], df2_measure['value']]).flatten()
    np.random.shuffle(all_values)
    df1['value'] = all_values[:df1_rows]
    df2['value'] = all_values[df1_rows:]
    return df1, df2
    
def bootstrapped_observed_overlap(df, measurement, **kwargs):
    '''
    Returns
    -------
    single_multi, multi_virtualmulti, single_virtualmulti : float
        3 values which correspond to the chosen overlap metric between the three pairs of groups. 
    '''
    single_multi_virtual_multi = extract_single_multi_virtualmulti_by_measurement(df, measurement)
    one_segment_per_file = [ resample_one_segment_from_each_audio(each) for each in single_multi_virtual_multi]
    
    # overlap single-multi
    single_multi,_ = shuffle_overlap.calculate_overlap(one_segment_per_file[0]['value'],
                                                     one_segment_per_file[1]['value'], **kwargs)
    # overlap multi-virtualmulti
    multi_virtualmulti,_ = shuffle_overlap.calculate_overlap(one_segment_per_file[1]['value'], 
                                                           one_segment_per_file[2]['value'], **kwargs)
    # overlap single-virtualmulti
    single_virtualmulti,_ = shuffle_overlap.calculate_overlap(one_segment_per_file[0]['value'], 
                                                           one_segment_per_file[2]['value'], **kwargs)
    return single_multi, multi_virtualmulti, single_virtualmulti

def bootstrapped_shuffled_overlap(df, measurement, **kwargs):
    '''
    '''
    single_multi_virtual_multi = extract_single_multi_virtualmulti_by_measurement(df, measurement)
    one_segment_per_file = [ resample_one_segment_from_each_audio(each) for each in single_multi_virtual_multi]
    
    # shuffle single-multi
    shuffled_single, shuffled_multi = shuffle_values_between_grouptypes(single_multi_virtual_multi[0],
                                                                       single_multi_virtual_multi[1],
                                                                       measurement)
    # overlap single-multi
    shuf_single_multi,_ = shuffle_overlap.calculate_overlap(shuffled_single['value'],
                                                     shuffled_multi['value'], **kwargs)

    # -----------------------------------------------------------------------------------------
    # shuffle multi-virtualmulti
    shuffled_multi, shuffled_virtualmulti = shuffle_values_between_grouptypes(single_multi_virtual_multi[1],
                                                                       single_multi_virtual_multi[2],
                                                                       measurement)
    # overlap multi-virtualmulti
    shuf_multi_virtualmulti,_ = shuffle_overlap.calculate_overlap(shuffled_multi['value'], 
                                                             shuffled_virtualmulti['value'], **kwargs)
    # -----------------------------------------------------------------------------------------
    
    # shuffle single-virtualmulti
    shuffled_single, shuffled_virtualmulti = shuffle_values_between_grouptypes(single_multi_virtual_multi[0],
                                                                       single_multi_virtual_multi[2],
                                                                       measurement)
    
    # overlap single-virtualmulti
    shuf_single_virtualmulti,_ = shuffle_overlap.calculate_overlap(shuffled_single['value'], 
                                                           shuffled_virtualmulti['value'], **kwargs)
    return shuf_single_multi, shuf_multi_virtualmulti, shuf_single_virtualmulti
    
    

