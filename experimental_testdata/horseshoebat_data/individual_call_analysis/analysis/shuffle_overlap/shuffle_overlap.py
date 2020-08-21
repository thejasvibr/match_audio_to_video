'''Module which generates kernel density estimates of distributions, 
calculates their overlap and also performs the shuffling of data
across the distributions. 

This module implements  shuffling + overlap calculation to assess
if two distributions are relatively similar or different from each other. 
'''
import numpy as np 
from scipy import stats
import joblib
from joblib import Parallel, delayed

def generate_kde(data):
    '''
    '''
    return stats.gaussian_kde(data)

def get_global_minmax(dist1, dist2):
    '''
    Parameters
    ----------
    dist1, dist2 : np.array 
        M points x N columns arrays with the raw data. 

    Returns
    -------
    min_vals, max_vals : np.array
        1 x N_columns np.arrays which hold the global minimum/maximum value for that column
    '''
    if np.ndim(dist1)>1:
        joint_dataset = np.row_stack((dist1, dist2))
        min_vals = np.min(joint_dataset, 0)
        max_vals = np.max(joint_dataset, 0)
    else:
        joint_dataset = np.concatenate((dist1, dist2))
        min_vals = np.min(joint_dataset)
        max_vals = np.max(joint_dataset)
        
    return min_vals, max_vals
    
def remove_nans(X):
    if np.sum(np.isnan(X)) >0:
        nan_indices = np.isnan(X)
        return X[~nan_indices]
    else:
        return X


def shuffle_datasets(distbn_1, distbn_2):
    '''
    joins datasets, shuffles them and then splits them into 
    their original sizes. 
    The output datasets have the same dimensions as the input, but 
    the entries are shuffled. 
    '''
    size_1, size_2 = [each.size for each in [distbn_1, distbn_2]]
    
    joint_shuffled = np.concatenate((distbn_1, distbn_2))
    np.random.shuffle(joint_shuffled)
    shuffled_1, shuffled_2 = joint_shuffled[:size_1], joint_shuffled[size_1:] 
    return shuffled_1, shuffled_2


        
    
    
def calculate_overlap(main_distbn, other_distbn, **kwargs):
    '''Calculates the overlap between two distributions through the following steps:
    1. Remove NaNs from data
    2. Generate Kernel Density Estimates(KDEs)
    3. Calculate probability distribution function (PDF) for both KDEs across the 
       global minimum and maximum value of both input datasets
    4. Calculate overlap by multiplying the two PDFs and summing up the 'joint PDF'
       This summed up value is called the 'overlap'

    Parameters
    ----------
    main_distbn, other_distbn : np.array 
        Each array is an 1 x N_points np.array 
    num_points: int>0, optional    
        Number of points to evaluate the PDF on between the global minimum
    overlap_method : str, optional
        The method used to calculate the distribution overlap. Defaults to 
        custom, where the two PDFs are multiplied and summed up. 
        The other method is 'integral'. The 'integral' method may/may not 
        have more power in detecting differences between distributions - use 
        after testing for a simulated dataset.
    
    Returns 
    -------
    overlap : float >0 
        The overlap between the kernel density estimates of the 
        two datasets
    (minmax_range, main_pdf, other_pdf) : tuple
        Tuple with intermediate data.

    Note
    ----
    The final 'overlap' value only makes sense when compared with a sensible and similar value. 
    The obtained overlap value generated cannot be interpreted directly because it's not normalised in 
    any way. 
    
    The sensible way to understand the obtained overlap is to compare it with the overlap values 
    where the data is shuffled between the two input distributions.

    '''
    main_distbn = remove_nans(main_distbn)
    other_distbn = remove_nans(other_distbn)
    main_kde = generate_kde(main_distbn,)
    other_kde = generate_kde(other_distbn,)
    
    # calculate overlap between the two KDEs by integrating between the global min and max values
    global_min, global_max = get_global_minmax(main_distbn, other_distbn)
    
    # calculate the pdf over the global min-max
    minmax_range = np.linspace(global_min, global_max, kwargs.get('num_points', 1000))
    main_pdf = main_kde.evaluate(minmax_range)
    other_pdf = other_kde.evaluate(minmax_range)
    
    # overlap 
    overlap_method = kwargs.get('overlap_method', 'custom')
    if overlap_method == 'custom':
        overlap = np.sum(np.sqrt(main_pdf*other_pdf))
    elif overlap_method == 'integral':
        overlap = main_kde.integrate_kde(other_kde)
    else:
        raise ValueError(f'Invalid overlap method: {overlap_method} given!')
        
   
    return overlap, (minmax_range, main_pdf, other_pdf) 

def shuffle_and_calc_overlap(main_distbn, other_distbn, **kwargs):
    shuffled_1, shuffled_2 = shuffle_datasets(main_distbn, other_distbn)
    this_shuffled_overlap,_ = calculate_overlap(shuffled_1, shuffled_2, **kwargs)
    return this_shuffled_overlap

def overlap_w_shuffling(main_distbn, other_distbn, **kwargs):
    '''
    1. Calculates overlap between the distributions of the two input 
    distributions. 
    2. Performs data shuffling betweent he two distbns, and calculates overlap
    
    Parameters
    ----------
    main_distbn, other_distbn : array-like
        The measurement/parameter values
    num_points : int >0, optional 
        Number of points to evaluate the PDF on between the global minimum 
        and maximum. Defaults to 1000
    num_shuffles : int>0, optional 
        Number of shuffling rounds to be performed. Defaults to 2500
    
    Returns 
    -------
    original_overlap : float
        Overlap value between the PDFs of the two input distributions 
    overlap_results : tuple
        Intermediate data from calculate_overlap of the original datasets.
    shuffled_overlap : array-like 
        List with all overlap values from shuffled data

    Note
    ----
    *This is not the full story - still needs to be clarified....*
    The overlap calculated is the overlap of *non-normalised* counts/units that the 
    input distribution is actually given in. The scipy.gaussian_kde.evaluate function 
    actually returns the counts if counts are given, and PDF if pdf is given. 
    
    See Also
    --------
    calculate_overlap
    '''
    original_overlap, overlap_results = calculate_overlap(main_distbn, other_distbn, **kwargs)
    
    num_shuffles = kwargs.get('num_shuffles', 2500)
    

    shuffled_overlap = Parallel(n_jobs=4)(delayed(shuffle_and_calc_overlap)(main_distbn, other_distbn, **kwargs) for i in range(num_shuffles))


    return original_overlap, overlap_results, shuffled_overlap
        
def percentile_score_of_value(X, distribution):
    return stats.percentileofscore(distribution, X)
