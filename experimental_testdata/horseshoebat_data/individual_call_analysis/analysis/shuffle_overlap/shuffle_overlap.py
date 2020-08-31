'''Module which generates kernel density estimates of distributions, 
calculates their overlap and also performs the shuffling of data
across the distributions. 

This module implements  shuffling + overlap calculation to assess
if two distributions are relatively similar or different from each other. 

Histograms of the two datasets are made according to user defined bins. 
These bins represent what the user thinks of as biologically and 
experimentally relevant. 

The overlap between the two histograms are then calculated using a variety of 
different similarity/overlap coefficients.

'''
import numpy as np 
from scipy import stats
#import joblib
#from joblib import Parallel, delayed



# def overlap_w_shuffling(main_distbn, other_distbn, **kwargs):
#     '''
#     1. Calculates overlap between the distributions of the two input 
#     distributions. 
#     2. Performs data shuffling betweent he two distbns, and calculates overlap
    
#     Parameters
#     ----------
#     main_distbn, other_distbn : array-like
#         The measurement/parameter values
#     num_points : int >0, optional 
#         Number of points to evaluate the PDF on between the global minimum 
#         and maximum. Defaults to 1000
#     num_shuffles : int>0, optional 
#         Number of shuffling rounds to be performed. Defaults to 2500
    
#     Returns 
#     -------
#     original_overlap : float
#         Overlap value between the PDFs of the two input distributions 
#     overlap_results : tuple
#         Intermediate data from calculate_overlap of the original datasets.
#     shuffled_overlap : array-like 
#         List with all overlap values from shuffled data

#     Note
#     ----
#     *This is not the full story - still needs to be clarified....*
#     The overlap calculated is the overlap of *non-normalised* counts/units that the 
#     input distribution is actually given in. The scipy.gaussian_kde.evaluate function 
#     actually returns the counts if counts are given, and PDF if pdf is given. 
    
#     See Also
#     --------
#     calculate_overlap
#     '''
#     original_overlap, overlap_results = calculate_overlap(main_distbn, other_distbn, **kwargs)
    
#     num_shuffles = kwargs.get('num_shuffles', 2500)
    

#     shuffled_overlap = Parallel(n_jobs=4)(delayed(shuffle_and_calc_overlap)(main_distbn, other_distbn, **kwargs) for i in range(num_shuffles))


#     return original_overlap, overlap_results, shuffled_overlap

def calculate_overlap(main_distbn, other_distbn, **kwargs):
    '''Calculates the overlap between two distributions through the following steps:
        
    1) Generate histogram from data using common bins for both datasets
    2) Quantify similarity of datsets by comparing similarity of histograms
       using overlap index of choice.

    Parameters
    ----------
    main_distbn, other_distbn : np.array 
        Each array is an 1 x N_points np.array 
    bin_width : float. 
        The width of the hisotgram bins used to bin all the data. 
    overlap_method : str, optional
        The method used to quantify the distribution overlap. Options
        include 'bhattacharya', 'hellinger'. Defaults to 'bhattacharya'.
    pmf : bool, optional
        Defaults to False
    
    Returns 
    -------
    overlap : float >0 
        The overlap between the kernel density estimates of the 
        two datasets
    (minmax_range, main_pdf, other_pdf) : tuple
        Tuple with intermediate data.

    Note
    ----
    The choice of overlap method is very important in how to interpret the results. 
    
    The Hellinger distance is a bounded distance between 0-np.sqrt(2). 0 means the two 
    distributions are identical, and higher numbers means the distributions 
    are more different. The lower the Hellinger distance, the more similar the
    two distributions are. 
    
    The Bhattacharya coefficient (BC) is an unbounded measure of similarity. 
    The higher the BC, the more similar the two distributions are. 

    Another important factor that will influence the calculated overlap measures
    is the user-defined 'bin_width'. If the bin width is too wide, then it will 
    obscure the actual similarity/differences between the two distributions. 

    See Also
    --------
    generate_histogram
    '''
    global_minmax = get_global_minmax(main_distbn, other_distbn)
    kwargs['minmax'] = global_minmax
    main_data_hist, bins = generate_histogram(main_distbn, **kwargs)
    other_data_hist, bins = generate_histogram(other_distbn, **kwargs)
   
    # overlap 
    overlap_method = kwargs.get('overlap_method', 'bhattacharyya')
    if overlap_method == 'bhattacharyya':
        overlap = bhattacharyya_coefficient(main_data_hist, other_data_hist)
    elif overlap_method == 'hellinger':
        overlap = hellinger_distance(main_data_hist, other_data_hist)
    else:
        raise ValueError(f'Invalid overlap index: {overlap_method} given!')
   
    return overlap, (global_minmax, main_data_hist, other_data_hist) 

def bhattacharyya_coefficient(P,Q):
    '''
    

    Parameters
    ----------
    P,Q : np.arrays
        Arrays with the probability mass functions across a set of bins

    Returns
    -------
    bhattacharyya_coef : float. 

    '''
    return np.sum(np.sqrt(P*Q))
    
def hellinger_distance(P,Q):
    '''
    
    References
    ----------
    https://en.wikipedia.org/wiki/Hellinger_distance , Accessed 24 August, 2020
    
    '''
    if np.logical_or(np.min([P,Q])<0, np.max([P,Q])>1):
        raise ValueError('probability values for Hellinger distance cannot be <0 or >1')
    bc = bhattacharyya_coefficient(P, Q)
    return np.sqrt(1-bc)

def generate_histogram(dataset, **kwargs):
    '''
    Generates a histogram of the input dataset. A set of bins are generated
    that include the minimum and maximum value that is user-defiend or obtained
    from the dataset.
    
    Parameters
    ----------
    dataset : np.array
    bin_width : float >0
    minimax : tuple, optional
        Tuple with two entries of the form (minimum, maximum).
        When NaNs are present in the data, they are just ignored.
        Defaults to the minimum and maximum of the input datset. 
    pmf : bool, optional 
        Whether to output the probability mass function or not. 
        Defaults to False, which leads to output of the counts only. 
    
    Returns 
    -------
    height : np.array
        Either the counts in each bin or the probability of the bin in 
        the probability mass function. This is decided by the 'pmf' argument. 
    bins : np.array
        The actual bins used to bin the data. Remember that in np.histogram
        the last bin is special in that it includes the right edge too. 
    
    See Also 
    --------
    np.histogram
   
    '''
    minmax = kwargs.get('minmax', (np.nanmin(dataset), np.nanmax(dataset)))
    data_bins = generate_bins(minmax, kwargs['bin_width'])
    height, bins = np.histogram(dataset, bins=data_bins)
    
    if kwargs.get('pmf', False):
        height = height/sum(height)
        
    return height, bins

def generate_bins(minmax, bin_width):
    minimum, maximum = minmax
    bins = np.arange(minimum, maximum, bin_width)
    if not maximum in bins:
        bins = np.concatenate((bins, np.array([maximum]))).flatten()
    return bins

def shuffle_and_calc_overlap(main_distbn, other_distbn, **kwargs):
    shuffled_1, shuffled_2 = shuffle_datasets(main_distbn, other_distbn)
    this_shuffled_overlap,_ = calculate_overlap(shuffled_1, shuffled_2, **kwargs)
    return this_shuffled_overlap


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
def generate_kde(data):
    '''
    Historial function from the time the overlap analysis happened with kernels
    '''
    return stats.gaussian_kde(data)
    
def percentile_score_of_value(X, distribution):
    return stats.percentileofscore(distribution, X)


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
        min_vals = np.nanmin(joint_dataset, 0)
        max_vals = np.nanmax(joint_dataset, 0)
    else:
        joint_dataset = np.concatenate((dist1, dist2))
        min_vals = np.nanmin(joint_dataset)
        max_vals = np.nanmax(joint_dataset)
    return min_vals, max_vals
    
def remove_nans(X):
    if np.sum(np.isnan(X)) >0:
        nan_indices = np.isnan(X)
        return X[~nan_indices]
    else:
        return X

# if __name__=='__main__':
#     import matplotlib.pyplot as plt 
    
#     num_datapoints = 2000
    
#     source  = np.random.normal(100000, 5000, num_datapoints*2)
#     source  = np.concatenate((np.random.normal(95000, 2500, num_datapoints), source)).flatten()
#     a = np.random.choice(source, num_datapoints)
#     b = np.random.choice(source, int(num_datapoints*0.25))
#     bc, data = calculate_overlap(a, b, bin_width=500, pmf=True, overlap_method='hellinger') 
#     print(bc)
    
    
#     obs, _, b = overlap_w_shuffling(a,b, bin_width=500, pmf=True,
#                                     overlap_method='hellinger')

#     plt.figure()
#     plt.hist(b)
#     plt.vlines(obs, 0, len(b)/2.0, 'r')

    


