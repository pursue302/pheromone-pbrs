import numpy as np

def find_best_results(time,series):
    
    best_times = []
    total_times =[]
    best_indexes = []
    best_values = []
    inds = 0
    for seq in time:
        
        seq_series = series[inds]
        
        index = np.argmin(seq_series)
        min_value = seq_series[index]
                
        best_indexes.append(index) 
        best_times.append(np.sum(seq[0:index]))
        total_times.append(np.sum(seq))
        best_values.append(min_value)
        inds+=1
        
    dict_pack = {"values":best_values, "indexes":best_indexes, "times":best_times, "execution_times":total_times} 
    return dict_pack




