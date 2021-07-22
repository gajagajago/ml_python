import numpy as np

def multi_var_numerical_derivative(f, np_obj):
    delta = 1e-4
    
    res = np.zeros_like(np_obj)
    
    it = np.nditer(np_obj, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        x = np_obj[idx] # flag to save org value
        
        np_obj[idx] = x + delta # x + delta_x
        f_x_plus_delta = f(np_obj)
    
        
        np_obj[idx] = x - delta  # x - delta_x
        f_x_minus_delta = f(np_obj)
        
        res[idx] = ( f_x_plus_delta - f_x_minus_delta ) / (2*delta)
        
        np_obj[idx] = x # restore x value
        
        it.iternext()
    
    return res
