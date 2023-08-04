#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:48:16 2020

@author: glavrent
"""
#load variables
#arithmetic libraries
import numpy as np
from scipy import linalg
#sparse matrices
from scipy import linalg as scipylalg
#statistics libraries
import pandas as pd


def ConvertPandasDf2NpArray(df_array):
    
    array = df_array.values if isinstance(df_array, pd.DataFrame) or isinstance(df_array, pd.Series) else df_array
    
    return array

#Moving statistics
#------------------
def RollingStats(t_array, x_array, tq_array, width, flag_log=False, flag_nan=True):
    
    #input assertions
    assert(len(x_array)==len(t_array)),'Error. x and t arrays are incompatible.'

    #convert to numpy arrays is they are pandas data-frames
    t_array  = ConvertPandasDf2NpArray(t_array)
    x_array  = ConvertPandasDf2NpArray(x_array)
    tq_array = ConvertPandasDf2NpArray(tq_array)
    
    #half width
    dw = width/2
    
    #reshape arrays
    x_array  = x_array.flatten()
    t_array  = (np.log(t_array)  if flag_log else t_array).flatten() 
    tq_array = (np.log(tq_array) if flag_log else tq_array).flatten() 
    
    if flag_nan:
        i_nan = ~np.isnan(x_array)
        t_array = t_array[i_nan]
        x_array = x_array[i_nan]
        
    #size of arrays
    n_x = len(t_array)
    n_q = len(tq_array)
        
    #indices of elements belonging to each bin
    i_bin     = [np.where((tq-dw <= t_array) & (t_array < tq+dw))[0]  for tq in tq_array] 
    #number of points per bin
    xq_npt = np.array([len(i_b) for i_b in i_bin])
    #non-empty input flags
    i_bin_full = np.where(xq_npt > 0)[0] 
    #keep only non-empy bins
    i_bin = [i_bin[i_b_f] for i_b_f in i_bin_full]
        
    #initialize arrays
    xq_mean = np.full(n_q, np.nan)
    xq_rms  = np.full(n_q, np.nan)
    xq_std  = np.full(n_q, np.nan)
    #moving mean
    xq_mean[i_bin_full] = np.array([x_array[i_b].mean()                 for i_b in i_bin])
    #moving standard deviation
    xq_std[i_bin_full]  = np.array([x_array[i_b].std(ddof=1)            for i_b in i_bin])
    #moving root mean square
    xq_rms[i_bin_full]  = np.array([np.sqrt( (x_array[i_b]**2).mean() ) for i_b in i_bin])
    
    return(xq_mean, xq_std, xq_rms, xq_npt)

#Piecewise regression and interpolation
#------------------
def PiecewiseReg(x_data, y_data, x_brk, 
                 x_llim =-np.infty, x_ulim = np.infty, y_llim =-np.infty, y_ulim = np.infty, 
                 flag_logx = False, flag_logy = False):

    #convert to numpy arrays    
    x_data = np.array([x_data]).flatten()
    y_data = np.array([y_data]).flatten()
    x_brk  = np.array([x_brk]).flatten()
    
    #number of data-points
    n_pt  = len(x_data)
    #number of breaks
    n_brk = len(x_brk)
    assert(n_brk>1),'Error. Number of break points should exceed 1'
    
    #covert to log values if flags are true
    #inputs
    x_data = np.log(x_data) if flag_logx else x_data
    x_brk  = np.log(x_brk)  if flag_logx else x_brk
    x_llim = np.log(np.max((x_llim,0)))  if flag_logy else x_llim
    x_ulim = np.log(x_ulim)              if flag_logy else x_ulim
    #dependent variable
    y_data = np.log(y_data)              if flag_logy else y_data
    y_llim = np.log(np.max((y_llim,0)))  if flag_logy else y_llim
    y_ulim = np.log(y_ulim)              if flag_logy else y_ulim
    
    #create regression matrix for piecewise interpolation
    X_data = [np.ones(n_pt)]
    for j, (x_b1, x_b2) in enumerate(zip(x_brk[:-1],x_brk[1:])):
        X_data.append( np.minimum(np.maximum(x_data-x_b1,0), x_b2-x_b1) ) 
    X_data = np.vstack(X_data).T
    
    #keep use only data within upper and lower limits
    # i_d_lim = np.logical_and(y_data>=y_llim ,y_data<=y_ulim)
    i_d_lim = np.all([x_data>=x_llim,x_data<=x_ulim,y_data>=y_llim,y_data<=y_ulim], axis=0)
    X_data = X_data[i_d_lim,:]
    y_data = y_data[i_d_lim]
    
    #least-square coefficients
    c_brk = np.array(linalg.lstsq(X_data, y_data)[0])
       
    return c_brk

def PiecewisePredict(x_new, x_brk, c_brk, flag_logx = False, flag_logy = False):

    #convert to numpy arrays    
    x_new = np.array([x_new]).flatten()
    x_brk = np.array([x_brk]).flatten()

    #covert to log values if flags are true
    #inputs
    x_new = np.log(x_new) if flag_logx else x_new
    x_brk = np.log(x_brk) if flag_logx else x_brk

    #number of data-points
    n_pt  = len(x_new)
    #number of breaks
    n_brk = len(x_brk)
    assert(n_brk>1),'Error. Number of break points should exceed 1'

    #create matrix for predictions
    X_new = [np.ones(n_pt)]
    for j, (x_b1, x_b2) in enumerate(zip(x_brk[:-1],x_brk[1:])):
        X_new.append( np.minimum(np.maximum(x_new-x_b1,0), x_b2-x_b1) ) 
    X_new = np.vstack(X_new).T

    #predictions
    y_new = X_new @ c_brk
    y_new = np.exp(y_new) if flag_logy else y_new
    
    return y_new

#piece-wise linear interpolation
def PiecewiseInt(x_q, y_q, x_brk, x_i,
                 x_llim =-np.infty, x_ulim = np.infty, y_llim =-np.infty, y_ulim = np.infty, 
                 flag_logx = False, flag_logy = False):
    
    #compute interpolation coefficients
    c_brk = PiecewiseReg(x_q, y_q, x_brk, x_llim, x_ulim, y_llim, y_ulim, flag_logx, flag_logy)
    
    #predictions
    y_i = PiecewisePredict(x_i, x_brk, c_brk, flag_logx, flag_logy)
   
    return y_i

# Multivariate normal distribution random samples
#------------------
def MVNRnd(mean=None, cov=None, seed=None, n_samp=None, flag_sp=False, flag_list=False):
    
    #if not already covert to list
    if flag_list:
        seed_list = seed if not seed is None else [None]
    else:
        seed_list = [seed]
        
    #number of dimensions
    n_dim = len(mean) if not mean is None else cov.shape[0]
    assert(cov.shape == (n_dim,n_dim)),'Error. Inconsistent size of mean array and covariance matrix'
        
    #set mean array to zero if not given
    if mean is None: mean = np.zeros(n_dim)

    #compute L D L' decomposition
    if flag_sp: cov = cov.toarray()
    L, D, _ = scipylalg.ldl(cov)
    assert( not np.count_nonzero(D - np.diag(np.diagonal(D))) ),'Error. D not diagonal'
    assert( np.all(np.diag(D) > -1e-1) ),'Error. D diagonal is negative'
    #extract diagonal from D matrix, set to zero any negative entries due to bad conditioning
    d      = np.diagonal(D).copy()
    d[d<0] = 0
    #compute Q matrix
    Q = L @ np.diag(np.sqrt(d))

    #generate random sample
    samp_list = list()
    for k, seed in enumerate(seed_list):
        #genereate seed numbers if not given 
        if seed is None: seed = np.random.standard_normal(size=(n_dim, n_samp))
   
        #generate random multi-normal random samples
        samp = Q @ (seed )
        samp += mean[:,np.newaxis] if samp.ndim > 1 else mean
        
        #summarize samples
        samp_list.append( samp )

    
    return samp_list if flag_list else samp_list[0]