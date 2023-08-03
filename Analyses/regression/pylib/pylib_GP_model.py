#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:39:04 2020

@author: glavrent
"""
#load variables
import pathlib
import glob
#arithmetic libraries
import numpy as np
from scipy import linalg
#statistics libraries
import pandas as pd

## User defined functions
# ---------------------------
# Contents:
# 1. Functions to sample non-ergodic terms at new locations
# 2. Functions to make conditional gm predictions at new locations
# 3. Functions to compute non-ergodic coefficinets from standardized variables
# 4. Functions to create covariance matrices

# Functions to sample non-ergodic terms at new locations
#---  ---  ---  ---  ---  ---  ---  ---
#function to sample coefficients at new locations
def SampleCoeffs(X_new, X_data, 
                 c_data_mu, c_data_sig = None,
                 hyp_ell = 0, hyp_omega = 0, hyp_pi = 0) :
    """Sample coefficients at new locations conditioned on the old ones"""
    
    #unique coordinates
    X_data_unq, i_eq_unq = np.unique(X_data, axis=0, return_index=True)    
    
    #number of data points
    n_pt_data = X_data.shape[0]
    assert(n_pt_data == len(i_eq_unq)),'Error. Non-unique coefficients'
    
    #convert mean to 1d array
    c_data_mu = c_data_mu.flatten()

    #uncertainty in estimating the data
    if c_data_sig is None: c_data_sig = np.zeros(n_pt_data)
    c_data_cov = np.diag(c_data_sig**2) if c_data_sig.ndim == 1 else c_data_sig
    assert( np.all(np.array(c_data_cov.shape) == n_pt_data) ),'Error. Inconsistent size of c_data_sig'
    
    #compute covariance between data 
    K      = CreateCovMatern(X_data, X_data, hyp_ell, hyp_omega, hyp_pi, delta=1e-9)
    #covariance between data and new locations
    k      = CreateCovMatern(X_new,  X_data, hyp_ell, hyp_omega, hyp_pi)
    #covariance between new locations
    k_star = CreateCovMatern(X_new,  X_new,  hyp_ell, hyp_omega, hyp_pi)
    
    #inverse of covariance matrix
    K_inv = linalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    c_new_mu  = kK_inv.dot(c_data_mu)
    c_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_data_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    c_new_sig = np.sqrt(np.diag(c_new_cov))
    
    return c_new_mu, c_new_sig, c_new_cov

#function to sample location terms
def SampledL2L(Lid_new, Lid_data, dL2L_data_mu, dL2L_data_sig, sigma_e):
    """Sample location terms at new locations conditioned on the old ones"""

    #number of data points
    n_pt_data = Lid_data.shape[0]
    
    #convert mean to 1d array
    dL2L_data_mu = dL2L_data_mu.flatten()
    
    #uncertainty in estimating the data
    if dL2L_data_sig is None: dL2L_data_sig = np.zeros(n_pt_data)
    dL2L_data_cov = np.diag(dL2L_data_sig**2) if dL2L_data_sig.ndim == 1 else dL2L_data_sig
    assert( np.all(np.array(dL2L_data_cov.shape) == n_pt_data) ),'Error. Inconsistent size of dL2L_data_cov'
    
    #compute covariance between data
    K = CreateCovL2L(Lid_data, Lid_data, sigma_e, delta = 1e-9)
    #covariance between data and new locations
    k = CreateCovL2L(Lid_new, Lid_data, sigma_e)
    #covariance between new locations
    k_star = CreateCovL2L(Lid_new, Lid_new, sigma_e)

    #inverse of covariance matrix
    K_inv = linalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at new locations
    dL2L_new_mu  = kK_inv.dot(dL2L_data_mu) 
    dL2L_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( dL2L_data_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at new locations
    dL2L_new_sig = np.sqrt(np.diag(dL2L_new_cov))
    
    return dL2L_new_mu, dL2L_new_sig, dL2L_new_cov

#function to sample station terms
def SampledS2S(Sid_new, Sid_data, dS2S_data_mu, dS2S_data_sig=None, sigma_s=0):
    """Sample site terms at new locations conditioned on the old ones"""

    #number of data points
    n_pt_data = Sid_data.shape[0]
    
    #convert mean to 1d array
    dS2S_data_mu = dS2S_data_mu.flatten()

    #uncertainty in estimating the data
    if dS2S_data_sig is None: dS2S_data_sig = np.zeros(n_pt_data)
    dS2S_data_cov = np.diag(dS2S_data_sig**2) if dS2S_data_sig.ndim == 1 else dS2S_data_sig
    assert( np.all(np.array(dS2S_data_cov.shape) == n_pt_data) ),'Error. Inconsistent size of dS2S_data_cov'

    #compute covariance between data
    K  = CreateCovS2S(Sid_data, Sid_data, sigma_s, delta = 1e-9)
    #covariance between data and new locations
    k = CreateCovS2S(Sid_new,  Sid_data, sigma_s)
    #covariance between new locations
    k_star = CreateCovS2S(Sid_new,  Sid_new,  sigma_s)

    #inverse of covariance matrix
    K_inv = linalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at grid node
    dS2S_new_mu  = kK_inv.dot(dS2S_data_mu) 
    dS2S_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( dS2S_data_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at grid node
    dS2S_new_sig = np.sqrt(np.diag(dS2S_new_cov))
    
    return dS2S_new_mu, dS2S_new_sig, dS2S_new_cov

#function to sample station terms
def SampledBe(eqid_new, eqid_data, dB_data_mu, dB_data_sig=None, tau_0=0):
    """Sample between event terms based on earthquake id"""

    #number of data points
    n_pt_data = eqid_data.shape[0]
    
    #convert mean to 1d array
    dB_data_mu = dB_data_mu.flatten()
    
    #uncertainty in estimating the data
    if dB_data_sig is None: dB_data_sig = np.zeros(n_pt_data)
    dB_data_cov = np.diag(dB_data_sig**2) if dB_data_sig.ndim == 1 else dB_data_sig
    assert( np.all(np.array(dB_data_cov.shape) == n_pt_data) ),'Error. Inconsistent size of dB_data_cov'
    
    #compute covariance between data
    K = CreateCovBe(eqid_data, eqid_data, tau_0, delta = 1e-9)
    #covariance between data and new locations
    k = CreateCovBe(eqid_new, eqid_data, tau_0)
    #covariance between new locations
    k_star = CreateCovBe(eqid_new, eqid_new, tau_0)

    #inverse of covariance matrix
    K_inv = linalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at grid node
    dB_new_mu  = kK_inv.dot(dB_data_mu)
    dB_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( dB_data_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at grid node
    dB_new_sig = np.sqrt(np.diag(dB_new_cov))
    
    return dB_new_mu, dB_new_sig, dB_new_cov

#function to sample attenuation cells
def SampleAttenCoeffsNegExp(T_cells_new, X_cells_data, cA_data_mu, cA_data_sig = None,
                            mu_ca = 0, ell_ca = 1e-9, omega_ca = 0, sigma_ca = 0, pi_ca = 0):
    """Sample cell coefficients at new locations conditioned on the old ones"""
    
    #number of data points
    n_pt_data = X_cells_data.shape[0]
    n_pt_new  = T_cells_new.shape[0]
    
    #remove mean effect
    cA_data_mu = cA_data_mu.copy() - mu_ca
    
    #uncertainty in estimating the data
    if cA_data_sig is None: cA_data_sig = np.zeros(n_pt_data)
    cA_data_cov = np.diag(cA_data_sig**2) if cA_data_sig.ndim == 1 else cA_data_sig
    assert( np.all(np.array(cA_data_sig.shape) == n_pt_data) ),'Error. Inconsistent size of cA_data_sig'
    
    #path lengths of training data new locations
    L_new = np.eye(n_pt_new)
    L_data = np.eye(n_pt_data)
    
    # import pdb; pdb.set_trace()
    #compute covariance between training data
    K = CreateCovCellsNegExp(L_data, L_data, X_cells_data, X_cells_data,
                              ell_ca, omega_ca, sigma_ca, pi_ca, delta = 1e-16)
    #covariance between data and new locations
    k = CreateCovCellsNegExp(L_new, L_data, T_cells_new, X_cells_data,
                              ell_ca, omega_ca, sigma_ca, pi_ca)
    
    #compute covariance matrix between grid points
    k_star = CreateCovCellsNegExp(L_new, L_new, T_cells_new, T_cells_new,
                                   ell_ca, omega_ca, sigma_ca, pi_ca)
    
    #inverse of covariance matrix
    K_inv = linalg.inv(K)
    #product of k * K^-1
    kK_inv = k.dot(K_inv)
    
    #posterior mean and variance at grid nodes
    cA_new_mu  = kK_inv.dot(cA_data_mu)
    cA_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( cA_data_cov.dot(kK_inv.transpose()) )
    #posterior standard dev. at grid node
    cA_new_sig = np.sqrt(np.diag(cA_new_cov))
    #add mean effect on coefficients
    cA_new_mu += mu_ca
    
    return cA_new_mu.flatten(), cA_new_sig.flatten(), cA_new_cov

# Functions to make conditional gm predictions at new locations
#---  ---  ---  ---  ---  ---  ---  ---
def GPPrediction(y_train, X_train,      T_train,      eqid_train,      sid_train = None, lid_train = None,
                          X_new = None, T_new = None, eqid_new = None, sid_new   = None, lid_new   = None,
                 dc_0 = 0.,
                 Tid_list = None, Hyp_list = None, phi_0 = None, tau_0 = None,
                 sigma_s = None, sigma_e = None):
    """
    Make ground motion predictions at new locations conditioned on the training data
    
    Parameters
    ----------
    y_train : np.array(n_train_pt)
        Array with ground-motion observations associated with training data
    X_train : np.array(n_train_pt, n_dim)
        Design matrix for training data.
    T_train : np.array(n_train_pt, 2x n_coor)
        Coordinates matrix for training data.
    eqid_train : np.array(n_train_pt)
        Earthquake IDs for training data.
    sid_train : np.array(n_train_pt), optional
        Station IDs for training data. The default is None.
    lid_train : np.array(n_train_pt), optional
        Source IDs for training data. The default is None.
    X_new : np.array(n_new_pt, n_dim), optional
        Desing matrix for predictions. The default is None.
    T_new : np.array(n_new_pt, 2 x n_coor), optional
        Coordinate matrix for predictions. The default is None.
    eqid_new : np.array(n_new_pt), optional
        Earthquake IDs for predictions. The default is None.
    sid_new : np.array(n_new_pt), optional
        Station IDs for predictions. The default is None.
    lid_new : np.array(n_new_pt), optional
        Source IDs for predictions. The default is None.
    dc_0 : float, optional
        Mean offset. The default is zero.
    Tid_list : n_dim list
        List to specify the coordinate pair or each dimension.
    Hyp_list : TYPE, optional
        List of hyper-parameters for each dimension of the covariance fuction.
    phi_0 : double
        Within-event standard deviation.
    tau_0 : double
        Between-event standard deviation.
    sigma_s : double, optional
        Standard deviation for zero correlation site-to-site term. The default is None.
    sigma_e : double, optional
        Standard deviation for zero correlation source-to-source term. The default is None.

    Returns
    -------
    np.array(n_new_pt)
        median estimate of new predictions.
    np.array(n_new_pt, n_new_pt)
        epistemic uncertainty of new predictions.

    """
    
    #import pdb; pdb.set_trace()
    
    #remove mean offset from conditioning data
    y_train = y_train - dc_0
    
    #number of grid nodes
    n_pt_train = X_train.shape[0]
    n_pt_new = X_new.shape[0]
    
    #initialize covariance matrices
    cov_data = np.zeros([n_pt_train,n_pt_train]) 
    cov_star = np.zeros([n_pt_new,n_pt_train])
    cov_star2 = np.zeros([n_pt_new,n_pt_new])
    
    
    #create covariance matrices
    for k, (hyp, tid) in enumerate(zip(Hyp_list,Tid_list)):
        #covariance between train data
        cov_data +=  CreateCovMaternDimX(X_train[:,k], X_train[:,k], 
                                         T_train[tid], T_train[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 1e-6)
    
        #covariance between train data and predictions
        cov_star +=  CreateCovMaternDimX(X_new[:,k], X_train[:,k], 
                                         T_new[tid], T_train[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 0)
    
        #covariance between prediction data
        cov_star2 += CreateCovMaternDimX(X_new[:,k], X_new[:,k], 
                                         T_new[tid], T_new[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 1e-6)

    #add site to site systematic effects if sigma_s is specified
    if not (sigma_s is None):
        assert(not(sid_train is None)), 'Error site id for training data not specified'
        cov_data += CreateCovS2S(sid_train, sid_train, sigma_s, delta = 1e-6)

    #add source to source systematic effects if phi_L2L is specified
    if not (sigma_e is None):
        assert(not(lid_train is None)), 'Error location id for training data not specified'
        cov_data += CreateCovL2L(lid_train, lid_train, sigma_e, delta = 1e-6)   

    #add between and within event covariance matrices
    cov_data += CreateCovWe(eqid_train, eqid_train, phi_0)
    cov_data += CreateCovBe(eqid_train, eqid_train, tau_0, delta = 1e-6)
    
    #consider site to site systematic effects in predictions if sigma_s is specified
    if not ( (sigma_s is None) or (sid_new is None)):
        cov_star2 += CreateCovS2S(sid_new, sid_new,  sigma_s, delta = 1e-6)
        cov_star  += CreateCovS2S(sid_new, sid_train, sigma_s)
            
    #consider site to site systematic effects in predictions if sigma_s is specified
    if not ( (sigma_e is None) or (lid_new is None)):
        cov_star2 += CreateCovL2L(lid_new, lid_new,  sigma_e, delta = 1e-6)
        cov_star  += CreateCovL2L(lid_new, lid_train, sigma_e)

    #consider earthquake aleatory terms if eqid_new is specified
    if not (eqid_new is None):
        cov_star2 += CreateCovBe(eqid_new, eqid_new,   tau_0, delta = 1e-6)
        cov_star  += CreateCovBe(eqid_new, eqid_train, tau_0)

    #posterior mean and variance at new locations
    y_new_mu =  cov_star.dot(linalg.solve(cov_data, y_train))
    #add mean offset to new predictions
    y_new_mu = y_new_mu + dc_0
        
    y_new_cov = cov_star2 - cov_star.dot(linalg.solve(cov_data, cov_star.transpose()))
    #posterior standard dev. at new locations
    y_new_sig = np.sqrt(np.diag(y_new_cov))        
       
    return y_new_mu.flatten(), y_new_sig.flatten(), y_new_cov


def GPPredictionCells(y_train, X_train,      T_train,      eqid_train,      sid_train = None, lid_train = None,
                               X_new = None, T_new = None, eqid_new = None, sid_new   = None, lid_new   = None,
                       dc_0 = 0.,
                       Tid_list = None, Hyp_list = None, phi_0 = None, tau_0 = None,
                       sigma_s = None, sigma_e = None, 
                       T_cells_new = None, T_cells_train = None, L_cells_new = None, L_cells_train = None,
                       mu_ca = 0, ell_ca = 0, omega_ca = 0, sigma_ca = 0, pi_ca = 0):
    """
    Make ground motion predictions at new locations conditioned on the training data
    
    Parameters
    ----------
    y_train : np.array(n_train_pt)
        Array with ground-motion observations associated with training data
    X_train : np.array(n_train_pt, n_dim)
        Design matrix for training data.
    T_train : np.array(n_train_pt, 2x n_coor)
        Coordinates matrix for training data.
    eqid_train : np.array(n_train_pt)
        Earthquake IDs for training data.
    sid_train : np.array(n_train_pt), optional
        Station IDs for training data. The default is None.
    lid_train : np.array(n_train_pt), optional
        Source IDs for training data. The default is None.
    X_new : np.array(n_new_pt, n_dim), optional
        Desing matrix for predictions. The default is None.
    T_new : np.array(n_new_pt, 2 x n_coor), optional
        Coordinate matrix for predictions. The default is None.
    eqid_new : np.array(n_new_pt), optional
        Earthquake IDs for predictions. The default is None.
    sid_new : np.array(n_new_pt), optional
        Station IDs for predictions. The default is None.
    lid_new : np.array(n_new_pt), optional
        Source IDs for predictions. The default is None.
    dc_0 : float, optional
        Mean offset. The default is zero.
    Tid_list : n_dim list
        List to specify the coordinate pair or each dimension.
    Hyp_list : TYPE, optional
        List of hyper-parameters for each dimension of the covariance fuction.
    phi_0 : double
        Within-event standard deviation.
    tau_0 : double
        Between-event standard deviation.
    sigma_s : double, optional
        Site-to-site standard deviation. The default is None.
    sigma_e : double, optional
        Source-to-source standard deviation. The default is None.
    T_cells_new : np.array(n_c_new,2) , optional
        Coordinate matrix for cells for new predictions
    T_cells_train : np.array(n_c_train,2) , optional
        Coordinate matrix for cells for training data
    L_cells_new : np.array() , optional 
        Cell path matrix for new predictions.    
    L_cells_train : np.array() , optional
        Cell path matrix for new training data.
    mu_ca : real, optional
        Mean of cell attenuation.
    ell_ca : real, optional
        Correlation length for anelastic attenuation cells.
    omega_ca : real, optional
        Standard-deviation of spatially varying anelastic attenuation cells. 
    sigma_ca, real, optional 
        Uncorrelated standard-deviation of cell attenuation
    pi_ca : real, optional
        Constant standard-deviation of spatially varying anelastic attenuation cells. 
    Returns
    -------
    np.array(n_new_pt)
        median estimate of new predictions.
    np.array(n_new_pt, n_new_pt)
        epistemic uncertainty of new predictions.

    """
    
    #import pdb; pdb.set_trace()
    
    #number of grid nodes
    n_pt_train = X_train.shape[0]
    n_pt_new = X_new.shape[0]
    
    #number of cells 
    n_c_train = T_cells_train.shape[0]
    n_c_new = T_cells_new.shape[0]
        
    #remove mean offset from conditioning data
    y_atten_train = np.matmul(L_cells_train, np.ones(n_c_train) * mu_ca)
    y_train       = y_train - dc_0 - y_atten_train
    
    
    #initialize covariance matrices
    cov_data = np.zeros([n_pt_train,n_pt_train]) 
    cov_star = np.zeros([n_pt_new,n_pt_train])
    cov_star2 = np.zeros([n_pt_new,n_pt_new])
    
    #create covariance matrices
    for k, (hyp, tid) in enumerate(zip(Hyp_list,Tid_list)):
        #covariance between train data
        cov_data +=  CreateCovMaternDimX(X_train[:,k], X_train[:,k], 
                                         T_train[tid], T_train[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 1e-9)
    
        #covariance between train data and predictions
        cov_star +=  CreateCovMaternDimX(X_new[:,k], X_train[:,k], 
                                         T_new[tid], T_train[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 0)
    
        #covariance between prediction data
        cov_star2 += CreateCovMaternDimX(X_new[:,k], X_new[:,k], 
                                         T_new[tid], T_new[tid],
                                         hyp_ell = hyp[0], hyp_omega = hyp[1], hyp_pi = hyp[2],
                                         delta = 1e-9)

    #add site to site systematic effects if sigma_s is specified
    if not (sigma_s is None):
        assert(not(sid_train is None)), 'Error site id for training data not specified'
        cov_data += CreateCovS2S(sid_train, sid_train, sigma_s, delta = 1e-9)

    #add source to source systematic effects if phi_L2L is specified
    if not (sigma_e is None):
        assert(not(lid_train is None)), 'Error location id for training data not specified'
        cov_data += CreateCovL2L(lid_train, lid_train, sigma_e, delta = 1e-9)   

    #add between and within event covariance matrices
    cov_data += CreateCovWe(eqid_train, eqid_train, phi_0)
    cov_data += CreateCovBe(eqid_train, eqid_train, tau_0, delta = 1e-9)
    
    #consider site to site systematic effects in predictions if sigma_s is specified
    if not ( (sigma_s is None) or (sid_new is None)):
        cov_star2 += CreateCovS2S(sid_new, sid_new,  sigma_s, delta = 1e-9)
        cov_star  += CreateCovS2S(sid_new, sid_train, sigma_s)
            
    #consider site to site systematic effects in predictions if sigma_s is specified
    if not ( (sigma_e is None) or (lid_new is None)):
        cov_star2 += CreateCovL2L(lid_new, lid_new,  sigma_e, delta = 1e-9)
        cov_star  += CreateCovL2L(lid_new, lid_train, sigma_e)

    #consider earthquake aleatory terms if eqid_new is specified
    if not (eqid_new is None):
        cov_star2 += CreateCovBe(eqid_new, eqid_new,   tau_0, delta = 1e-9)
        cov_star  += CreateCovBe(eqid_new, eqid_train, tau_0)

    #consider cell att
    cov_data += CreateCovCellsNegExp(L_cells_train, L_cells_train, T_cells_train, T_cells_train,
                                     ell_ca, omega_ca, sigma_ca, pi_ca, delta = 1e-9)
    cov_star += CreateCovCellsNegExp(L_cells_new, L_cells_train, T_cells_new, T_cells_train,
                                     ell_ca, omega_ca, sigma_ca, pi_ca)
    cov_star2 += CreateCovCellsNegExp(L_cells_new, L_cells_new, T_cells_new, T_cells_new,
                                      ell_ca, omega_ca, sigma_ca, pi_ca, delta = 1e-9)


    #posterior mean and variance at new locations
    y_new_mu =  cov_star.dot(linalg.solve(cov_data, y_train))
    #add mean offset to new predictions
    y_new_atten = np.matmul( L_cells_new, np.ones(n_c_new) * mu_ca )   
    y_new_mu    = y_new_mu + dc_0 + y_new_atten
        
    y_new_cov = cov_star2 - cov_star.dot(linalg.solve(cov_data, cov_star.transpose()))
    #posterior standard dev. at new locations
    y_new_sig = np.sqrt(np.diag(y_new_cov))        
       
    return y_new_mu.flatten(), y_new_sig.flatten(), y_new_cov

# Functions to compute non-ergodic coefficinets from standardized variables
#---  ---  ---  ---  ---  ---  ---  ---
#function to compute coefficients at event or station locations
def ComputeCoeffsEventOrSta(X_cor, z_array, delta = 1e-9,
                           hyp_ell = 0, hyp_omega = 0, hyp_pi = 0):
    """
    Compute coefficients at event or station locations based on standardized
    variables
    """
    #number of covariates
    n_pt = X_cor.shape[0]

    #create cov. matrix
    cov_mat = np.zeros([n_pt,n_pt]) #initialize
    for i in range(n_pt):
        dist =linalg.norm(X_cor[i] - X_cor[:,:],axis=1)
        cov_mat[i,:] = hyp_pi**2 + hyp_omega ** 2 * np.exp(- dist/hyp_ell)
        cov_mat[i,i] += delta

    #compute coeffs considering the correlation structure 
    chol_mat = linalg.cholesky(cov_mat).transpose()
    coeff_array = chol_mat.dot(z_array)

    return coeff_array, cov_mat


# Functions to create covariance matrices
#---  ---  ---  ---  ---  ---  ---  ---
def CreateCovMatern(t_1, t_2,
                    hyp_ell = 0, hyp_omega = 0, hyp_pi = 0, delta = 1e-9):
    "Compute Matern Matern kernel function"

    #number of grid nodes
    n_pt_1 = t_1.shape[0]
    n_pt_2 = t_2.shape[0]
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        dist =linalg.norm(t_1[i] - t_2[:,:],axis=1)
        cov_mat[i,:] = hyp_pi**2 + hyp_omega ** 2 * np.exp(- dist/hyp_ell)
    
    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

def CreateCovMaternDimX(x_1, x_2, t_1, t_2,
                        hyp_ell = 0, hyp_omega = 0, hyp_pi = 0,
                        delta = 1e-9):
    "Compute Matern single dimention kernel function"

    #number of grid nodes
    n_pt_1 = x_1.shape[0]
    n_pt_2 = x_2.shape[0]
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        dist =linalg.norm(t_1[i] - t_2[:,:],axis=1)
        cov_mat[i,:] = x_1[i] * x_2 * (hyp_pi**2 + hyp_omega ** 2 * np.exp(- dist/hyp_ell))
    
    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

def CreateCovBe(eq_1, eq_2, tau_0, delta = 0):
    "Compute between event covariance matrix"
    
    #tolerance for location id comparison
    r_tol = np.min([0.01/np.max([np.abs(eq_1).max(), np.abs(eq_2).max()]), 1e-11])
    #number of grid nodes
    n_pt_1 = eq_1.shape[0]
    n_pt_2 = eq_2.shape[0]
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        cov_mat[i,:] =  tau_0**2 * np.isclose(eq_1[i], eq_2, rtol=r_tol).flatten()
    
    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

def CreateCovWe(eq_1, eq_2, phi_0, delta = 0):
    "Compute within event covariance matrix"

    #number of grid nodes
    n_pt_1 = eq_1.shape[0]
    n_pt_2 = eq_2.shape[0]
    
    #create cov. matrix
    cov_mat = (phi_0**2 + delta) * np.eye(n_pt_1,n_pt_2)

    return cov_mat

def CreateCovL2L(loc_1, loc_2, sigma_e, delta = 0):
    "Compute location to location covariance matrix"

    #tolerance for location id comparison
    r_tol = np.min([0.01/np.max([np.abs(loc_1).max(), np.abs(loc_2).max()]), 1e-11])
    #number of grid nodes
    n_pt_1 = loc_1.shape[0]
    n_pt_2 = loc_2.shape[0]
    
    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        cov_mat[i,:] =  sigma_e**2 * np.isclose(loc_1[i], loc_2, rtol=r_tol).flatten()

    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

def CreateCovS2S(sta_1, sta_2, sigma_s, delta = 0):
    "Compute site to site covariance matrix"
    
    #tolerance for  station id comparison
    r_tol = np.min([0.01/np.max([np.abs(sta_1).max(), np.abs(sta_2).max()]), 1e-11])
    #number of grid nodes
    n_pt_1 = sta_1.shape[0]
    n_pt_2 = sta_2.shape[0]

    #create cov. matrix
    cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
    for i in range(n_pt_1):
        cov_mat[i,:] =  sigma_s**2 * np.isclose(sta_1[i], sta_2, rtol=r_tol).flatten()

    if n_pt_1 == n_pt_2:
        for i in range(n_pt_1):
            cov_mat[i,i] += delta

    return cov_mat

def CreateCovCellsNegExp(L_1, L_2, t_cells_1, t_cells_2, ell_ca, omega_ca, sigma_ca, pi_ca, delta = 0, dthres = 1e-1):
    "Compute cell covariance matrix based on a negative exponential kernel function"
    
    #number of cells in training and prediction data-sets
    n_c_1 = t_cells_1.shape[0]
    n_c_2 = t_cells_2.shape[0]
    
    #create cell cov. matrix
    covm_cells = np.zeros([n_c_1,n_c_2]) #initialize
    #correlated part
    for i in range(n_c_1):
        dist =linalg.norm(t_cells_1[i] - t_cells_2[:,:],axis=1)
        covm_cells[i,:] = pi_ca**2 + omega_ca ** 2 * np.exp(- dist/ ell_ca )
    #independent part
    for i in range(n_c_1):
        dist =linalg.norm(t_cells_1[i] - t_cells_2[:,:],axis=1)
        covm_cells[i, dist<dthres ] += sigma_ca ** 2
    
    if n_c_1 == n_c_2:
        for i in range(n_c_1):
            covm_cells[i,i] += delta

    #covariance matrix for predictions
    cov_mat = np.matmul(L_1, np.matmul(covm_cells, L_2.T))
        
    return cov_mat

def CreateCovCellsSqExp(L_1, L_2, t_cells_1, t_cells_2, ell_ca, omega_ca, pi_ca, delta = 0):
    "Compute cell covariance matrix based on a squared exponential kernel function"
    
    #number of cells in training and prediction data-sets
    n_c_1 = t_cells_1.shape[0]
    n_c_2 = t_cells_2.shape[0]
    
    #create cell cov. matrix
    covm_cells = np.zeros([n_c_1,n_c_2]) #initialize
    for i in range(n_c_1):
        dist =linalg.norm(t_cells_1[i] - t_cells_2[:,:],axis=1)
        covm_cells[i,:] = pi_ca**2 + omega_ca ** 2 * np.exp(- dist**2/ ell_ca**2 )
        
    if n_c_1 == n_c_2:
        for i in range(n_c_1):
            covm_cells[i,i] += delta
        
    #covariance matrix for predictions
    cov_mat = np.matmul(L_1, np.matmul(covm_cells, L_2.T))
        
    return cov_mat
