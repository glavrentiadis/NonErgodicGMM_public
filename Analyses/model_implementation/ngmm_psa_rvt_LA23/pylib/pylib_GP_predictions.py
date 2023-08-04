#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 19:01:59 2020

@author: glavrent
"""

import numpy as np
from scipy import linalg
import pylib.pylib_GP_model as pygp

def SamplePredictFromCoeffs(n_samp, c_train, hyp_list, Tid_list, X_train = None, T_train = None, X_new = None, T_new = None,
                                             b_0 = 0.,
                                             phi0 = None, 
                                             tau0 = None,    dB_train = None,   eqid_train = None, eqid_new = None,
                                             phiS2S = None,  dS2S_train = None, sid_train = None,  sid_new = None,
                                             tauL2L = None,  dL2L_train = None, lid_train = None,  lid_new = None,
                                             mu_cA = 0, rho_cA = 0, theta_cA = 0, sigma_cA = 0, pi_cA = 0,
                                             cA_train = None, 
                                             T_cells_train = None, L_cells_train = None, 
                                             T_cells_new = None, L_cells_new = None):
    """
    

    Parameters
    ----------
    n_samp : TYPE
        DESCRIPTION.
    c_train : TYPE
        DESCRIPTION.
    hyp_list : TYPE
        DESCRIPTION.
    Tid_list : TYPE
        DESCRIPTION.
    X_train : TYPE, optional
        DESCRIPTION. The default is None.
    T_train : TYPE, optional
        DESCRIPTION. The default is None.
    X_new : TYPE, optional
        DESCRIPTION. The default is None.
    T_new : TYPE, optional
        DESCRIPTION. The default is None.
    b_0 : TYPE, optional
        DESCRIPTION. The default is 0..
    phi0 : TYPE, optional
        DESCRIPTION. The default is None.
    tau0 : TYPE, optional
        DESCRIPTION. The default is None.
    dB_train : TYPE, optional
        DESCRIPTION. The default is None.
    eqid_train : TYPE, optional
        DESCRIPTION. The default is None.
    eqid_new : TYPE, optional
        DESCRIPTION. The default is None.
    phiS2S : TYPE, optional
        DESCRIPTION. The default is None.
    dS2S_train : TYPE, optional
        DESCRIPTION. The default is None.
    sid_train : TYPE, optional
        DESCRIPTION. The default is None.
    sid_new : TYPE, optional
        DESCRIPTION. The default is None.
    tauL2L : TYPE, optional
        DESCRIPTION. The default is None.
    dL2L_train : TYPE, optional
        DESCRIPTION. The default is None.
    lid_train : TYPE, optional
        DESCRIPTION. The default is None.
    lid_new : TYPE, optional
        DESCRIPTION. The default is None.
    mu_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    rho_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    theta_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    sigma_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    pi_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    cA_train : TYPE, optional
        DESCRIPTION. The default is None.
    T_cells_train : TYPE, optional
        DESCRIPTION. The default is None.
    L_cells_train : TYPE, optional
        DESCRIPTION. The default is None.
    T_cells_new : TYPE, optional
        DESCRIPTION. The default is None.
    L_cells_new : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    #import pdb; pdb.set_trace()
    
    #number of prediciton data
    n_pt_new = X_new.shape[0]
    #number of cells 
    if not (cA_train is None):
        n_c_new = T_cells_new.shape[0]
    
    #initialize prediction array
    y_new = b_0 * np.ones([n_pt_new,n_samp])
    
    #sample spatially varying coefficients
    k = 1;
    hyp = hyp_list[k]
    tid = Tid_list[k]
    for k, (hyp, tid) in enumerate(zip(hyp_list,Tid_list)):
        #determine unique spatially varing coefficients
        _, c_idx, c_inv = np.unique(T_new[tid], axis=0, return_inverse=True, return_index=True)
        #compute mean and covariance of spatially varing coefficients
        c_new_mu, _, c_new_cov = pygp.SampleCoeffs(T_new[tid][c_idx], T_train[tid], c_data_mu = c_train[k],
                                                   hyp_rho = hyp[0], hyp_theta = hyp[1], hyp_pi = hyp[2])
        #compute cholesky of covariance matrix
        c_new_chol = linalg.cholesky(c_new_cov, lower=True)
        #sample coefficients at grid locations
        c_new_samp = c_new_mu[:, np.newaxis] + c_new_chol.dot( np.random.normal(size=[len(c_idx),n_samp]) )
        #add effect of coefficients
        y_new +=  c_new_samp[c_inv,:] * X_new[:,k][:, np.newaxis]
        
        
    #add site to site systematic effects if phi_S2S is specified
    if not (phiS2S is None):
        assert(not(sid_train is None)), 'Error site id for training data not specified'
        #determine unique stations
        _, s_idx, s_inv = np.unique(sid_new, axis=0, return_inverse=True, return_index=True)
        #compute mean and covariance of station terms
        dS2S_new_mu, _, dS2S_new_cov = pygp.SampledS2S(sid_new[s_idx], sid_train, dS2S_train, phiS2S)
        #compute cholesky of covariance matrix
        dS2S_new_chol = linalg.cholesky(dS2S_new_cov, lower=True)
        #site term coefficients at grid locations
        dS2S_new_samp = dS2S_new_mu[:, np.newaxis] + dS2S_new_chol.dot( np.random.normal(size=[len(s_idx),n_samp]) )
        #add effect of coefficients
        y_new += dS2S_new_samp[s_inv,:]
        
    #add source to source site systematic effects if tau_L2L is specified
    if not (tauL2L is None):
        assert(not(sid_train is None)), 'Error site id for training data not specified'
        #determine unique sources
        _, l_idx, l_inv = np.unique(sid_new, axis=0, return_inverse=True, return_index=True)
        #compute mean and covariance of source terms
        dL2L_new_mu, _, dL2L_new_cov = pygp.SampledL2L(lid_new[l_idx], lid_train, dL2L_train, tauL2L)
        #compute cholesky of covariance matrix
        dL2L_new_chol = linalg.cholesky(dS2S_new_cov, lower=True)
        #sample location terms at grid locations
        dL2L_new_samp = dL2L_new_mu[:, np.newaxis] + dL2L_new_chol.dot( np.random.normal(size=[len(l_idx),n_samp]) )
        #add effect of coefficients
        y_new += dL2L_new_samp[l_inv,:]    
        
    #add cell effects
    if not (cA_train is None):
        n_c_new = T_cells_new.shape[0]
        cA_new_mu, _, cA_new_cov = pygp.SampleAttenCoeffsNegExp2(T_cells_new, T_cells_train, cA_train,
                                                                 mu_cA, rho_cA, theta_cA, sigma_cA, pi_cA)
        #compute cholesky of covariance matrix
        cA_new_chol = linalg.cholesky(cA_new_cov, lower=True)  
        #sample location terms at grid locations
        cA_new_samp = cA_new_mu[:, np.newaxis] + cA_new_chol.dot( np.random.normal(size=[n_c_new,n_samp]) )
        cA_new_samp[cA_new_samp > 0]  = 0
        #add effect of cells
        y_new += L_cells_new.dot(cA_new_samp)
    
    #estimate aleatory terms
    if not (eqid_new is None):
        #estimate between event terms based on earthquake ids
        dB_mu, _, dB_cov = pygp.SampledBe(eqid_new, eqid_train, dS2S_train, tau0)
        aleat_cov = phi0**2 * np.eye(n_pt_new) + dB_cov;
    else:
        dB_mu = np.zeros(n_pt_new)
        aleat_cov = (phi0**2 + tau0**2) * np.eye(n_pt_new)
        
    aleat_sig = np.sqrt(np.diag(aleat_cov))
        
    return(y_new, dB_mu, aleat_sig, aleat_cov)

#Sample predictions based on     
def SamplePredictFromCondModel(n_samp, y_train, hyp_list, Tid_list, X_train = None, T_train = None, 
                                                                    X_new = None,   T_new = None,
                                                b_0 = 0.,
                                                phi0 = None, 
                                                tau0 = None,    dB_train = None, eqid_train = None, eqid_new = None,
                                                phiS2S = None,                   sid_train = None,  sid_new = None,
                                                tauL2L = None,                   lid_train = None,  lid_new = None,
                                                mu_cA = 0, rho_cA = 0, theta_cA = 0, sigma_cA = 0, pi_cA = 0,
                                                T_cells_train = None, L_cells_train = None, 
                                                T_cells_new = None,   L_cells_new = None):
    """
    

    Parameters
    ----------
    n_samp : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    hyp_list : TYPE
        DESCRIPTION.
    Tid_list : TYPE
        DESCRIPTION.
    X_train : TYPE, optional
        DESCRIPTION. The default is None.
    T_train : TYPE, optional
        DESCRIPTION. The default is None.
    X_new : TYPE, optional
        DESCRIPTION. The default is None.
    T_new : TYPE, optional
        DESCRIPTION. The default is None.
    b_0 : TYPE, optional
        DESCRIPTION. The default is 0..
    phi0 : TYPE, optional
        DESCRIPTION. The default is None.
    tau0 : TYPE, optional
        DESCRIPTION. The default is None.
    dB_train : TYPE, optional
        DESCRIPTION. The default is None.
    eqid_train : TYPE, optional
        DESCRIPTION. The default is None.
    eqid_new : TYPE, optional
        DESCRIPTION. The default is None.
    phiS2S : TYPE, optional
        DESCRIPTION. The default is None.
    sid_train : TYPE, optional
        DESCRIPTION. The default is None.
    sid_new : TYPE, optional
        DESCRIPTION. The default is None.
    tauL2L : TYPE, optional
        DESCRIPTION. The default is None.
    lid_train : TYPE, optional
        DESCRIPTION. The default is None.
    lid_new : TYPE, optional
        DESCRIPTION. The default is None.
    mu_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    rho_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    theta_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    sigma_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    pi_cA : TYPE, optional
        DESCRIPTION. The default is 0.
    T_cells_train : TYPE, optional
        DESCRIPTION. The default is None.
    L_cells_train : TYPE, optional
        DESCRIPTION. The default is None.
    T_cells_new : TYPE, optional
        DESCRIPTION. The default is None.
    L_cells_new : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    #number of prediciton data
    n_pt_new = X_new.shape[0]
    
    #import pdb; pdb.set_trace()

    
    y_new_mu, _, y_new_cov = pygp.GPPredictionCells2(y_train, X_train = X_train, T_train = T_train, 
                                                              eqid_train = eqid_train, sid_train = sid_train, lid_train = lid_train,
                                                              X_new = X_new,  T_new = T_new,
                                                              eqid_new = None, sid_new = sid_new, lid_new = lid_new,
                                                              b_0 = b_0,
                                                              Tid_list = Tid_list, Hyp_list = hyp_list, 
                                                              phi_v = phi0, tau_v = tau0, phi_S2S = phiS2S, 
                                                              T_cells_new = T_cells_new, T_cells_train = T_cells_train, 
                                                              L_cells_new = L_cells_new, L_cells_train = L_cells_train,
                                                              mu_cA = mu_cA, rho_cA = rho_cA, theta_cA = theta_cA, 
                                                              pi_cA = pi_cA, sigma_cA = sigma_cA)

    #compute cholesky of covariance matrix
    y_new_chol = linalg.cholesky(y_new_cov, lower=True)
    #sample ground motions at grid locations
    y_new = y_new_mu[:, np.newaxis] + y_new_chol.dot( np.random.normal(size=[n_pt_new,n_samp]) )

    #estimate aleatory terms
    if not (eqid_new is None):
        #estimate between event terms based on earthquake ids
        dB_mu, _, dB_cov = pygp.SampledBe(eqid_new, eqid_train, dB_train, tau0)
        aleat_cov = phi0**2 * np.eye(n_pt_new) + dB_cov;
    else:
        dB_mu = np.zeros(n_pt_new)
        aleat_cov = (phi0**2 + tau0**2) * np.eye(n_pt_new)
        
    aleat_sig = np.sqrt(np.diag(aleat_cov))
        
    return(y_new, dB_mu, aleat_sig, aleat_cov)