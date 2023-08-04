#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:51:01 2020

@author: glavrent
"""

#load libraries
#arithmetic libraries
import numpy as np

def CalcFcSimple(mag):
    '''Compute the corner frequency based on the fault dimensions'''
    
    rup_A = 10**(mag - 4)    #rupture area
    rup_w = max(np.sqrt(rup_A), 15)        #rupture width
    rup_l = rup_A / rup_w               #rupture length

    #corner frequency
    f_c = 2.4 / (0.5 * rup_l)
    return f_c

def CalcFc(mag, deltasig=0):
    '''Compute the corner frequency based on stress drop and seismic moment'''
    
    
    #stress drop
    if not deltasig:
        deltasig = 10 ** (3.45 - 0.2 * max(7, 5.))
    #seismic moment
    M0 = 10**(1.5*mag + 16.05)
    #shear wave-velocity
    beta0 = 3.5

    f_c = 4.9 * 10**6 * beta0 * (deltasig/M0)**(1./3.)
    return f_c
    
    
def ExtendFAStoLF(freq, fas, freq_q, f_c, f_bin_ratio = 1.2):
    
    assert(np.all(freq >= 0)),'freq must only contain positive frequencies'
    
    def Omega2Srouce(freq, f_c):
        '''Return the omega squared Fourier amplitude spectrum'''
        o2s = freq**2 / ( 1 + freq**2/f_c**2 )
        #o2s = (freq**2/f_c**2) / ( 1 + freq**2/f_c**2 )
        return o2s
    
    #minimum frequency
    #import pdb; pdb.set_trace()
    f_min = np.min(freq[~np.isnan(fas)])
    
    #frequency bin for estimating omega2 amplitude
    f_bin_lims = f_min * np.array([1., f_bin_ratio])
    i_f_bin = np.logical_and(freq >= f_bin_lims[0], freq <= f_bin_lims[1])
    #frequencies and amplitudes for estimating omega2 amplitude
    freq_bin = freq[i_f_bin]
    fas_bin = fas[i_f_bin]
    
    #spectral shape of omega2 source model
    omega2 = Omega2Srouce(freq_bin, f_c)
    #amplitude of omega2 model
    #omega2_amp =  np.exp( np.log(fas_bin).mean() / np.log(omega2).mean() )
    omega2_amp =  np.mean( fas_bin / omega2 )
     
    #interpolated fas at queried frequencies
    fas_q = np.exp(np.interp(np.log(np.abs(freq_q)), np.log(freq), np.log(fas), left=-np.nan, right=-np.nan))

    #indices fas is extrapolated with the omega2 model
    i_f_low  = freq_q < f_min
    #extrapolated ampltudes
    fas_q[i_f_low] = omega2_amp * Omega2Srouce(freq_q[i_f_low], f_c)
    
    return(freq_q, fas_q)

def ExtendFAStoHF(freq, fas, freq_q, kappa = None, vs30 = None, f_bin_ratio = 0.8):
    """
    

    Parameters
    ----------
    freq : np.array(n_f)
        frequency array for fas spectrum.
    fas : np.array(n_f)
        amplitude array for fas spectrum.
    freq_q : np.array(n_f2)
        query frequencies for .
    kappa : real, optional
        kappa for shallow crustal amplification. The default is None.
    vs30 : real, optional
        Vs30 value. The default is None.
    f_bin_ratio : real, optional
        ratio for high frequency bin to connect the kappa model. The default is 0.8.

    Returns
    -------
    freq_q : np.array
        frequency array of extended Fourier spectrum.
    fas_q : np.array(n_f2)
        amplitudes of extended Fourier amplitude spectrum.

    """
    
    assert(np.all(freq >= 0)),'freq must only contain positive frequencies'
    
    #estimate kappa if not provided
    if not vs30 is None: #estimate kappa from vs30
        assert(kappa is None),'User provided both values for kappa and Vs30'
        kappa = np.exp( -0.4 * np.log(vs30/760) - 3.5 )
    elif kappa is None: #default kappa value
        import pdb; pdb.set_trace()
        kappa = 0.04
        
    
    def ShallowCrustAmp(freq, kappa):
        '''Shallow crustal amplification based on Kappa'''
        scrust = np.exp(-kappa * np.pi * freq)
        return scrust
    
    #interpolated fas at queried frequencies
    fas_q = np.exp(np.interp(np.log(np.abs(freq_q)), np.log(freq), np.log(fas), left=-np.nan, right=-np.nan))
    
    #maximum frequency
    f_max = np.max(freq[~np.isnan(fas)])

    #indices fas is extrapolated with the omega2 model
    i_f_max = freq_q > f_max
    
    #extend spectrum with kappa model if necessary
    if i_f_max.any():
        #frequency bin for estimating omega2 amplitude
        f_bin_lims = f_max * np.array([f_bin_ratio, 1.])
        i_f_bin = np.logical_and(freq >= f_bin_lims[0], freq <= f_bin_lims[1])
        #frequencies and amplitudes for estimating omega2 amplitude
        freq_bin = freq[i_f_bin]
        fas_bin = fas[i_f_bin]    
    
        #spectral shape of crustal amplification
        scrust = ShallowCrustAmp(freq_bin, kappa)
        if (scrust < 1e-3).any(): import pdb; pdb.set_trace()
        #amplitude of omega2 model
        scrust_amp =  np.mean( fas_bin / scrust )
    
        #extrapolated ampltudes
        fas_q[i_f_max] = scrust_amp * ShallowCrustAmp(freq_q[i_f_max], kappa)
    
    return(freq_q, fas_q)


     