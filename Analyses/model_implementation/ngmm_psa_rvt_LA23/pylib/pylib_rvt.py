#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:03:50 2020

@author: glavrent
"""
#arithmetic libraries
import numpy as np
import numpy.matlib
#geographic coordinates
import pyproj
import geopy.distance as geopydist
#gmm libraries
import pygmm
import pyrvt
#user libraries
import pylib.pylib_extrapfas as pylib_extrapfas

def PSDOFImpRespFreqD(freqs, osc_freq, osc_damping):
    '''
    SDOF Impulse response

    Parameters
    ----------
    freqs : np.array
        Frequency.
    osc_freq : real
        Oscillator's natural frequency.
    osc_damping : np.array
        Oscillator's damping.

    Returns
    -------
    IR : np.array
        Impulse response.
    IR_amp : np.array
        Amplitude of impulse response.

    '''
    
    #Pseudo spectral acceleration impulse response
    IR =  -osc_freq ** 2. / (freqs ** 2 - osc_freq ** 2 - 2.j * osc_damping * osc_freq * freqs)
    #amplitude spectrum of SDOF impulse response
    IR_amp = abs(IR)
    
    return IR, IR_amp

def CalcDuration(corner_freq, hypo_distance, region = 'wna'):
    """
    Compute the duration by combination of source and path.

    Parameters
    ----------
    corner_freq : float
        Corner frequency.
    hypo_distance : flot
        hypo-center distance.
    region : string, optional
        Region for the attenuation. The default is = 'wna'.

    Returns
    -------
    duration : float
        Computed duration

    """

    # Source component
    duration_source = 1. / corner_freq
    
    # Path component
    if region == 'wna':
        duration_path = 0.05 * hypo_distance
    elif region == 'cena':
        duration_path = 0.
        if hypo_distance > 10:
            # 10 < R <= 70 km
            duration_path +=  0.16 * (min(hypo_distance, 70) - 10.)
        if hypo_distance > 70:
            # 70 < R <= 130 km
            duration_path += -0.03 * (min(hypo_distance, 130) - 70.)
        if hypo_distance > 130:
            # 130 km < R
            duration_path +=  0.04 * (hypo_distance - 130.)
    else:
        raise NotImplementedError
    
    return duration_source + duration_path

def CalcPSaRVTSingScen(freq_rvt, freq, fas, mag, rrup, vs30, dur = None, dur_interval = 0.85,
                       freq_int = None, m_PF = 'BT15'):
    '''
    Compute PSA for FAS scenario using RVT

    Parameters
    ----------
    freq_rvt : np.array(n_f_rvt)
        Frequencies for PSa RVT.
    freq : np.array(n_f)
        Frequencies of FAS array.
    fas : np.array(n_f)
        Amplitudes of FAS array.
    mag : real
        Earthquake magnitude.
    rrup : real
        Rupture distance.
    vs30 : real
        Vs30.
    dur : real, optional
        Ground-motion duration. If set to None, dur_interval is used.
    dur_interval : real, optional
        Significant duration interval. The default is 0.80.
    freq_int : np.array(n_f_int), optional
        Frequencies for interpolated FAS. The default is None.
    m_PF : real, optional
        Peak factor. The default is 'BT15'.

    Returns
    -------
    freq_rvt : np.array([n_f_rvt])
        Frequencies for PSa RVT
    psa_rvt : np.array([n_f_rvt])
        Amplitudes of PSa RVT.
    freq_int : np.array([n_f_int])
        Frequencies of interpolated FAS.
    fas_int : np.array([n_f_int])
        Amplitudes of interpolated FAS.

    '''
    
    #RVT information
    if freq_int is None: freq_int = np.logspace(np.log10(.01), np.log10(100), 300)

    #soil conditions
    scond = 'soil' if vs30 < 500 else 'rock'
    #define scenario GMM
    gmm_scen = pygmm.model.Scenario(mag = mag, dist_jb = rrup, dist_rup = rrup, dist_x = 0, site_cond = scond)
    #duration GMM
    gm_dur = dur if dur else pygmm.AbrahamsonSilva1996(gmm_scen).interp( dur_interval )
    #extend Fourier spectrum
    fc = pylib_extrapfas.CalcFc(mag)
    #ergodic interpolated eas
    _, fas_int = pylib_extrapfas.ExtendFAStoLF(freq,     fas,     freq_int, f_c=fc,    f_bin_ratio=1.05)
    _, fas_int = pylib_extrapfas.ExtendFAStoHF(freq_int, fas_int, freq_int, vs30=vs30, f_bin_ratio = 0.95)
    #rvt peak factor
    rvt_event_kwds = {'mag': mag, 'dist': max(rrup,2), 'region': 'WNA'}
    peak_calc = pyrvt.peak_calculators.get_peak_calculator(m_PF, rvt_event_kwds)
    #rvt objects
    rvtm = pyrvt.motions.RvtMotion(freqs=freq_int, fourier_amps=fas_int, 
                                   calc_kwds=rvt_event_kwds, duration=gm_dur,
                                   peak_calculator = peak_calc)
    #compute spectral accelerations with RVT
    psa_rvt = rvtm.calc_osc_accels(osc_freqs = freq_rvt)
    
    return freq_rvt, psa_rvt, freq_int, fas_int

def CalcPSaRVTMultScen(freq_rvt, freq, fas, mag_array, rrup_array, vs30_array, dur_array = None, dur_interval = 0.85,
                       freq_int = None, m_PF = 'BT15'):
    '''
    

    Parameters
    ----------
    freq_rvt : np.array([n_f_rvt])
        Frequencies for PSa RVT.
    freq : np.array([n_f])
        Frequencies of FAS array.
    fas : np.array([n_pt, n_f])
        Amplitudes of FAS array.
    mag_array : np.array([n_pt])
        Earthquake magnitude array.
    rrup_array : np.array([n_pt])
        Rupture distance array.
    vs30_array : np.array([n_pt])
        Vs30 array.
    dur_array : np.array([n_pt]), optional
        Ground-motion duration array. The default is None.
    dur_interval : real, optional
        Significant duration interval. The default is 0.80.
    freq_int : np.array(n_f_int), optional
        Frequencies for interpolated FAS. The default is None.
    m_PF : real, optional
        Peak factor. The default is 'BT15'.
        
    Returns
    -------
    freq_rvt : np.array([n_f_rvt])
        Frequencies for PSa RVT
    psa_rvt : np.array([n_pt, n_f_rvt])
        Amplitudes of PSa RVT.
    freq_int : np.array([n_f_int])
        Frequencies of interpolated FAS.
    fas_int : np.array([n_pt, n_f_int])
        Amplitudes of interpolated FAS.

    '''
    
    #RVT information
    if freq_int is None: freq_int = np.logspace(np.log10(.01), np.log10(100), 300)

    #convert gmm parameters to numpy 
    mag_array  = np.array([mag_array]).flatten()
    rrup_array = np.array([rrup_array]).flatten()
    vs30_array = np.array([vs30_array]).flatten()
    dur_array  = None if dur_array is None else np.array([dur_array]).flatten()

    #convert fas to two dim array 
    if fas.ndim == 1: fas = fas.reshape(1,len(fas))

    #initialize eas and psa rvt matrices
    n_pt = len(mag_array)
    fas_int = np.full([n_pt, len(freq_int)], np.nan)
    psa_rvt = np.full([n_pt, len(freq_rvt)], np.nan)
    
    #iterate over grid points
    for k, (mag, rrup, vs30),  in enumerate(zip(mag_array, rrup_array, vs30_array)):
        #soil conditions
        scond = 'soil' if vs30 < 500 else 'rock'
        #define scenario GMM
        gmm_scen = pygmm.model.Scenario(mag = mag, dist_jb = rrup, dist_rup = rrup, dist_x = 0, site_cond = scond)
        #duration GMM
        gm_dur = pygmm.AbrahamsonSilva1996(gmm_scen).interp( dur_interval ) if dur_array is None else dur_array[k] 
        #extend Fourier spectrum
        fc = pylib_extrapfas.CalcFc(mag)
        #ergodic interpolated eas
        _, fas_int[k,:] = pylib_extrapfas.ExtendFAStoLF(freq,     fas[k,:],     freq_int, f_c=fc,    f_bin_ratio=1.005)
        # _, fas_int[k,:] = pylib_extrapfas.ExtendFAStoLF(freq,     fas[k,:],     freq_int, f_c=fc,    f_bin_ratio=1.05)
        _, fas_int[k,:] = pylib_extrapfas.ExtendFAStoHF(freq_int, fas_int[k,:], freq_int, vs30=vs30, f_bin_ratio = 0.95)
        #rvt peak factor
        rvt_event_kwds = {'mag': mag, 'dist': max(rrup,2), 'region': 'WNA'}
        peak_calc = pyrvt.peak_calculators.get_peak_calculator(m_PF,rvt_event_kwds)
        #rvt objects
        rvtm = pyrvt.motions.RvtMotion(freqs=freq_int, fourier_amps=fas_int[k,:], 
                                       calc_kwds=rvt_event_kwds, duration=gm_dur,
                                       peak_calculator = peak_calc)
        #compute spectral accelerations with RVT
        psa_rvt[k,:]  = rvtm.calc_osc_accels(osc_freqs = freq_rvt)
    
    return freq_rvt, psa_rvt, freq_int, fas_int, rrup_array

def CalcPSaSingleScenGrid(freq_rvt, eq_scen, freq, eas, freq_int = None, dur_array = None, dur_interval = 0.85, m_PF = 'BT15',
                                    rrup_array=None, eq_latlon=None, sta_latlon=None, utm_zone=None):
    
    #RVT information
    if freq_int is None: freq_int = np.logspace(np.log10(.01), np.log10(100), 300)

    #compute distance
    if not eq_latlon is None:
        assert(not sta_latlon is None),'Error. Undefined station coordinates'
        
        #compute rupture distance if not defined
        if rrup_array is None:
            if (not utm_zone is None):
                #utm projection object
                proj_utm = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
                #compute earthquake and station utm coordinates
                eq_X  = np.array([proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(eq_latlon[:,0],  eq_latlon[:,1])])  / 1000
                sta_X = np.array([proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(sta_latlon[:,0], sta_latlon[:,1])]) / 1000
                #rupture distance without depth
                rrup_array  = np.linalg.norm(eq_X - sta_X, axis=1)
            else:
                #rupture distance without depth
                rrup_array = np.array([geopydist.distance(sta_latlon[1,0], eq_ll[1,0]).km for eq_ll in eq_latlon])
    
            #add rupture depth on rupture distance
            rrup_array = np.sqrt(rrup_array**2 + eq_scen['ztor']**2)

    #initialize eas and psa rvt matrices
    n_pt = eas.shape[0] if eas.ndim > 1 else 1
    eas_int = np.full([n_pt, len(freq_int)], np.nan)
    psa_rvt = np.full([n_pt, len(freq_rvt)], np.nan)
    
    #iterate over grid points
    for k, rrup in enumerate(rrup_array):
        #define scenario GMM
        gmm_scen = pygmm.model.Scenario(mag = eq_scen['mag'], dist_jb = rrup, dist_rup = rrup, dist_x = 0,
                                        mechanism = eq_scen['mech'], dip = eq_scen['dip'], depth_tor = eq_scen['ztor'],
                                        v_s30 = eq_scen['vs30'], site_cond = eq_scen['scond'])
        #duration GMM
        gm_dur = dur_array[k] if dur_array else pygmm.AbrahamsonSilva1996(gmm_scen).interp( dur_interval ) 
        #extend Fourier spectrum
        fc = pylib_extrapfas.CalcFc(eq_scen['mag'])
        #ergodic interpolated eas
        _, eas_int[k,:] = pylib_extrapfas.ExtendFAStoLF(freq,     eas[k,:],     freq_int, fc, f_bin_ratio=1.005)
        # _, eas_int[k,:] = pylib_extrapfas.ExtendFAStoLF(freq,     eas[k,:],     freq_int, fc, f_bin_ratio=1.05)
        _, eas_int[k,:] = pylib_extrapfas.ExtendFAStoHF(freq_int, eas_int[k,:], freq_int, vs30 = eq_scen['vs30'], f_bin_ratio = 0.95)
        #rvt peak factor
        rvt_event_kwds = {'mag': eq_scen['mag'], 'dist': rrup, 'region': 'WNA'}
        peak_calc = pyrvt.peak_calculators.get_peak_calculator(m_PF,rvt_event_kwds)
        #rvt objects
        rvtm = pyrvt.motions.RvtMotion(freqs=freq_int, fourier_amps=eas_int[k,:], 
                                       calc_kwds=rvt_event_kwds, duration=gm_dur,
                                       peak_calculator = peak_calc)
        #compute spectral accelerations with RVT
        psa_rvt[k,:]  = rvtm.calc_osc_accels(osc_freqs = freq_rvt)
    
    
    return freq_rvt, psa_rvt, freq_int, eas_int, rrup_array




