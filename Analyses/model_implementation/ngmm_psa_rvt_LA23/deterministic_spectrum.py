#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:50:45 2023

@author: glavrent
"""

#load variables
import os
import pathlib
import glob
import re           #regular expression package
import pickle
#arithmetic libraries
import numpy as np
import numpy.matlib
#statistics libraries
import pandas as pd
#plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
#gmm libraries
import pygmm
import pyrvt
#user-derfined functions
import pylib.pylib_stats as pystats
import pylib.pylib_gmm_eas as py_gmm_eas
import pylib.pylib_gmm_psa as py_gmm_psa

# %% Input Variables
# ======================================
# input files 
fname_flatfile_easNGA = '../../../Data/model_implementation/NGA2West_FAS_CA.csv'
fname_cellinfo        = '../../../Data/model_implementation/atten_cells/NGA2WestCA_cellinfo.csv'
#inter-frequency correlation coefficients
fname_ifreqcorr       = '../../../Data/model_implementation/ifreq_corr/coeffs_ifreq_correlation.csv'
#stan regression results
dir_stan       = '../../../Data/model_implementation/ngmm_coeffs_eas/'
analysis_sufix = '_laten_var_unbound_hyp'



#stan sub-directories for different frequencies
freq_nerg_array = np.array([0.11220184, 0.12022643,                       0.15135611,           0.177828,  0.1995262,  0.2290868, 0.2398833, 0.2511886, 0.2754229, 0.3019952, 0.3311311, 0.3630781, 0.39810714, 0.4365158, 0.5011872, 0.5495409, 0.6025595, 0.69183093,             0.7585776, 0.79432821, 0.8912509, 1.000000,  
                            1.1220182,  1.2022641,             1.28825,   1.513561,   1.659587, 1.7782794, 1.9952621,  2.290868,  2.398833,  2.5118863, 2.7542283, 3.019952,  3.311311,  3.63078,   3.981071,   4.4668354, 5.0119,    5.495409,  6.025596,  6.9183082,  7.0794563,  7.585776,              8.912507,  10.00000,   
                            11.220183,  12.022642,  12.589251, 12.882492, 15.135614,  16.59587, 17.782793, 19.952621 ])

#define scenario realizations for sampling epistemic uncertainty
nsamp = 1000

#event info
eq_latlon   = np.array([37.88862,-122.1779]) 
mag_array   = np.array([5.0, 7.0])
ztor        = 3.0
sof         = 0
dip         = 90
width_array = np.minimum([np.sqrt( 1/1.5 * 10**(m-4) ) for m in mag_array], 15)
#site of interest
name_site     = 'BerkeleyCA'
vs30          = 400
site_latlon   = np.array([37.87,-122.26]) 
# station sequence number, if station available at the site, otherwise set to None
# site_ssn    = 100046
site_ssn      = None
#RVT options
sdur = 0.85     #significant duration
m_PF = 'BT15'   #ground motion duration

# ergodic PSA options
# ASK14
name_nerg_psa_gmm     = r'Non-erg PSA GMM$_1$'
name_erg_psa_gmm      = r'ASK14'
erg_psa_gmm           = pygmm.AbrahamsonSilvaKamai2014
fname_erg_psa_gmm_hyp = '../../../Data/model_implementation/ngmm_aleatvar_psa/ASK14BT15_durAS96_smoothed_aleat_hyp.csv'
# # CY14
# name_nerg_psa_gmm     = r'Non-erg PSA GMM$_2$'
# name_erg_psa_gmm      = r'CY14'
# erg_psa_gmm           = pygmm.ChiouYoungs2014
# fname_erg_psa_gmm_hyp = '../../../Data/model_implementation/ngmm_aleatvar_psa/CY14BT15_durAS96_smoothed_aleat_hyp.csv'

#include North/South small magnitude adjustment
flag_include_c0NS = False

#output directory
dir_out = '../../../Data/prediction_output/'
dir_fig = dir_out + 'figures/'

# %% Load Data
# ======================================
#flatfile and psa hyperparameters
df_flatfile_eas = pd.read_csv(fname_flatfile_easNGA)
df_cellinfo     = pd.read_csv(fname_cellinfo)
df_psa_hyp      = pd.read_csv(fname_erg_psa_gmm_hyp, index_col=0)

#%% Process Data
### ======================================
zone_utm  = np.unique(df_flatfile_eas.zoneUTM)[0]
#earthquake and station ids
eq_id_train  = df_flatfile_eas['eqid'].values.astype(int)
sta_id_train = df_flatfile_eas['ssn'].values.astype(int)
eq_id, eq_idx_inv   = np.unique(eq_id_train,  return_index=True)
sta_id, sta_idx_inv = np.unique(sta_id_train, return_index=True)
#cell ids
cell_id = np.array([int( re.search('c.(\\d+)', text).group(1) ) for text in df_cellinfo.cell_names])

#define gmm objects
BA18    = py_gmm_eas.BA18()
#set up non-ergodic EAS model
NergEas = py_gmm_eas.NonErgEASGMMCoeffCond(eqid = eq_id, ssn = sta_id, cA_id = cell_id ,zone_utm = zone_utm, 
                                           cA_Xutm = df_cellinfo[['q1UTMx','q1UTMy','q1UTMz','q8UTMx','q8UTMy','q8UTMz']].values,
                                           cAmpt_Xutm = df_cellinfo[['mptUTMx','mptUTMy']].values,
                                           flg_sparse_cov = True)

del eq_id_train, sta_id_train, eq_id, eq_idx_inv, sta_id, sta_idx_inv

#iterate over frequencies
for k, freq in enumerate(freq_nerg_array):
    print('Reading freq: %.2f (%i of %i)'%(freq, k+1, len(freq_nerg_array)))
    #define input file name
    GMMmodel = 'BA19_f%.2fhz_NGAWestCA' % freq
    fname_analysis = '%s%s' % (GMMmodel, analysis_sufix)
    
    # regression results
    fname_data = dir_stan + fname_analysis + '_data_all.h5'
    df_hyp = pd.read_hdf(fname_data,  key='hyp_param').loc['mean', :]
    df_c1a = pd.read_hdf(fname_data,  key='data_c1a')
    df_c1b = pd.read_hdf(fname_data,  key='data_c1b')
    df_dS2S = pd.read_hdf(fname_data, key='data_dS2S')
    df_cA = pd.read_hdf(fname_data,   key='data_cA')
    df_dBe = pd.read_hdf(fname_data,  key='data_dBe')
    
    #add non-ergodic frequency
    NergEas.AddFreq(freq, zone_utm, df_c1a[['utmX','utmY']].values, df_c1b[['utmX','utmY']].values, df_cA[['utmX','utmY']].values, 
                          df_dS2S.ssn.values, df_c1a.eqid.values, df_cA.cell_id.values,
                          df_c1a.c1a_mu.values, df_c1a.c1a_sig.values, df_c1b.c1b_mu.values, df_c1b.c1b_sig.values,
                          df_dS2S.dS2S_mu.values, df_dS2S.dS2S_sig.values, df_dBe.dBe_mu.values, df_dBe.dBe_sig.values, 
                          df_cA.cA_mu.values, df_cA.cA_sig.values,
                          df_hyp.c0, df_hyp.c0N, df_hyp.c0S, 
                          df_hyp.c1a_rho, df_hyp.c1a_theta, df_hyp.c1b_rho, df_hyp.c1b_theta, 
                          df_hyp.cA_mu, df_hyp.cA_rho, df_hyp.cA_theta, df_hyp.cA_sigma, df_hyp.cA_pi,
                          df_hyp.phi_S2S, df_hyp.tau_0, df_hyp.phi_0)

    del df_hyp, df_c1a, df_c1b, df_dS2S, df_cA, df_dBe

#interfrequency correlation
df_ifcorr = pd.read_csv(fname_ifreqcorr,  index_col=0)

NergEas.AddInterFreqCorrNCoeff(df_ifcorr.loc['c1a','A'],  df_ifcorr.loc['c1a','B'],  df_ifcorr.loc['c1a','C'],  df_ifcorr.loc['c1a','D'],
                               df_ifcorr.loc['c1b','A'],  df_ifcorr.loc['c1b','B'],  df_ifcorr.loc['c1b','C'],  df_ifcorr.loc['c1b','D'],
                               df_ifcorr.loc['dS2S','A'], df_ifcorr.loc['dS2S','B'], df_ifcorr.loc['dS2S','C'], df_ifcorr.loc['dS2S','D'],
                               df_ifcorr.loc['cA','A'],   df_ifcorr.loc['cA','B'],   df_ifcorr.loc['cA','C'],   df_ifcorr.loc['cA','D'] )

del df_ifcorr

#set up non-ergodic PSA model
NergPSa = py_gmm_psa.NonErgPSAGMM(df_psa_hyp.per, df_psa_hyp.c_0,
                                  df_psa_hyp.phi_0m1, df_psa_hyp.phi_0m2, df_psa_hyp.tau_0m1, df_psa_hyp.tau_0m2, 
                                  mag_brk_phi=np.unique(df_psa_hyp[['m1_phi','m2_phi']]), mag_brk_tau=np.unique(df_psa_hyp[['m1_tau','m2_tau']]),
                                  ErgEASGMM=BA18, NonErgEASGMM=NergEas, ErgPSAGMM=erg_psa_gmm)

#%% Main Calculations
### ======================================
#eq/site UTM
eq_X  = NergEas.ProjLatLon2UTM(eq_latlon)
sta_X = NergEas.ProjLatLon2UTM(site_latlon)

#rupture distance
rrup = np.sqrt(np.linalg.norm(eq_X - sta_X)**2 + ztor**2)

## Define Scenaria
## --------------------------------
#number of different mags
n_mags = len(mag_array)

#earthquake and site coordinates
eq_latlon = np.matlib.repmat(eq_latlon,   n_mags, 1)
eq_z      = np.full(n_mags, -ztor)
st_latlon = np.matlib.repmat(site_latlon, n_mags, 1)
eq_X      = np.matlib.repmat(eq_X,  n_mags, 1)
st_X      = np.matlib.repmat(sta_X, n_mags, 1)

#site information
ssn_array = np.full(n_mags, site_ssn) if not site_ssn is None else site_ssn
reg_id = 1

#source parameters
ztor_array = np.full(len(mag_array), ztor)
sof_array  = np.full(len(mag_array), sof)
dip_array  = np.full(len(mag_array), dip)
#path parameters
rrup_array  = np.sqrt(np.linalg.norm(eq_X - st_X, axis=1)**2 + ztor**2)
#site parameters
vs30_array  = np.full(len(mag_array), vs30)
z1_array    = np.full(len(mag_array), BA18.Z1(vs30))

## Evaluate Scenaria
## --------------------------------
# Compute ergodic and non-ergodic EAS and PSa (with inter-frequency correlation)
# ---   ---   ---   ---   ---   ---
per_psa, psa_nerg_corrfreq, _, psa_erg, rpsa_nerg_corrfreq, freq_eas, eas_nerg_corrfreq, eas_erg, reas_nerg_corrfreq = NergPSa.CalcNonErgPSa(eq_latlon, st_latlon, eqclst_latlon=eq_latlon, eqclst_z=eq_z,
                                                                                                                                             mag=mag_array, dip=dip_array, sof=sof_array, width=width_array, ztor=ztor_array, 
                                                                                                                                             rrup=rrup_array, rrjb=rrup_array, rx=None, ry0=None,
                                                                                                                                             vs30=vs30_array, z10=z1_array, z25=None, regid = reg_id,
                                                                                                                                             flag_as=None, flag_hw=None, crjb=None,
                                                                                                                                             ssn = ssn_array, 
                                                                                                                                             nsamp = nsamp, flag_samp_unqloc = True, flag_ifreq = True,
                                                                                                                                             flag_include_c0NS = flag_include_c0NS, 
                                                                                                                                             PF = m_PF, sdur = sdur)

# Mean of PSa and EAS
psa_nerg_corrfreq_mean  = np.exp( np.mean(np.log(psa_nerg_corrfreq), axis=2) )
eas_nerg_corrfreq_mean  = np.exp( np.mean(np.log(eas_nerg_corrfreq), axis=2) )
rpsa_nerg_corrfreq_mean = np.mean(rpsa_nerg_corrfreq, axis=2)
reas_nerg_corrfreq_mean = np.mean(reas_nerg_corrfreq, axis=2)
# Std of PSa and EAS
psa_nerg_corrfreq_std  = np.std(np.log(psa_nerg_corrfreq), axis=2)
eas_nerg_corrfreq_std  = np.std(np.log(eas_nerg_corrfreq), axis=2)
rpsa_nerg_corrfreq_std = np.std(rpsa_nerg_corrfreq, axis=2)
reas_nerg_corrfreq_std = np.std(reas_nerg_corrfreq, axis=2)

#%% Plotting
### ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

# Compare non-ergodic ratios
# ---  ---  ---  ---
#mean eas spectra for all magnitudes
fname = '%s_eas_mag_all'%(name_site)
fig, ax = plt.subplots(figsize = (10,8))
for k, (mag, e_erg, e_nerg_m) in enumerate(zip(mag_array, eas_erg, eas_nerg_corrfreq_mean)):
    pl1 = ax.loglog(freq_eas, e_erg,     '--', linewidth=2.0,                          label='BA18 ($M$=%.1f)'%mag)
    pl2 = ax.loglog(freq_eas, e_nerg_m, '-',  linewidth=2.0, color=pl1[0].get_color(), label='Non-erg EAS ($M$=%.1f)'%mag)
#edit figures
ax.set_xlabel('Frequency ($Hz$)', fontsize=30)
ax.set_ylabel('$EAS$ ($g~sec$)',  fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.1,100])
ax.legend(fontsize=25, loc='lower left')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
ax.set_title('%s'%(name_site), fontsize=22)
fig.tight_layout()
fig.savefig(dir_fig + fname + '.png')
ylim_eas = ax.get_ylim()

#mean psa spectra for all magnitudes
fname = '%s_psa_mag_all'%(name_site)
fig, ax = plt.subplots(figsize = (10,8))
for k, (mag, p_erg, p_nerg_m) in enumerate(zip(mag_array, psa_erg, psa_nerg_corrfreq_mean)):
    pl1 = ax.loglog(per_psa, p_erg,     '--', linewidth=2.0,                           label='%s ($M$=%.1f)'%(name_erg_psa_gmm,  mag))
    pl2 = ax.loglog(per_psa, p_nerg_m,  '-',  linewidth=2.0, color=pl1[0].get_color(), label='%s ($M$=%.1f)'%(name_nerg_psa_gmm, mag))
#edit figures
ax.set_xlabel('Period ($sec$)', fontsize=30)
ax.set_ylabel('PSA ($g$)',      fontsize=30)
ax.grid(which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.01,10])
ax.legend(fontsize=25, loc='lower left')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
ax.set_title('%s'%(name_site), fontsize=22)
fig.tight_layout()
fig.savefig(dir_fig + fname + '.png')
ylim_psa = ax.get_ylim()

#plot individual EAS spectra
# ---  ---  ---  ---
for k, (mag, e_erg, e_nerg_cf, e_nerg_cf_m, e_nerg_cf_s) in enumerate(zip(mag_array, eas_erg, eas_nerg_corrfreq, eas_nerg_corrfreq_mean, eas_nerg_corrfreq_std)):
    #eas spectra with inter-frequency correlation
    fname = '%s_eas_mag_%.1f'%(name_site, mag)
    fig, ax = plt.subplots(figsize = (10,8))
    pl1 = ax.loglog(freq_eas, e_erg,         '--', linewidth=2.0, color='k', label='BA18')
    pl2 = ax.loglog(freq_eas, e_nerg_cf_m,   '-',  linewidth=2.0, color='k', label='Non-erg EAS ($\mu$)')
    #epistemic uncertainty
    pl3 = ax.fill_between(freq_eas, y1=e_nerg_cf_m*np.exp(-e_nerg_cf_s), y2=e_nerg_cf_m*np.exp(e_nerg_cf_s), color='gray', alpha=0.2, label='Non-erg EAS ($\mu\pm\psi$)')
    #random sample
    pl4 = ax.loglog(freq_eas, e_nerg_cf[:,1], '-',  linewidth=1.2,            label='Non-erg EAS (random sample)')
    #edit figures
    ax.set_xlabel('Frequency ($Hz$)', fontsize=30)
    ax.set_ylabel('EAS ($g~sec$)',    fontsize=30)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.1,100])
    ax.set_ylim(ylim_eas)
    ax.legend(fontsize=25, loc='lower left')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
    ax.set_title('%s Mag: %.1f'%(name_site, mag), fontsize=22)
    fig.tight_layout()
    fig.savefig(dir_fig + fname + '.png')

#plot individual PSA spectra
# ---  ---  ---  ---
for k, (mag, p_erg, p_nerg_cf, p_nerg_cf_m, p_nerg_cf_s) in enumerate(zip(mag_array, psa_erg, psa_nerg_corrfreq, psa_nerg_corrfreq_mean, psa_nerg_corrfreq_std)):
    #psa spectra with inter-frequency correlation
    fname = '%s_psa_mag_%.1f'%(name_site, mag)
    fig, ax = plt.subplots(figsize = (10,8))
    pl1 = ax.loglog(per_psa, p_erg,       '--', linewidth=2.0, color='k', label='%s'%name_erg_psa_gmm)
    pl2 = ax.loglog(per_psa, p_nerg_cf_m, '-',  linewidth=2.0, color='k', label='%s ($\mu$)'%name_nerg_psa_gmm)
    #epistemic uncertainty 
    pl3 = ax.fill_between(per_psa, y1=p_nerg_cf_m*np.exp(-p_nerg_cf_s), y2=p_nerg_cf_m*np.exp(p_nerg_cf_s), color='gray', alpha=0.2, label='Non-erg PSA GMM$_1$ ($\mu\pm\psi$)')
    #random sample
    pl4 = ax.loglog(per_psa, p_nerg_cf[:,1], '-',  linewidth=1.5,         label='Non-erg PSA GMM$_1$ (random sample)', zorder=0)
    #edit figures
    ax.set_xlabel('Period ($sec$)', fontsize=30)
    ax.set_ylabel('PSA ($g$)', fontsize=30)
    ax.set_ylim(ylim_psa)
    ax.grid(which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.01,10])
    ax.legend(fontsize=25, loc='lower left')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7,  width=2, direction='in', right='on')
    ax.set_title('%s Mag: %.1f'%(name_site, mag), fontsize=22)
    fig.tight_layout()
    fig.savefig(dir_fig + fname + '.png')

