#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:35:49 2023

@author: glavrent
"""
# load packages
import os
import re  # regular expression package
import pathlib
import glob
# arithmetic libraries
import numpy as np
import numpy.matlib
# geometry libraries
# statistics libraries
import pandas as pd
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
# plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
# gmm libraries
import pygmm
# user-derfined functions
import pylib.pylib_contour_plots as pycplt
import pylib.pylib_gmm_psa as py_gmm_psa
import pylib.pylib_gmm_eas as py_gmm_eas

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

# non-ergodic frequencies
freq_nerg_array = np.array([0.11220184, 0.12022643,                       0.15135611,           0.177828,  0.1995262,  0.2290868, 0.2398833, 0.2511886, 0.2754229, 0.3019952, 0.3311311, 0.3630781, 0.39810714, 0.4365158, 0.5011872, 0.5495409, 0.6025595, 0.69183093,             0.7585776, 0.79432821, 0.8912509, 1.000000,
                            1.1220182,  1.2022641,             1.28825,   1.513561,   1.659587, 1.7782794, 1.9952621,  2.290868,  2.398833,  2.5118863, 2.7542283, 3.019952,  3.311311,  3.63078,   3.981071,   4.4668354, 5.0119,    5.495409,  6.025596,  6.9183082,  7.0794563,  7.585776,              8.912507,  10.00000,
                            11.220183,  12.022642,  12.589251, 12.882492, 15.135614,  16.59587, 17.782793, 19.952621])

# define scenario realizations for PSHA input
nsamp = 100

# earthquake info
mag  = np.array([5.0, 6.0, 7.0, 8.0])
ztor = 3.0
sof  = 0
vs30 = 400
dip  = 90
width = np.minimum([np.sqrt(1/1.5 * 10**(m-4)) for m in mag], 15)
# site location
site_latlon = np.array([37.8762, -122.2356])
# station sequence number, if station available at the site, otherwise set to -1
site_ssn    = 100046

# grid discretization for non-ergodic adjustments
grid_dlatdlon = [0.1, 0.1]
grid_size     = [1.8,1.8]
grid_win      = np.array([[-grid_size[0], -grid_size[0]], [grid_size[1], grid_size[1]]]) + site_latlon

# prediction domain
polyg_latlon = np.array([[42.0, -114.0], [31.3, -114.0], [31.3, -117.0],
                         [33.2, -119.6], [34.7, -121.1], [37.7, -123.1],
                         [40.3, -124.5], [42.0, -124.5], ])

# frequencies to compute psa non-ergodic ratios
per4psha = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, ])

# RVT options
sdur = 0.85
m_PF = 'BT15'

# ergodic PSA options
erg_psa_gmm1 = pygmm.AbrahamsonSilvaKamai2014
name_erg_psa_gmm1 = r'ASK14'
fname_erg_psa_gmm1_hyp = '../../../Data/model_implementation/ngmm_aleatvar_psa/ASK14BT15_durAS96_smoothed_aleat_hyp.csv'
erg_psa_gmm2 = pygmm.ChiouYoungs2014
name_erg_psa_gmm2 = r'CY14'
fname_erg_psa_gmm2_hyp = '../../../Data/model_implementation/ngmm_aleatvar_psa/CY14BT15_durAS96_smoothed_aleat_hyp.csv'

# haz info
name_haz = 'nerg_GMM1-GMM2_Berkeley_SS'
nerg_jcalc0 = -16000
# jcalc_mean = [2791, 2798] #measured Vs30
jcalc_mean = [2787, 2797] #inferred Vs30
jcalc_sig  = [15508, 15509]

# output directory
dir_out       = '../../../Data/prediction_output/'

# %% Load Data
# ======================================
df_flatfile_easNGA = pd.read_csv(fname_flatfile_easNGA)
df_cellinfo = pd.read_csv(fname_cellinfo)
df_psa_gmm1_hyp = pd.read_csv(fname_erg_psa_gmm1_hyp, index_col=0)
df_psa_gmm2_hyp = pd.read_csv(fname_erg_psa_gmm2_hyp, index_col=0)

# utm zone
zone_utm = re.sub('[a-zA-Z]','',np.unique(df_flatfile_easNGA.zoneUTM)[0])
# earthquake and station ids
eq_id_train  = df_flatfile_easNGA['eqid'].values.astype(int)
sta_id_train = df_flatfile_easNGA['ssn'].values.astype(int)
eq_id, eq_idx_inv   = np.unique(eq_id_train,  return_index=True)
sta_id, sta_idx_inv = np.unique(sta_id_train, return_index=True)
# cell ids
cell_id = np.array([int(re.search('c.(\\d+)', text).group(1))
                   for text in df_cellinfo.cell_names])

# define gmm objects
BA18 = py_gmm_eas.BA18()
# set up non-ergodic EAS model
NergEas = py_gmm_eas.NonErgEASGMMCoeffCond(eqid=eq_id, ssn=sta_id, cA_id=cell_id, zone_utm=zone_utm,
                                           cA_Xutm=df_cellinfo[[
                                               'q1UTMx', 'q1UTMy', 'q1UTMz', 'q8UTMx', 'q8UTMy', 'q8UTMz']].values,
                                           cAmpt_Xutm=df_cellinfo[[
                                               'mptUTMx', 'mptUTMy']].values,
                                           flg_sparse_cov=True)

del eq_id_train, sta_id_train, eq_id, eq_idx_inv, sta_id, sta_idx_inv

# load non-ergodic eas frequencies
for k, freq in enumerate(freq_nerg_array):
    print('Reading freq: %.2f (%i of %i)' % (freq, k+1, len(freq_nerg_array)))
    # define input file name
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

    # add non-ergodic frequency
    NergEas.AddFreq(freq, zone_utm, df_c1a[['utmX', 'utmY']].values, df_c1b[['utmX', 'utmY']].values, df_cA[['utmX', 'utmY']].values,
                          df_dS2S.ssn.values, df_c1a.eqid.values, df_cA.cell_id.values,
                          df_c1a.c1a_mu.values, df_c1a.c1a_sig.values, df_c1b.c1b_mu.values, df_c1b.c1b_sig.values,
                          df_dS2S.dS2S_mu.values, df_dS2S.dS2S_sig.values, df_dBe.dBe_mu.values, df_dBe.dBe_sig.values,
                          df_cA.cA_mu.values, df_cA.cA_sig.values,
                          df_hyp.c0, df_hyp.c0N, df_hyp.c0S,
                          df_hyp.c1a_rho, df_hyp.c1a_theta, df_hyp.c1b_rho, df_hyp.c1b_theta,
                          df_hyp.cA_mu, df_hyp.cA_rho, df_hyp.cA_theta, df_hyp.cA_sigma, df_hyp.cA_pi,
                          df_hyp.phi_S2S, df_hyp.tau_0, df_hyp.phi_0)

    del df_hyp, df_c1a, df_c1b, df_dS2S, df_cA, df_dBe

# load interfrequency correlation
df_ifcorr = pd.read_csv(fname_ifreqcorr,  index_col=0)

NergEas.AddInterFreqCorrNCoeff(df_ifcorr.loc['c1a',  'A'], df_ifcorr.loc['c1a',  'B'], df_ifcorr.loc['c1a',  'C'], df_ifcorr.loc['c1a',  'D'],
                               df_ifcorr.loc['c1b',  'A'], df_ifcorr.loc['c1b',  'B'], df_ifcorr.loc['c1b',  'C'], df_ifcorr.loc['c1b',  'D'],
                               df_ifcorr.loc['dS2S', 'A'], df_ifcorr.loc['dS2S', 'B'], df_ifcorr.loc['dS2S', 'C'], df_ifcorr.loc['dS2S', 'D'],
                               df_ifcorr.loc['cA',   'A'], df_ifcorr.loc['cA',   'B'], df_ifcorr.loc['cA',   'C'], df_ifcorr.loc['cA',   'D'])

del df_ifcorr

# set up non-ergodic EAS model
n_psa_gmm = len(jcalc_mean)
#gmm1
NergPSa1 = py_gmm_psa.NonErgPSAGMM(df_psa_gmm1_hyp.per, df_psa_gmm1_hyp.c_0,
                                   df_psa_gmm1_hyp.phi_0m1, df_psa_gmm1_hyp.phi_0m2, df_psa_gmm1_hyp.tau_0m1, df_psa_gmm1_hyp.tau_0m2,
                                   mag_brk_phi=np.unique(df_psa_gmm1_hyp[['m1_phi', 'm2_phi']]), mag_brk_tau=np.unique(df_psa_gmm1_hyp[['m1_tau', 'm2_tau']]),
                                   ErgEASGMM=BA18, NonErgEASGMM=NergEas, ErgPSAGMM=erg_psa_gmm1)
#gmm2
NergPSa2 = py_gmm_psa.NonErgPSAGMM(df_psa_gmm2_hyp.per, df_psa_gmm2_hyp.c_0,
                                   df_psa_gmm2_hyp.phi_0m1, df_psa_gmm2_hyp.phi_0m2, df_psa_gmm2_hyp.tau_0m1, df_psa_gmm2_hyp.tau_0m2,
                                   mag_brk_phi=np.unique(df_psa_gmm2_hyp[['m1_phi', 'm2_phi']]), mag_brk_tau=np.unique(df_psa_gmm2_hyp[['m1_tau', 'm2_tau']]),
                                   ErgEASGMM=BA18, NonErgEASGMM=NergEas, ErgPSAGMM=erg_psa_gmm2)

# %% Process Data
# ======================================
# Coordinates
# --------------------------------
# Site coordinates
site_X = NergEas.ProjLatLon2UTM(site_latlon)

# Domain polygon
polyg_X = np.array([NergEas.ProjLatLon2UTM(pt_ll) for pt_ll in polyg_latlon])
shp_polyg_X = shp_poly(polyg_X)

# create grid coordinates
lat_array = np.arange(grid_win[0, 0], grid_win[1, 0], grid_dlatdlon[0])
lon_array = np.arange(grid_win[0, 1], grid_win[1, 1], grid_dlatdlon[1])
# lat_grid, lon_grid = np.meshgrid(lat_array, lon_array)
lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)
# create coordinate array with all grid nodes
grid_latlon = np.vstack([lat_grid.T.flatten(), lon_grid.T.flatten()]).T
grid_X      = np.array([NergEas.ProjLatLon2UTM(pt_ll) for pt_ll in grid_latlon])
#number of lat/lon coordinates
n_latlon = [len(lat_array), len(lon_array)]
del lat_array, lon_array, lat_grid, lon_grid

# Define Scenaria
# --------------------------------
# number grid points
n_grid = len(grid_X)
# number of different mags
n_mags = len(mag)

# earthquake and site coordinates
eq_latlon = np.matlib.repmat(grid_latlon, n_mags, 1)
eq_z = np.full(n_grid*n_mags, -ztor)
st_latlon = np.matlib.repmat(site_latlon, n_mags*n_grid, 1)
eq_X = np.matlib.repmat(grid_X, n_mags, 1)
st_X = np.matlib.repmat(site_X, n_mags*n_grid, 1)

# site information
ssn_array = np.full(n_grid*n_mags, site_ssn if not site_ssn is None else np.nan) 
reg_id = 1

# source parameters
mag_array = np.hstack([np.full(n_grid, m) for m in mag])
ztor_array = np.full(len(mag_array), ztor)
sof_array = np.full(len(mag_array), sof)
# path parameters
rrup_array = np.sqrt(np.linalg.norm(eq_X - st_X, axis=1)**2 + ztor**2)
# site parameters
vs30_array = np.full(len(mag_array), vs30)
z1_array = np.full(len(mag_array), BA18.Z1(vs30))

# Evaluate Scenarios
# --------------------------------
# Compute non-ergodic ratios
per_psa, rpsa_nerg, _, _, freq_eas, reas_nerg, _, _ = NergPSa1.CalcNonErgRatios(eq_latlon, st_latlon, eqclst_latlon=eq_latlon, eqclst_z=eq_z,
                                                                                mag=mag_array, sof=sof_array, ztor=ztor_array, rrup=rrup_array,
                                                                                vs30=vs30_array, z10=z1_array, regid=reg_id,
                                                                                ssn=ssn_array,
                                                                                nsamp=nsamp, flag_samp_unqloc=True, flag_ifreq=True,
                                                                                flag_include_c0NS=False,
                                                                                PF=m_PF, sdur=sdur, per=per4psha, flag_psa_const=False)

#aleatory standard devation
sig0_gmm1_array = NergPSa1.GetSig0(mag, per4psha)
sig0_gmm2_array = NergPSa2.GetSig0(mag, per4psha)

#constant
c0_gmm1_array = NergPSa1.GetC0(per4psha)
c0_gmm2_array = NergPSa2.GetC0(per4psha)

#weights
wt_array = np.ones(nsamp)/nsamp/n_psa_gmm

# %% Save data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

# Create sample file
# --------------------------------
# psa ratios
# initialize header samples
header_rpsa = ['x1', 'lon', 'lat'] + ['sim_%i'%(j+1) for j in range(nsamp)]
# iterate over different mag and periods
for k, m in enumerate(mag):
    i_m = mag_array == m
    for j, p in enumerate(per_psa):
        #file name
        fname_psha_samp = 'nonerg_lnratios_%s_mag%.1f_T%.2fsec.txt'%(name_haz, m, p)
        #indices and psha data
        idx = np.arange( i_m.sum() ) + 1
        data4psha_samp = np.vstack([idx, np.fliplr(eq_latlon[i_m,:]).T, rpsa_nerg[i_m,j,:].T]).T
        df_data4psha_samp = pd.DataFrame(data4psha_samp, columns=header_rpsa).astype({'x1': 'int'})
        #write header file
        with open(dir_out + fname_psha_samp, 'w') as file_psha_samp:
            file_psha_samp.write(f'{site_latlon[1]:5.3f} {site_latlon[0]:5.3f}\n')
            file_psha_samp.write(f'{n_latlon[1]:5} {n_latlon[0]:5} {grid_dlatdlon[1]:5.4f} {grid_dlatdlon[0]:5.4f} ')
            file_psha_samp.write(f'{grid_win[0,1]:7.4f} {grid_win[0,0]:7.4f} {nsamp:5}\n')
        #write lat lon and random samples
        df_data4psha_samp.to_csv(dir_out + fname_psha_samp, mode='a', sep='\t', float_format='%9.5f', header=True, index=False)

# eas ratios
# initialize header samples
header_reas = ['x1', 'lon', 'lat'] + ['sim_%i'%(j+1) for j in range(nsamp)]
# iterate over different mag and periods
for k, m in enumerate(mag):
    i_m = mag_array == m
    for p in per_psa:
        j = np.argmin( np.abs(freq_eas - 1/p) )
        #file name
        fname_psha_samp = 'nonerg_eas_lnratios_%s_mag%.1f_T%.2fsec.txt'%(name_haz, m, p)
        #indices and psha data
        idx = np.arange( i_m.sum() ) + 1
        data4psha_samp = np.vstack([idx, np.fliplr(eq_latlon[i_m,:]).T, reas_nerg[i_m,j,:].T]).T
        df_data4psha_samp = pd.DataFrame(data4psha_samp, columns=header_reas).astype({'x1': 'int'})
        #write header file
        with open(dir_out + fname_psha_samp, 'w') as file_psha_samp:
            file_psha_samp.write(f'{site_latlon[1]:5.3f} {site_latlon[0]:5.3f}\n')
            file_psha_samp.write(f'{n_latlon[1]:5} {n_latlon[0]:5} {grid_dlatdlon[1]:5.4f} {grid_dlatdlon[0]:5.4f} ')
            file_psha_samp.write(f'{grid_win[0,1]:7.4f} {grid_win[0,0]:7.4f} {nsamp:5}\n')
        #write lat lon and random samples
        df_data4psha_samp.to_csv(dir_out + fname_psha_samp, mode='a', sep='\t', float_format='%9.5f', header=True, index=False)


# Create input file
# --------------------------------
for j, p in enumerate(per_psa):
    #file name
    fname_psha_input = 'nonerg_input_%s_T%.2fsec.txt'%(name_haz, p)
    #earthquake constant
    c0_gmm1 = c0_gmm1_array[j]
    c0_gmm2 = c0_gmm2_array[j]
    with open(dir_out + fname_psha_input, 'w') as file_psha_input:
        file_psha_input.write(f'{nsamp*n_psa_gmm} Number of GM models\n')
        #write out lines for random samples
        for i in range(nsamp):
            nerg_jcalc = nerg_jcalc0 -i -1
            wt = wt_array[i]
            for jcalc_m, jcalc_s, c0 in zip(jcalc_mean, jcalc_sig, [c0_gmm1, c0_gmm2]):
                file_psha_input.write(f'{nerg_jcalc} {c0:9.5f} 0 {wt} 0 1 {jcalc_s} 0 0\n')
                file_psha_input.write(f'{jcalc_m} 0 \n')
                    

