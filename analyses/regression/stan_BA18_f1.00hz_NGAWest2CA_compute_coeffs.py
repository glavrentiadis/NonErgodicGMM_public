#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:34:34 2020

@author: glavrent
"""
#change working directory
import os
os.chdir('/mnt/halcloud_nfs/glavrent/Research/Public_Repos/NonErgodicGMM_public/analyses/regression')

#load variables
import pathlib
import glob
import re           #regular expression package
import pickle
#arithmetic libraries
import numpy as np
from scipy import linalg
#geometry libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
#geographic coordinates
import utm
import pyproj
#plottign libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
#statistics libraries
import pandas as pd
#user-derfined functions
import pylib.pylib_GP_model as pygp
import pylib.pylib_contour_plots as pycplt

# Define Input Data
# ---------------------------
freq = 1.0000
GMMmodel       = 'NergEASGMM_f%.2fhz_NGAWestCA'%freq
analysis_sufix = '_laten_var_unbound_hyp'
#input filenames
fname_resfile  = '../../data/BA18resNGA2WestCA_freq%.4f_allnergcoef.csv'%freq
fname_cellinfo = '../../data/NGA2WestCA_cellinfo.csv'
dir_stan       = '../../data/output/NergEASGMM_f%.2fhz/'%freq

#flags option
flag_data_out = False
flag_pub = True #figures for publishing
flag_plot_path = False


# Load Data
# ---------------------------
fname_analysis = r'%s%s'%(GMMmodel,analysis_sufix)
df_cellinfo         = pd.read_csv(fname_cellinfo)
df_posterior_pdf    = pd.read_csv(dir_stan + fname_analysis + '_stan_posterior' +  '.csv', index_col=0)
df_coeffs           = pd.read_csv(dir_stan + fname_analysis + '_stan_coefficients' + '.csv')
df_cellatten        = pd.read_csv(dir_stan + fname_analysis + '_stan_catten' + '.csv')
df_resfile          = pd.read_csv(dir_stan + fname_analysis + '_stan_residuals' + '.csv')
dir_out             = dir_stan + 'postproc/'
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#override data output option
if flag_pub:
    # mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['axes.linewidth'] = 2
else:
    mpl.use('Agg')

# Process Data
# ---------------------------
#grid limits and size
coeff_latlon_win = np.array([[ 31.25, -124.50], [42.00, -114.00]])
coeff_X_dxdy =     [10, 10]
coeff_latlon_polyg = np.array([[42.0, -114.0], [31.25,-114.0], [31.25,-117.0], 
                               [34.5, -121.5], [40.0, -124.5], [42.0, -124.5],]) 

#color bar limits
cbar_lim_c1e_mu   = [-.20, 0.20]
cbar_lim_c1e_sig  = [0.00, 0.15]
cbar_lim_c1as_mu  = [-.65, 0.65]
cbar_lim_c1as_sig = [0.00, 0.35]
cbar_lim_c1bs_mu  = [-2.25,2.25]
cbar_lim_c1bs_sig = [0.00, 0.40]
cbar_lim_dca_mu   = [-0.022,0.022]
cbar_lim_ca_sig   = [0.0, 0.0065]
plot_latlon_win = np.array([[ 32.50, -124.25],[42.00, -114.25]])

#hyper-parameters
dc_0      = df_posterior_pdf.loc['mean','dc_0']
#epistemic uncetainty terms
ell_1e    = df_posterior_pdf.loc['mean','ell_1e']
omega_1e  = df_posterior_pdf.loc['mean','omega_1e']
ell_1as   = df_posterior_pdf.loc['mean','ell_1as']
omega_1as = df_posterior_pdf.loc['mean','omega_1as']
omega_1bs = df_posterior_pdf.loc['mean','omega_1bs']
#cell attenuation
mu_ca     = df_cellatten.c_a_erg.values[0]
ell_ca1   = df_posterior_pdf.loc['mean','ell_ca1']
omega_ca1 = df_posterior_pdf.loc['mean','omega_ca1']
omega_ca2 = df_posterior_pdf.loc['mean','omega_ca2']
pi_ca     = 0.0

#coordinates and projection system
# projection system
utm_zone = np.unique(df_resfile.zoneUTM)[0] #utm zone
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
#earthquake and station points on UTM coordinates
eq_X_train          = df_resfile[['eqUTMx', 'eqUTMy']].values
stat_X_train        = df_resfile[['staUTMx', 'staUTMy']].values
cell_X_train        = df_cellatten[['cellX','cellY']].values
#earthquake and station coordinates
eq_latlon_train     = df_resfile[['eqLat', 'eqLon']].values
stat_latlon_train   = df_resfile[['staLat', 'staLon']].values   
cell_latlon_train   = np.fliplr(np.array([utmProj(c_x*1000, c_y*1000, inverse=True) for c_x, c_y in 
                                zip(cell_X_train[:,0], cell_X_train[:,1])]))

#earthquake and station ids
eq_id_train  = df_coeffs['eqid'].values.astype(int)
sta_id_train = df_coeffs['ssn'].values.astype(int)
eq_id, eq_idx_inv   = np.unique(eq_id_train,  return_index=True)
sta_id, sta_idx_inv = np.unique(sta_id_train, return_index=True)

#check consistency of coordinates in coefficient and GM data-frames 
assert(np.all(np.abs(eq_X_train - df_coeffs[['eqX','eqY']].values)<1e-6)), \
    'Inconsistent coordinates between ground motion and coefficient flatfiles'
assert(np.all(np.abs(stat_X_train - df_coeffs[['staX','staY']].values)<1e-6)), \
    'Inconsistent coordinates between ground motion and coefficient flatfiles'

#unique earthquake and station coordinates
eq_latlon   = eq_latlon_train[eq_idx_inv,:]
stat_latlon = stat_latlon_train[sta_idx_inv,:]
eq_X        = eq_X_train[eq_idx_inv,:]
stat_X      = stat_X_train[sta_idx_inv,:]

# Coefficient grid
# ---------------------------
#grid limits in UTM
coeff_X_win = np.array([utmProj(c_lon, c_lat) for c_lat, c_lon in 
                        zip(coeff_latlon_win[:,0], coeff_latlon_win[:,1])]) / 1000
coeff_X_win = np.vstack((np.floor(coeff_X_win[0,:]/100), np.ceil(coeff_X_win[1,:]/100)))*100

#polygon in UTM
coeff_X_polyg = np.array([utmProj(c_lon, c_lat) for c_lat, c_lon in 
                          zip(coeff_latlon_polyg[:,0], coeff_latlon_polyg[:,1])]) / 1000
shp_coeff_X_polyg = shp_poly(coeff_X_polyg)

#create coordinate grid
coeff_x_array = np.arange(coeff_X_win[0,0],coeff_X_win[1,0],coeff_X_dxdy[0])
coeff_y_array = np.arange(coeff_X_win[0,1],coeff_X_win[1,1],coeff_X_dxdy[0])
coeff_x_grid,coeff_y_grid = np.meshgrid(coeff_x_array,coeff_y_array)
#create coordinate array with all grid nodes
coeff_X = np.vstack([coeff_x_grid.T.flatten(), coeff_y_grid.T.flatten()]).T
#compute lat/lon coordinate array
coeff_latlon = np.fliplr(np.array([utmProj(c_x*1000, c_y*1000, inverse=True) for c_x, c_y in 
                                   zip(coeff_X[:,0], coeff_X[:,1])]))

grid_id = np.arange(len(coeff_X), dtype=int) + 1
grid_id = ['g.%i'%gid for gid in np.arange(len(coeff_X), dtype=int) + 1]

del coeff_x_array, coeff_y_array
del coeff_x_grid, coeff_y_grid

# Cell grid
# ---------------------------
#cell utm coordiantes 
cell_X      = df_cellinfo[['mptUTMx','mptUTMy']].values 
#cell latitude and longitude coordinates
cell_latlon = np.fliplr(np.array([utmProj(c_x*1000, c_y*1000, inverse=True) for c_x, c_y in 
                                          zip(cell_X[:,0], cell_X[:,1])])) 

cell_id     = np.array([int( re.search('c.(\\d+)', text).group(1) ) for text in df_cellinfo.cell_names])
cell_id     = ['c.%i'%cid for cid in cell_id]

# Grid points in polygon
# ---------------------------
#coefficient grid points in polygon
i_coeff_X = np.zeros(coeff_X.shape[0], dtype=bool)
for k, c_x in enumerate(coeff_X):
    shp_c_x = shp_pt(c_x)
    i_coeff_X[k] = shp_c_x.within(shp_coeff_X_polyg)

grid_id      = [grid_id[k] for k, i_c in enumerate(i_coeff_X) if i_c ]
coeff_X      = coeff_X[i_coeff_X]
coeff_latlon = coeff_latlon[i_coeff_X]
n_coeff_test = sum(i_coeff_X)

#cells inside the polygon
i_cell_X = np.zeros(cell_X.shape[0], dtype=bool)
for k, (c_mpt_x, c_pt5_x, c_pt6_x, c_pt7_x, c_pt8_x) in enumerate(zip(cell_X, df_cellinfo[['q5UTMx','q5UTMy']].values, df_cellinfo[['q6UTMx','q6UTMy']].values, 
                                                                              df_cellinfo[['q7UTMx','q7UTMy']].values, df_cellinfo[['q8UTMx','q8UTMy']].values)):
    i_c_mpt_x = shp_pt(c_mpt_x).within(shp_coeff_X_polyg)
    i_c_pt5_x = shp_pt(c_pt5_x).within(shp_coeff_X_polyg)
    i_c_pt6_x = shp_pt(c_pt6_x).within(shp_coeff_X_polyg)
    i_c_pt7_x = shp_pt(c_pt7_x).within(shp_coeff_X_polyg)
    i_c_pt8_x = shp_pt(c_pt8_x).within(shp_coeff_X_polyg)
    i_cell_X[k] = np.any([i_c_mpt_x,i_c_pt5_x,i_c_pt6_x,i_c_pt7_x,i_c_pt8_x])

cell_id      = [cell_id[k] for k, i_c in enumerate(i_cell_X) if i_c ]
cell_X      = cell_X[i_cell_X, :]
cell_latlon = cell_latlon[i_cell_X]

# Sample coefficients
# ---------------------------
#earthquake constant (c_1e)
c1e_mu, c1e_sig, c1e_cov        = pygp.SampleCoeffs(coeff_X, eq_X_train[eq_idx_inv,:], 
                                                    c_data_mu = df_coeffs.loc[eq_idx_inv,'dc_1e_mean'].values, c_data_sig = df_coeffs.loc[eq_idx_inv,'dc_1e_unc'].values,
                                                    hyp_ell = ell_1e, hyp_omega = omega_1e)

#site constant with finite correlation length(c_1as)
c1as_mu, c1as_sig, c1as_cov  = pygp.SampleCoeffs(coeff_X, stat_X_train[sta_idx_inv,:], 
                                                    c_data_mu = df_coeffs.loc[sta_idx_inv,'dc_1as_mean'].values, c_data_sig = df_coeffs.loc[sta_idx_inv,'dc_1as_unc'].values,
                                                    hyp_ell = ell_1as, hyp_omega = omega_1as)

#earthquake constant with zero correlation length (c_1bs)
c1bs_mu, c1bs_sig               = df_coeffs.loc[sta_idx_inv,'b_1bi_mean'].values, df_coeffs.loc[sta_idx_inv,'b_1bi_unc'].values
c1bs_cov                        = np.diag(c1bs_sig**2)

#attenuation coefficient (c_cA)
ca_mu, ca_sig, ca_cov           = pygp.SampleAttenCoeffsNegExp(cell_X, cell_X_train, 
                                                                cA_data_mu = df_cellatten['b_cA_mean'].values, cA_data_sig = df_cellatten['b_cA_unc'].values,
                                                                mu_ca = mu_ca, ell_ca = ell_ca1, omega_ca = omega_ca1, 
                                                                pi_ca = pi_ca, sigma_ca=omega_ca2)
ca_mu[ca_mu>0] = 0.
dca_mu = ca_mu - mu_ca

#between event terms
dBe_mu  = df_resfile.loc[eq_idx_inv,'res_between'].values
dBe_sig = np.zeros(dBe_mu.shape)

# Summarize coefficient data-files
# ---------------------------
#save coeffs with finite-cor length
summary_coeffs = {'grid_id': grid_id, 'lat': coeff_latlon[:,0], 'lon': coeff_latlon[:,1], 
                                       'utmX': coeff_X[:,0],    'utmY': coeff_X[:,1],     'utmZone': [utm_zone] * n_coeff_test,
                                       'dc1e_mu': c1e_mu, 'dc1e_sig': c1e_sig, 'dc1as_mu': c1as_mu,'dc1as_sig': c1as_sig}
pd_summary_coeffs = pd.DataFrame(summary_coeffs)

#save covariance matrices
#c1e
pd_summary_c1e_cov = pd.concat((pd_summary_coeffs['grid_id'],  pd.DataFrame(c1e_cov, columns=grid_id)), axis=1)
#c1as
pd_summary_c1as_cov = pd.concat((pd_summary_coeffs['grid_id'],  pd.DataFrame(c1as_cov, columns=grid_id)), axis=1)

#save c1bs
summary_c1bs = {'ssn': sta_id, 'lat': stat_latlon[:,0], 'lon': stat_latlon[:,1], 
                               'utmX': stat_X[:,0],     'utmY': stat_X[:,1],     'utmZone': [utm_zone] * len(stat_latlon),
                               'dc1bs_mu': c1as_mu, 'dc1bs_sig': c1bs_sig}
pd_summary_c1bs = pd.DataFrame(summary_c1bs)

#save anelastic attenuation coeffs
summary_cA = {'cell_id': cell_id, 'lat': cell_latlon[:,0], 'lon': cell_latlon[:,1], 
                                  'utmX': cell_X[:,0],     'utmY': cell_X[:,1],     'utmZone': [utm_zone] * len(cell_X),
                                  'ca_mu': ca_mu, 'dca_mu': dca_mu, 'ca_sig': ca_sig}
pd_summary_cA = pd.DataFrame(summary_cA)
#covariance matrix
pd_summary_ca_cov = pd.concat((pd_summary_cA['cell_id'],  pd.DataFrame(ca_cov, columns=cell_id)), axis=1)

#save between event terms
summary_dBe = {'eqid': eq_id, 'lat': eq_latlon[:,0], 'lon': eq_latlon[:,1], 
                              'utmX': eq_X[:,0],     'utmY': eq_X[:,1],     'utmZone': [utm_zone] * len(eq_latlon),
                              'dBe_mu': dBe_mu, 'dBe_sig': dBe_sig}
pd_summary_dBe = pd.DataFrame(summary_dBe)

#save hyper-parameters
pd_summary_hyp = df_posterior_pdf.rename(columns={'b_X0':'c0', 'rho_eqX1a':'c1a_rho',                                 'rho_statX1bii':'c1b_rho', 
                                                               'theta_eqX1a':'c1a_theta', 'theta_statX1bi':'omega_1bs', 'theta_statX1bii':'c1b_theta',
                                                               'ell_ca1':'cA_rho', 'omega_ca1':'cA_theta', 'omega_ca2': 'ca_sigma'})
pd_summary_hyp.loc[:,'c0N']   = np.nan
pd_summary_hyp.loc[:,'c0S']   = np.nan
pd_summary_hyp.loc[:,'ca_mu'] = np.nan
pd_summary_hyp.loc[:,'cA_pi'] = np.nan
pd_summary_hyp.loc[:,'c0N'][-1]   = 0.
pd_summary_hyp.loc[:,'c0S'][-1]   = 0.
pd_summary_hyp.loc[:,'ca_mu'][-1] = mu_ca
pd_summary_hyp.loc[:,'cA_pi'][-1] = pi_ca
pd_summary_hyp = pd_summary_hyp[['c0','c0N','c0S',
                                 'c1a_rho', 'c1a_theta', 'c1b_rho', 'c1b_theta', 
                                 'ca_mu', 'cA_rho',  'cA_theta', 'ca_sigma',  'cA_pi',
                                 'omega_1bs', 'tau_0', 'phi_0']]

#save regression data
#save c1a data
summary_data_c1a = {'eqid': eq_id, 'lat': eq_latlon[:,0], 'lon': eq_latlon[:,1], 
                                   'utmX': eq_X[:,0],     'utmY': eq_X[:,1],     'utmZone': [utm_zone] * len(eq_latlon),
                                   'c1e_mu': df_coeffs.loc[eq_idx_inv,'b_1a_mean'].values, 'c1e_sig':  df_coeffs.loc[eq_idx_inv,'b_1a_unc'].values}
pd_summary_data_c1a = pd.DataFrame(summary_data_c1a)

#save c1bi data
pd_summary_data_c1bi = pd_summary_c1bs

#save c1bii data
summary_data_c1bii = {'ssn': sta_id, 'lat': stat_latlon[:,0], 'lon': stat_latlon[:,1], 
                                     'utmX': stat_X[:,0],     'utmY': stat_X[:,1],     'utmZone': [utm_zone] * len(stat_latlon),
                                     'c1as_mu': df_coeffs.loc[sta_idx_inv,'b_1bii_mean'].values, 'c1as_sig':  df_coeffs.loc[sta_idx_inv,'b_1bii_unc'].values}
pd_summary_data_c1bii = pd.DataFrame(summary_data_c1bii)

#save cA data
pd_summary_data_cA = df_cellatten.rename(columns={'cell_i':'cell_id','cellX':'utmX','cellY':'utmY','zoneUTM':'utmZone',
                                                  'b_c7_erg':'cA_erg','b_cA_mean':'ca_mu','b_cA_unc':'ca_sig'})
pd_summary_data_cA.loc[:,'dca_mu']  = pd_summary_data_cA.ca_mu - mu_ca
pd_summary_data_cA.loc[:,['lat','lon']] = cell_latlon_train
pd_summary_data_cA = pd_summary_data_cA[['cell_id','lat','lon','utmX','utmY','utmZone','cA_erg','ca_mu','dca_mu','ca_sig']]

#save dBe data
pd_summary_data_dBe = pd_summary_dBe

#save hyper-parameters data
pd_summary_data_hyp = pd_summary_hyp

del c1e_mu,   c1e_sig,   c1e_cov
del c1as_mu,  c1as_sig, c1as_cov
del c1bs_mu,  c1bs_sig,  c1bs_cov
del ca_mu,    ca_sig,    ca_cov 
del dBe_mu,   dBe_sig
del summary_coeffs, summary_c1bs

# Save coefficients data-files
# ---------------------------
if flag_data_out:
    #save as .h5
    #original data
    pd_summary_data_c1a.to_hdf(dir_out    + fname_analysis + '_data_all' + '.h5', key='data_c1a',  mode='w', complevel=9, index=False)
    pd_summary_data_c1bi.to_hdf(dir_out   + fname_analysis + '_data_all' + '.h5', key='data_dS2S', mode='a', complevel=9, index=False)
    pd_summary_data_c1bii.to_hdf(dir_out  + fname_analysis + '_data_all' + '.h5', key='data_c1b',  mode='a', complevel=9, index=False)
    pd_summary_data_dBe.to_hdf(dir_out    + fname_analysis + '_data_all' + '.h5', key='data_dBe',  mode='a', complevel=9, index=False)
    pd_summary_data_cA.to_hdf(dir_out     + fname_analysis + '_data_all' + '.h5', key='data_cA',   mode='a', complevel=9, index=False)
    pd_summary_data_hyp.to_hdf(dir_out    + fname_analysis + '_data_all' + '.h5', key='hyp_param', mode='a', complevel=9)
    #interpolated data
    pd_summary_coeffs.to_hdf(dir_out    + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_spvar',     mode='w', complevel=9, index=False)
    pd_summary_c1bs.to_hdf(dir_out      + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_dS2S',      mode='a', complevel=9, index=False)
    pd_summary_c1e_cov.to_hdf(dir_out   + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_c1e_cov',   mode='a', complevel=9, index=False)
    pd_summary_c1as_cov.to_hdf(dir_out + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_c1b_cov',   mode='a', complevel=9, index=False)
    pd_summary_cA.to_hdf(dir_out        + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_cA',        mode='a', complevel=9, index=False)
    pd_summary_ca_cov.to_hdf(dir_out    + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_ca_cov',    mode='a', complevel=9, index=False)
    pd_summary_dBe.to_hdf(dir_out       + fname_analysis + '_coeffs_all' + '.h5', key='coeffs_dBe',       mode='a', complevel=9, index=False)
    pd_summary_hyp.to_hdf(dir_out       + fname_analysis + '_coeffs_all' + '.h5', key='hyp_param',        mode='a', complevel=9)
    
    #save as .csv
    #original data
    pd_summary_data_c1a.to_csv(dir_out    + fname_analysis + '_data_c1a'     + '.csv', index=False)
    pd_summary_data_c1bi.to_csv(dir_out   + fname_analysis + '_data_dS2S'    + '.csv', index=False)
    pd_summary_data_c1bii.to_csv(dir_out  + fname_analysis + '_data_c1b'     + '.csv', index=False)
    pd_summary_data_dBe.to_csv(dir_out    + fname_analysis + '_data_dBe'     + '.csv', index=False)
    pd_summary_data_cA.to_csv(dir_out     + fname_analysis + '_data_cA'      + '.csv', index=False)
    # pd_summary_data_hyp.to_csv(dir_out    + fname_analysis + '_data_hyp'     + '.csv')
    #interpolated data
    pd_summary_coeffs.to_csv(dir_out    + fname_analysis + '_coeffs'         + '.csv', index=False)
    pd_summary_c1bs.to_csv(dir_out      + fname_analysis + '_coeff_dS2S'     + '.csv', index=False)
    pd_summary_c1e_cov.to_csv(dir_out   + fname_analysis + '_ceoff_c1e_cov'  + '.csv', index=False)
    pd_summary_c1as_cov.to_csv(dir_out + fname_analysis + '_coeff_c1b_cov'  + '.csv', index=False)
    pd_summary_cA.to_csv(dir_out        + fname_analysis + '_cell_ca1_coeffs' + '.csv', index=False)
    pd_summary_ca_cov.to_csv(dir_out    + fname_analysis + '_cell_ca1_cov'    + '.csv', index=False)
    pd_summary_dBe.to_csv(dir_out       + fname_analysis + '_coeff_dBe'      + '.csv', index=False)
    pd_summary_hyp.to_csv(dir_out       + fname_analysis + '_hyp_param'      + '.csv')

# Plot coefficients
# ---------------------------
# earthquake constant (c_1a)
# ----   ----   ----   ----
#median estimate
fname_fig = fname_analysis + '_c1a_med'
cbar_label = '$\mu_{\delta c_{1,e}}$' if flag_pub else '$\mu_{\delta c_{1,e}} (f=%.2fhz)$'%freq
cbar_label = '$\mu( \delta c_{1,e} )$' if flag_pub else '$\mu( \delta c_{1,e} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_coeffs[['lat','lon','c1e_mu']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCoeffCAMapMed(data2plot, cmin=cbar_lim_c1e_mu[0],  cmax=cbar_lim_c1e_mu[1], log_cbar = False, frmt_clb = '%.2f')
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
ax.plot(eq_latlon[:,1], eq_latlon[:,0],  '^', transform = data_crs, color = 'k', markersize = 6, zorder=11)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

#epistemic uncertnaity
fname_fig = fname_analysis + '_c1e_sigma'
cbar_label = '$\psi_{\delta c_{1,e}}$' if flag_pub else '$\psi_{\delta c_{1,e}} (f=%.2fhz)$'%freq
cbar_label = '$\psi( \delta c_{1,e} )$' if flag_pub else '$\psi( \delta c_{1,e})$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_coeffs[['lat','lon','c1e_sig']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCoeffCAMapSig(data2plot, cmin=cbar_lim_c1e_sig[0],  cmax=cbar_lim_c1e_sig[1], log_cbar=False)
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
ax.plot(eq_latlon[:,1], eq_latlon[:,0],  '^', transform = data_crs, color = 'k', markersize = 6, zorder=11)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

# site constant (c_1bii)
# ----   ----   ----   ----
#median estimate
fname_fig = fname_analysis + '_c1b_med'
cbar_label = '$\mu_{\delta c_{1a,s}}$'  if flag_pub else '$\mu_{\delta c_{1a,s}} (f=%.2fhz)$'%freq
cbar_label = '$\mu( \delta c_{1a,s} )$' if flag_pub else '$\mu( \deltac_{1a,s} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_coeffs[['lat','lon','c1as_mu']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCoeffCAMapMed(data2plot, cmin=cbar_lim_c1as_mu[0],  cmax=cbar_lim_c1as_mu[1], log_cbar = False, frmt_clb = '%.2f')
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

#epistemic uncertnaity
fname_fig = fname_analysis + '_c1as_sigma'
cbar_label = '$\psi_{\delta c_{1a,s}}$'  if flag_pub else '$\psi_{\delta c_{1a,s}} (f=%.2fhz)$'%freq
cbar_label = '$\psi( \delta c_{1a,s} )$' if flag_pub else '$\psi( \delta c_{1a,s} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_coeffs[['lat','lon','c1as_sig']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCoeffCAMapSig(data2plot, cmin=cbar_lim_c1as_sig[0],  cmax=cbar_lim_c1as_sig[1], log_cbar=False)
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

# site term (dS2S)
# ----   ----   ----   ----
#median estimate
fname_fig = fname_analysis + '_dS2S_med'
cbar_label = '$\mu_{\delta c_{1b,s}}$'  if flag_pub else '$\mu_{\delta c_{1b,s}} (f=%.2fhz)$'%freq
cbar_label = '$\mu( \delta c_{1b,s} )$' if flag_pub else '$\mu( \delta c_{1b,s} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_data_c1bi[['lat','lon','dc1bs_mu']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCellsCAMapMed(data2plot, cmin=cbar_lim_c1bs_mu[0],  cmax=cbar_lim_c1bs_mu[1], log_cbar = False, frmt_clb = '%.2f')
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
#ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

#epistemic uncertnaity
fname_fig = fname_analysis + '_dc1bs_sigma'
cbar_label = '$\psi_{\delta c_{1b,s}}$'  if flag_pub else '$\psi_{\delta c_{1b,s}} (f=%.2fhz)$'%freq
cbar_label = '$\psi( \delta c_{1b,s} )$' if flag_pub else '$\psi( \delta c_{1b,s} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_data_c1bi[['lat','lon','dc1bs_sig']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCellsCAMapSig(data2plot, cmin=cbar_lim_c1as_sig[0],  cmax=cbar_lim_c1as_sig[1], log_cbar=False)
ax.set_xlim( plot_latlon_win[:,1] ) #ax.set_xlim(ax.get_xlim())
ax.set_ylim( plot_latlon_win[:,0] ) #ax.set_ylim(ax.get_ylim())
#ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')


# anelastic attenuation cells (c_cA)
# ----   ----   ----   ----
#cell edge coordinates
cell_edge_latlon = []
for cell_edge in [['q5UTMx','q5UTMy'], ['q6UTMx','q6UTMy'], ['q8UTMx','q8UTMy'], 
                  ['q7UTMx','q7UTMy'], ['q5UTMx','q5UTMy']]:
    
    cell_edge_latlon.append( np.fliplr(np.array([utmProj(c_xy[0]*1000, c_xy[1]*1000, inverse=True) for c_xy in 
                                                 df_cellinfo.loc[:,cell_edge].values])) )                       
cell_edge_latlon = np.hstack(cell_edge_latlon)

#median estimate
fname_fig = fname_analysis + '_cA_med'
cbar_label = '$\mu_{c_{ca,p}}$'  if flag_pub else '$\mu_{c_{ca,p}} (f=%.2fhz)$'%freq
cbar_label = '$\mu( c_{ca,p} )$' if flag_pub else '$\mu( c_{ca,p})$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_cA[['lat','lon','ca_mu']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCellsCAMapMed(data2plot, cmin=cbar_lim_dca_mu[0]+mu_ca,  cmax=cbar_lim_dca_mu[1]+mu_ca, log_cbar = False, frmt_clb = '%.4f')
ax.set_xlim( plot_latlon_win[:,1] )
ax.set_ylim( plot_latlon_win[:,0] )
#plot earthquake and station locations
ax.plot(eq_latlon[:,1],   eq_latlon[:,0],    '^', transform = data_crs, color = 'k', markersize = 6, zorder=11)
ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#plot earthquake-station paths
if flag_plot_path:
    for rec in df_flatfile[['eqLat','eqLon','staLat','staLon']].iterrows():
        ax.plot(rec[1][['eqLon','staLon']], rec[1][['eqLat','staLat']], transform = data_crs, color = 'k', linestyle='--', linewidth=0.05, zorder=10)
#plot cells
for ce_xy in cell_edge_latlon:
    ax.plot(ce_xy[[1,3,5,7,9]],ce_xy[[0,2,4,6,8]],color='gray', transform = data_crs)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

#median delta estimate
fname_fig = fname_analysis + '_deltacA_med'
cbar_label = '$\mu_{\delta c_{ca,p}}$'  if flag_pub else '$\mu_{\delta c_{ca,p}} (f=%.2fhz)$'%freq
cbar_label = '$\mu( \delta c_{ca,p} )$' if flag_pub else '$\mu( \delta c_{ca,p} )$ ($f=%.2fhz$)'%freq

data2plot = pd_summary_cA[['lat','lon','dca_mu']].values
#create figure
fig, ax, cbar, data_crs, _ = pycplt.PlotCellsCAMapMed(data2plot, cmin=cbar_lim_dca_mu[0],  cmax=cbar_lim_dca_mu[1], log_cbar = False, frmt_clb = '%.4f')
ax.set_xlim( plot_latlon_win[:,1] )
ax.set_ylim( plot_latlon_win[:,0] )
#plot earthquake and station locations
ax.plot(eq_latlon[:,1],   eq_latlon[:,0],    '^', transform = data_crs, color = 'k', markersize = 6, zorder=11)
ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#plot earthquake-station paths
if flag_plot_path:
    for rec in df_flatfile[['eqLat','eqLon','staLat','staLon']].iterrows():
        ax.plot(rec[1][['eqLon','staLon']], rec[1][['eqLat','staLat']], transform = data_crs, color = 'k', linestyle='--', linewidth=0.05, zorder=10)
#plot cells
for ce_xy in cell_edge_latlon:
    ax.plot(ce_xy[[1,3,5,7,9]],ce_xy[[0,2,4,6,8]],color='gray', transform = data_crs)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

#epistemic uncertnaity
fname_fig = fname_analysis + '_ca_sigma'
cbar_label = '$\psi_{c_{ca,p}}$'  if flag_pub else '$\psi_{c_{ca,p}} (f=%.2fhz)$'%freq
cbar_label = '$\psi( c_{ca,p} )$' if flag_pub else '$\psi( c_{ca,p} )$ ($f=%.2fhz$)'%freq
data2plot = pd_summary_cA[['lat','lon','ca_sig']].values
#create figure
frmt_clb = '%.1e' if flag_pub else '%.2e'
fig, ax, cbar, data_crs, _ = pycplt.PlotCellsCAMapSig(data2plot, cmin=cbar_lim_ca_sig[0],  cmax=cbar_lim_ca_sig[1], log_cbar = False, frmt_clb = frmt_clb)
ax.set_xlim( plot_latlon_win[:,1] )
ax.set_ylim( plot_latlon_win[:,0] )
#plot earthquake and station locations
ax.plot(eq_latlon[:,1],   eq_latlon[:,0],    '^', transform = data_crs, color = 'k', markersize = 6, zorder=11)
ax.plot(stat_latlon[:,1], stat_latlon[:,0],  'o', transform = data_crs, color = 'k', markersize = 2, zorder=12)
#plot earthquake-station paths
if flag_plot_path:
    for rec in df_flatfile[['eqLat','eqLon','staLat','staLon']].iterrows():
        ax.plot(rec[1][['eqLon','staLon']], rec[1][['eqLat','staLat']], transform = data_crs, color = 'k', linestyle='--', linewidth=0.05, zorder=10)
#plot cells
for ce_xy in cell_edge_latlon:
    ax.plot(ce_xy[[1,3,5,7,9]],ce_xy[[0,2,4,6,8]],color='gray', transform = data_crs)
#update colorbar 
if flag_pub: 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, size=25)
else:
    cbar.set_label(cbar_label, size=18)
#grid lines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.xlocator = mticker.FixedLocator([-123,-121,-119,-117,-115])
gl.ylocator = mticker.FixedLocator([ 33,  35,  37,  39,  41])
#apply tight layout
fig.show()
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png')

