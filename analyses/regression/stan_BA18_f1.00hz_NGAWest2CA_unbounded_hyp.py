#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:54:37 2020

@author: glavrent
"""

#change working directory
import os
os.chdir('/mnt/halcloud_nfs/glavrent/Research/Public_repos/NonErgodicGMM_public/analyses/regression')

#load variables
import pathlib
import glob
import re           #regular expression package
import pickle
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
import pystan
#plot libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
import arviz as az
#user-derfined functions
import pylib.pylib_GP_model as pygp

# Define Variables
# ---------------------------
freq = 1.0000
flag_runstan = True
#analyis filename
GMMmodel        = 'NergEASGMM_f%.2fhz_NGAWestCA'%freq
analysis_sufix  = '_laten_var_unbound_hyp'
#maximum magnitude for sub-regional classification
mag_sreg = 5
#residuals flatfile
fname_flatfile      = '../../data/BA18resNGA2WestCA_freq%.4f_allnergcoef.csv'%freq
fname_celldistfile  = '../../data/NGA2WestCA_distancematrix.csv'
fname_cellinfo      = '../../data/NGA2WestCA_cellinfo.csv'
#output directory
dir_out             = '../../data/output/NergEASGMM_f%.2fhz/'%freq

#create output directory
if not os.path.isdir(dir_out ): pathlib.Path(dir_out ).mkdir(parents=True, exist_ok=True) #create output dir if not existent

#latlon window
win_eq_latlon = np.array([[-np.inf, +np.inf],[-np.inf, +np.inf]])

## Stan Code
# ---------------------------
stan_model_code = """
/*********************************************
Stan program to obtain VCM parameters
lower dimensions is used (event terms/station terms)

This model explicitly estimates the latent (uncorrelated) event terms and station terms
 ********************************************/

data {
  int N;      // number of records
  int NEQ;    // number of earthquakes
  int NSTAT;  // number of stations
  int NCELL;  // number of cells
  int NSREG;  // number of sub-regions
  
  //event and station ID
  int<lower=1,upper=NEQ> eq[N];     // event id (in numerical order from 1 to last)
  int<lower=1,upper=NSTAT> stat[N]; // station id (in numerical order from 1 to last)

  //observations
  vector[N] Y; // median predictions for each record with anelasic attenuation taken out

  //mean ground motion
  vector[N] mu_rec; 

  //Earthquake, Station coordinates
  vector[2] X_e[NEQ];   // event coordinates for each record
  vector[2] X_s[NSTAT]; // station coordinates for each record
  //cell coordinates
  vector[2] X_c[NCELL];
  
  //cell distance matrix
  vector[NCELL] RC[N];  // cell paths for each record

  // classification of recordings into different sub-regions
  vector [NSREG] SREG[NEQ];

  //ergodic anelastic attenuation coefficient
  real c_a_erg;
}

transformed data {
  real delta = 1e-9;
  
  //varying coefficient prior means
  vector[NEQ]   dc_1e_mu = rep_vector(0.,NEQ);
  vector[NSTAT] dc_1as_mu = rep_vector(0.,NSTAT);

  //cells global mean
  vector[NCELL]  c_ca_mu = rep_vector(c_a_erg,NCELL);
}

parameters {
  //Aleatory Variability Terms
  real<lower=0> phi_0;  // phi_0 - remaining aleatory variability of within-event residuals
  real<lower=0> tau_0;  // tau_0 - remaining aleatory variability of between-event residuals
  
  //Epistemic Uncertainty Terms
  real<lower=0.0>  ell_1e;
  real<lower=0.0>  omega_1e;
  real<lower=0.0>  ell_1as;
  real<lower=0.0>  omega_1as;
  real<lower=0.0>  omega_1bs;
  //attenuation cells
  real<lower=0.0>  ell_ca1;
  real<lower=0.0>  omega_ca1; 
  real<lower=0>    omega_ca2;      // std of cell-specific attenuation
 
  //spatially correlated coefficients
  real dc_0;             //constant shift
  vector[NSREG] dc_0e;   //sub-regional shift
  vector[NEQ]   dc_1e;   //spatially varying eq coeff
  vector[NSTAT] dc_1as;  //spatially varying stat coeff
  vector[NSTAT] dc_1bs;  //zero correlation station term
  
  //cell-specific prior distributions
  vector<upper=0>[NCELL]  c_ca;

  //between event terms
  vector[NEQ]   dB;
}

model {
  //non-ergodic mean
  vector[NEQ] shift_sr;     //sub-regional shift for every earthquake
  vector[N] shift_inattent; //anelastic attenuation shift
  vector[N] mu_rec_nerg_dB;
  
  //Aleatory Variability Terms
  phi_0 ~ lognormal(-1.30,0.3);
  tau_0 ~ lognormal(-1,0.3);
  //Station and earthquake paramters
  dB ~ normal(0,tau_0);
  
  //non-ergodic hyper-parameters
  ell_1e  ~ inv_gamma(2.,50);
  ell_1as ~ inv_gamma(2.,50);
  ell_ca1 ~ inv_gamma(2.,50.);
  omega_1e  ~ exponential(20);
  omega_1as ~ exponential(20);
  omega_1bs ~ lognormal(-0.8,0.3); //phi_S2S
  omega_ca1 ~ exponential(20);
  omega_ca2 ~ exponential(20);

  //constant swift
  dc_0 ~ normal(0.,0.1);
  //regional swift for small mag
  dc_0e ~ normal(0.,0.20);
  
  //station contributions with zero correlation length
  dc_1bs ~ normal(0,omega_1bs);
    
  //spatillay latent variable event contributions to GP
  {
    matrix[NEQ,NEQ] cov_1e;
    
    for(i in 1:NEQ) {
      for(j in i:NEQ) {
        real d_e;
        real c_1e;
        
        d_e = distance(X_e[i],X_e[j]);
  
        c_1e = (omega_1e^2 * exp(-d_e/ell_1e));
  
        cov_1e[i,j] = c_1e;
        cov_1e[j,i] = cov_1e[i,j];
      }
      cov_1e[i,i] = cov_1e[i,i] + delta;
    }
    dc_1e ~ multi_normal(dc_1e_mu, cov_1e);
  }

  //Spatially latent variable station contributions to GP
  { 
    matrix[NSTAT,NSTAT] cov_1as;

    for(i in 1:NSTAT) {
      for(j in i:NSTAT) {
        real d_s;
        real c_1as;
  
        d_s = distance(X_s[i],X_s[j]);
        
        c_1as = (omega_1as^2  * exp(-d_s/ell_1as));
  
        cov_1as[i,j] = c_1as;
        cov_1as[j,i] = cov_1as[i,j];
      }
      cov_1as[i,i] = cov_1as[i,i] + delta;
    }
    dc_1as ~ multi_normal(dc_1as_mu, cov_1as);
  }
  
  //Spatially varying latent variable for anelastic contributions to GP  
  {
    matrix[NCELL, NCELL] cov_ca;
    
    for(i in 1:NCELL) {
      for(j in i:NCELL) {
        real d_c;
        real co_ca;
        
        d_c = distance(X_c[i],X_c[j]);
  
        co_ca = (omega_ca1^2 * exp(-d_c/ell_ca1));  //negative exp cov matrix

        cov_ca[i,j] = co_ca;
        cov_ca[j,i] = cov_ca[i,j];
      }
      cov_ca[i,i] = cov_ca[i,i] + omega_ca2^2 + delta;
    }
   c_ca ~ multi_normal(c_ca_mu, cov_ca);
  }

  //Effect of anelastic attenuation
  for(i in 1:N)
    shift_inattent[i] =  dot_product(c_ca,RC[i]);

  //sub-regional shift for every earthquake
  for(i in 1:NEQ)
    shift_sr[i] = dot_product(dc_0e,SREG[i]);

  //Mean non-ergodic including dB
  mu_rec_nerg_dB = dc_0 + shift_sr[eq] + dc_1as[eq] + dc_1as[stat] + dc_1bs[stat] + shift_inattent + dB[eq];
  
  Y ~ normal(mu_rec_nerg_dB,phi_0);

}
"""
#Compile Stan model
sm = pystan.StanModel(model_code=stan_model_code)
control_stan = {'adapt_delta':0.9,
                'max_treedepth':10}

# Read data
# ---------------------------
fname_analysis = r'%s%s'%(GMMmodel,analysis_sufix)

#spectral acc. flatfile
res_flatfile = pd.read_csv(fname_flatfile)
print(res_flatfile.columns)
res_flatfile.head()

#cell info file
cell_info = pd.read_csv(fname_cellinfo, index_col=0)

#cell distance file
cell_data = pd.read_csv(fname_celldistfile, index_col=0)
cell_data = cell_data.reindex(res_flatfile.rsn)

#keep only data in latlon window
i_inside_win = np.all(np.array([res_flatfile.eqLat >= win_eq_latlon[0,0], 
                               res_flatfile.eqLat < win_eq_latlon[0,1], 
                               res_flatfile.eqLon >= win_eq_latlon[1,0], 
                               res_flatfile.eqLon < win_eq_latlon[1,1]]),axis=0)
res_flatfile = res_flatfile[i_inside_win] 
del i_inside_win

# Sumarize Data
# ---------------------------
n_data = len(res_flatfile.eqid)

#earthquake data
data_eq_all = res_flatfile[['eqid','mag','X1a','eqUTMx', 'eqUTMy','sub_region']].values
_, eq_inv, eq_idx = np.unique(res_flatfile[['eqid']], axis=0, return_inverse=True, return_index=True)
data_eq = data_eq_all[eq_inv,:]
X_eq = data_eq[:,[3,4]] #earthquake coordinates
#sub region classification
sreg = np.array([data_eq[:,5] == sr for sr in np.unique(data_eq[:,5])]).astype(int).T
sreg[data_eq[:,1] > mag_sreg] = 0
#create earthquake indices for all covariates
eq_idx = eq_idx + 1
n_eq = len(data_eq)

#station data
data_stat_all = res_flatfile[['ssn','Vs30','X1b','X8','staUTMx','staUTMy']].values
_, sta_inv, sta_idx = np.unique( res_flatfile[['ssn']].values, axis = 0, return_inverse=True, return_index=True)
data_stat = data_stat_all[sta_inv,:]
X_stat = data_stat[:,[4,5]] #station coordinates
#create station indices for all covariates
sta_idx = sta_idx + 1
n_stat = len(data_stat)

#data dependent on both station and earthquake 
data_eq_stat = res_flatfile[['X7']].to_numpy().flatten()

#observations  
data_Y = res_flatfile.res.to_numpy().copy()
#remove anelastic attenuation from res
data_Y += res_flatfile.b_erg7.values * res_flatfile.X7.values

#cell data
cell_names_all = cell_data.columns.values
cell_i = [bool(re.match('^c\..*$',c_n)) for c_n in cell_names_all] #indices for cell columns
cell_names_all = cell_names_all[cell_i]
cell_ids_all = np.array([int( re.search('c.(\d+)', text).group(1) ) for text in cell_names_all])
#cell distance matrix
celldist_all = cell_data[cell_names_all] #cell-distance matrix with all cells
i_cells_valid = np.where(celldist_all.sum(axis=0) > 0)[0] #valid cells with more than one path
cell_names_valid = cell_names_all[i_cells_valid]
cell_ids_valid = cell_ids_all[i_cells_valid]
celldist_valid = celldist_all.loc[:,cell_names_valid] #cell-distance with only non-zero cells
n_cell = celldist_all.shape[1]
n_cell_valid = celldist_valid.shape[1]
#cell coordinates
X_cells = cell_info.loc[:,['mptUTMx','mptUTMy']].values
X_cells_valid = X_cells[i_cells_valid,:]

#print Rrup missfits
print('max R_rup misfit', max(res_flatfile.Rrup.values - celldist_valid.sum(axis=1)))
print('min R_rup misfit', min(res_flatfile.Rrup.values - celldist_valid.sum(axis=1)))


# Stan Data
# ---------------------------
stan_data = {'N':       n_data,
             'NEQ':     n_eq,
             'NSTAT':   n_stat,
             'NCELL':   n_cell_valid,
             'NSREG':   sreg.shape[1],
             'Y':       data_Y,
             'X_e':     X_eq,                 #earthquake coordinates
             'X_s':     X_stat,               #station coordinates
             'eq':      eq_idx,               #earthquake index
             'stat':    sta_idx,              #station index
             'X_c':     X_cells_valid,
             'RC':      celldist_valid.to_numpy(),
             'SREG':    sreg,
             'mu_rec':  np.zeros(data_Y.shape),
             'c_a_erg': res_flatfile.b_erg7.values[0],
            }
fname_stan_data = fname_analysis + '_stan_data' + '.Rdata'
#pystan.misc.stan_rdump(stan_data, dir_out +fname_stan_data)

## Run Stan, fit model 
# ---------------------------
fname_stan_fit = dir_out  + fname_analysis + '_stan_fit' + '.pkl'
if flag_runstan:
    #full Bayesian statistics
    fit_full = sm.sampling(data=stan_data, iter=600, chains = 4, refresh=1, control = control_stan)
    #save stan model and fit
    with open(fname_stan_fit, "wb") as f:
        pickle.dump({'model' : sm, 'fit_full' : fit_full}, f, protocol=-1)
else:
    #load model and fit
    with open(fname_stan_fit, "rb") as f:
        data_dict = pickle.load(f)
    fit_full = data_dict['fit_full']
    sm = data_dict['model']
    del data_dict

## Summarize Data
# ---------------------------
#hyper-parameters
col_names_hyp = ['dc_0','ell_1e', 'ell_1as', 'omega_1e', 'omega_1as', 'omega_1bs',
                 'ell_ca1','omega_ca1','omega_ca2',
                 'phi_0','tau_0']
col_names_hyp2 = col_names_hyp.copy()
#sub-region columns
col_names_sreg  = ['dc_0eN',   'dc_0eS']
col_names_sreg2 = ['dc_0e[1]', 'dc_0e[2]']
#column names of all hyper-parameteres
col_names  = col_names_hyp  + col_names_sreg
col_names2 = col_names_hyp2 + col_names_sreg2
#names for trace plots
trace_names  = col_names_hyp  + ['dc_0e']
trace_names2 = col_names_hyp2 + ['dc_0e']

col_names_dc_1e  = ['dc_1e.%i'%(k)  for k in range(n_eq)]
col_names_dc_1as = ['dc_1as.%i'%(k) for k in range(n_stat)]
col_names_dc_1bs = ['dc_1bs.%i'%(k) for k in range(n_stat)]
col_names_ca     = ['c_ca.%i'%(k)   for k in cell_ids_valid]
col_names_dB     = ['dB.%i'%(k)     for k in range(n_eq)]
col_names_all = col_names + col_names_dc_1e + col_names_dc_1as + col_names_dc_1bs + col_names_ca + col_names_dB

#sumarize raw posterior distributions
posterior_stan = np.stack([fit_full[v_n] for v_n in col_names2], axis=1)
#adjustment terms 
posterior_stan = np.concatenate((posterior_stan, fit_full['dc_1e']),  axis=1)
posterior_stan = np.concatenate((posterior_stan, fit_full['dc_1as']), axis=1)
posterior_stan = np.concatenate((posterior_stan, fit_full['dc_1bs']), axis=1)
posterior_stan = np.concatenate((posterior_stan, fit_full['c_ca']),   axis=1)
posterior_stan = np.concatenate((posterior_stan, fit_full['dB']),     axis=1)

#save raw-posterior distribution
df_posterior_pdf_raw = pd.DataFrame(posterior_stan, columns = col_names_all)
df_posterior_pdf_raw.to_csv(dir_out  + fname_analysis + '_stan_posterior_raw' + '.csv', index=False)

#summarize posterior distribution percentiles
perc_array = np.array([0.05,0.25,0.5,0.75,0.95])
df_posterior_pdf = df_posterior_pdf_raw[col_names].quantile(perc_array)
df_posterior_pdf = df_posterior_pdf.append(df_posterior_pdf_raw[col_names].mean(axis = 0), ignore_index=True)
df_posterior_pdf.index = ['prc_%.2f'%(prc) for prc in perc_array]+['mean'] 
df_posterior_pdf.to_csv(dir_out + fname_analysis + '_stan_posterior' +  '.csv', index=True)

del col_names_dc_1e, col_names_dc_1as, col_names_dc_1bs, col_names_ca, col_names_dB
del posterior_stan, col_names_all

## Sample spatially varying coefficients and predictions at record locations
# ---------------------------
# earthquake and station location in database
X_eq_all = res_flatfile[['eqUTMx', 'eqUTMy']].values
X_stat_all = res_flatfile[['staUTMx','staUTMy']].values    
#sub-region classification for all recordings in database
sreg_all = sreg[eq_idx-1, :] 

# GMM coefficients
#---  ---  ---  ---  ---  ---  ---  ---
#constant shift coefficient
coeff_0_mu  = df_posterior_pdf_raw.loc[:,'dc_0'].mean()   * np.ones(n_data)
coeff_0_med = df_posterior_pdf_raw.loc[:,'dc_0'].median() * np.ones(n_data)
coeff_0_sig = df_posterior_pdf_raw.loc[:,'dc_0'].std()    * np.ones(n_data)

#constant regional shift
coeff_0e_mu  = np.matmul( sreg_all, df_posterior_pdf_raw.loc[:,col_names_sreg].mean() )
coeff_0e_med = np.matmul( sreg_all, df_posterior_pdf_raw.loc[:,col_names_sreg].median() )
coeff_0e_sig = np.matmul( sreg_all, df_posterior_pdf_raw.loc[:,col_names_sreg].std() )

#spatially varying earthquake constant coefficient
coeff_1e_mu  = np.array([df_posterior_pdf_raw.loc[:,f'dc_1e.{k}'].mean()   for k in range(n_eq)])
coeff_1e_mu  = coeff_1e_mu[eq_idx-1]
coeff_1e_med = np.array([df_posterior_pdf_raw.loc[:,f'dc_1e.{k}'].median() for k in range(n_eq)])
coeff_1e_med = coeff_1e_med[eq_idx-1]
coeff_1e_sig = np.array([df_posterior_pdf_raw.loc[:,f'dc_1e.{k}'].std()    for k in range(n_eq)])
coeff_1e_sig = coeff_1e_sig[eq_idx-1]

#site term constant covariance
coeff_1as_mu  = np.array([df_posterior_pdf_raw.loc[:,f'dc_1as.{k}'].mean()   for k in range(n_stat)])
coeff_1as_mu  = coeff_1as_mu[sta_idx-1]
coeff_1as_med = np.array([df_posterior_pdf_raw.loc[:,f'dc_1as.{k}'].median() for k in range(n_stat)])
coeff_1as_med = coeff_1as_med[sta_idx-1]
coeff_1as_sig = np.array([df_posterior_pdf_raw.loc[:,f'dc_1as.{k}'].std()    for k in range(n_stat)])
coeff_1as_sig = coeff_1as_sig[sta_idx-1]

#spatially varying station constant covariance
coeff_1bs_mu  = np.array([df_posterior_pdf_raw.loc[:,f'dc_1bs.{k}'].mean()   for k in range(n_stat)])
coeff_1bs_mu  = coeff_1bs_mu[sta_idx-1]
coeff_1bs_med = np.array([df_posterior_pdf_raw.loc[:,f'dc_1bs.{k}'].median() for k in range(n_stat)])
coeff_1bs_med = coeff_1bs_med[sta_idx-1]
coeff_1bs_sig = np.array([df_posterior_pdf_raw.loc[:,f'dc_1bs.{k}'].std()    for k in range(n_stat)])
coeff_1bs_sig = coeff_1bs_sig[sta_idx-1]

# aleatory variability
phi_0_array = np.array([df_posterior_pdf_raw.phi_0.mean()]*X_stat_all.shape[0])
tau_0_array = np.array([df_posterior_pdf_raw.tau_0.mean()]*X_stat_all.shape[0])

# GMM anelastic attenuation
#---  ---  ---  ---  ---  ---  ---  ---
cells_ca_mu  = np.array([df_posterior_pdf_raw.loc[:,'c_ca.%i'%(k)].mean()   for k in cell_ids_valid])
cells_ca_med = np.array([df_posterior_pdf_raw.loc[:,'c_ca.%i'%(k)].median() for k in cell_ids_valid])
cells_ca_sig = np.array([df_posterior_pdf_raw.loc[:,'c_ca.%i'%(k)].std()    for k in cell_ids_valid])

#effect of anelastic attenuation in GM
cells_LcA_mu  = celldist_valid.values @ cells_ca_mu
cells_LcA_med = celldist_valid.values @ cells_ca_med
cells_LcA_sig = celldist_valid.values @ cells_ca_sig

# GMM prediction
#---  ---  ---  ---  ---  ---  ---  ---
eq_id_all  = eq_idx 
sta_id_all = sta_idx
data_train = res_flatfile[['X1a','X1b']].values
L_train = celldist_valid.to_numpy()

#hyper-parameters
dc_0       = df_posterior_pdf.loc['mean','dc_0']
omega_c1e  = df_posterior_pdf.loc['mean','omega_1e']
ell_c1e    = df_posterior_pdf.loc['mean','ell_1e']
omega_c1as = df_posterior_pdf.loc['mean','omega_1as']
ell_c1as   = df_posterior_pdf.loc['mean','ell_1as']
omega_c2bs = df_posterior_pdf.loc['mean','omega_1bs']
#cells
mu_ca     = res_flatfile.b_erg7.values[0]
ell_ca1   = df_posterior_pdf.loc['mean','ell_ca1']
omega_ca1 = df_posterior_pdf.loc['mean','omega_ca1']
omega_ca2 = df_posterior_pdf.loc['mean','omega_ca2']
pi_ca     = 0.

#set-up covariance function
cov_info = [0,1]
hyp_list = [[ell_c1e,     omega_c1e,     0.0],
            [ell_c1as,    omega_c1as,   0.0]]


phi_0 = df_posterior_pdf.loc['mean','phi_0']
tau_0 = df_posterior_pdf.loc['mean','tau_0']

#median prediction
y_new_mu  = (coeff_0_mu + coeff_0e_mu + coeff_1e_mu + coeff_1as_mu + coeff_1bs_mu + L_train @ cells_ca_mu)

#correct observations for regional shift
data_Y = data_Y - coeff_0e_mu
#median prediction
y_new_mu_cm = pygp.GPPredictionCells(data_Y, X_train = data_train, T_train = [X_eq_all,X_stat_all], 
                                              eqid_train = eq_id_all, sid_train = sta_id_all, 
                                              X_new = data_train,  T_new = [X_eq_all,X_stat_all],
                                                                        sid_new = sta_id_all,
                                              dc_0 = dc_0,
                                              Tid_list = cov_info, Hyp_list = hyp_list, 
                                              phi_0 = phi_0, tau_0 = tau_0, sigma_s = omega_c2bs, 
                                              T_cells_new = X_cells_valid, T_cells_train = X_cells_valid, 
                                              L_cells_new = L_train, L_cells_train = L_train,
                                              mu_ca = mu_ca, ell_ca = ell_ca1, omega_ca = omega_ca1, pi_ca = pi_ca,
                                              sigma_ca = omega_ca2)[0]

#median prediction and between event term
y_new_mudB_cm = pygp.GPPredictionCells(data_Y, X_train = data_train, T_train = [X_eq_all,X_stat_all], 
                                                eqid_train = eq_id_all, sid_train = sta_id_all, 
                                                X_new = data_train,  T_new = [X_eq_all,X_stat_all],
                                                eqid_new = eq_id_all,   sid_new = sta_id_all,
                                                dc_0 = dc_0,
                                                Tid_list = cov_info, Hyp_list = hyp_list, 
                                                phi_0 = phi_0, tau_0 = tau_0, sigma_s = omega_c2bs, 
                                                T_cells_new = X_cells_valid, T_cells_train = X_cells_valid, 
                                                L_cells_new = L_train, L_cells_train = L_train,
                                                mu_ca = mu_ca, ell_ca = ell_ca1, omega_ca = omega_ca1, pi_ca = pi_ca,
                                                sigma_ca = omega_ca2)[0]
#add regional shift
data_Y        += coeff_0e_mu
y_new_mu_cm   += coeff_0e_mu
y_new_mudB_cm += coeff_0e_mu

#compute residuals
res_tot     = data_Y - y_new_mu
res_tot_cm  = data_Y - y_new_mu_cm
#residuals computed directly from stan regression
res_between = [df_posterior_pdf_raw.loc[:,f'dB.{k}'].mean() for k in range(n_eq)]
res_between = np.array([res_between[k] for k in (eq_idx-1).astype(int)])
res_within  = res_tot - res_between
#residuals computed from conditional model
res_within_cm  = data_Y - y_new_mudB_cm
res_between_cm = res_tot_cm - res_within_cm

del ell_c1e, omega_c1e
del ell_ca1, omega_c1as
del cov_info, hyp_list, phi_0, tau_0, omega_c2bs

## Sumarize coefficients and residuals
# ---------------------------
#summary coefficients
coeffs_summary = np.vstack((coeff_0_mu,
                            coeff_0e_mu,
                            coeff_1e_mu, 
                            coeff_1as_mu,
                            coeff_1bs_mu,
                            cells_LcA_mu,
                            coeff_0_med,
                            coeff_0e_med,
                            coeff_1e_med, 
                            coeff_1as_med,
                            coeff_1bs_med,
                            cells_LcA_med,
                            coeff_0_sig,
                            coeff_0e_sig, 
                            coeff_1e_sig, 
                            coeff_1as_sig,
                            coeff_1bs_sig,
                            cells_LcA_sig)).T
coeffs_summary = np.vstack((res_flatfile[['rsn','eqid','ssn']].values.T,X_eq_all.T,X_stat_all.T,res_flatfile.zoneUTM.values,
                            coeffs_summary.T)).T
columns_names = ['rsn','eqid','ssn','eqX','eqY','staX','staY','zoneUTM',
                 'dc_0_mean','dc_0e_mean','dc_1e_mean','dc_1as_mean','dc_1bs_mean','Lc_ca_mean',
                 'dc_0_med', 'dc_0e_med', 'dc_1e_med', 'dc_1as_med', 'dc_1bs_med', 'Lc_ca_med',
                 'dc_0_unc', 'dc_0e_unc', 'dc_1e_unc', 'dc_1as_unc', 'dc_1bs_unc', 'Lc_ca_unc']

df_coeffs_summary = pd.DataFrame(coeffs_summary, columns = columns_names)
df_coeffs_summary[['rsn','eqid','ssn']] = df_coeffs_summary[['rsn','eqid','ssn']].astype(int)
df_coeffs_summary.to_csv(dir_out  + fname_analysis + '_stan_coefficients' + '.csv', index=False)

#summary attenuation cells
catten_summary = np.vstack((cell_ids_valid,
                            X_cells_valid.T,
                            np.tile(res_flatfile.zoneUTM.values[0], n_cell_valid),
                            np.tile(res_flatfile.b_erg7.values[0],  n_cell_valid),
                            cells_ca_mu,
                            cells_ca_med,
                            cells_ca_sig)).T
columns_names = ['cell_i','cellX','cellY','zoneUTM','c_a_erg','c_ca_mean','c_ca_med','c_ca_unc']

df_catten_summary = pd.DataFrame(catten_summary, columns = columns_names)
df_catten_summary['cell_i'] = df_catten_summary['cell_i'].astype(int)
df_catten_summary = pd.concat([pd.DataFrame({'cell_name': cell_names_valid}), df_catten_summary], axis=1)
df_catten_summary.to_csv(dir_out  + fname_analysis + '_stan_catten' + '.csv', index=False)

#predictions and residuals coefficients
predict_summary = np.vstack((eq_id_all, sta_id_all, X_eq_all.T, X_stat_all.T,res_flatfile.zoneUTM.values,
                             y_new_mu,    res_tot,    res_between,    res_within, 
                             y_new_mu_cm, res_tot_cm, res_between_cm, res_within_cm)).T
columns_names = ['eq_idx','sta_idx','eqX','eqY','staX','staY','zoneUTM',
                 'med_p','res_tot','res_between','res_within',
                 'med_p_cmodel','res_tot_cmodel','res_between_cmodel','res_within_cmodel']

df_predict_summary = pd.DataFrame(predict_summary, columns = columns_names)
df_predict_summary[['eq_idx','sta_idx']] = df_predict_summary[['eq_idx','sta_idx']].astype(int)
df_predict_summary = pd.concat([res_flatfile, df_predict_summary], axis=1)
df_predict_summary.to_csv(dir_out  + fname_analysis + '_stan_residuals' + '.csv', index=False)

del eq_id_all, sta_id_all, data_train

## Summary regression
# ---------------------------
#save summary statistics
fname_summary = dir_out  + fname_analysis + '_stan_summary' + '.txt'
with open(fname_summary, 'w') as f:
    print(fit_full, file=f)

#create and save trace plots
dir_out_summ_fig = dir_out  + 'summary_figs/'
#create figures directory if doesn't exit
if not os.path.isdir(dir_out_summ_fig): pathlib.Path(dir_out_summ_fig).mkdir(parents=True, exist_ok=True) 

#create stan trace plots
for t_name, t_name2 in zip(trace_names, trace_names2):
    fig = fit_full.traceplot(t_name2)
    fig.savefig(dir_out_summ_fig + fname_analysis + '_stan_traceplot' + t_name + '.png')
    #create trace plot with arviz
    ax = az.plot_trace(fit_full,  var_names=t_name2, figsize=(10,5) ).ravel()
    ax[0].yaxis.set_major_locator(plt_autotick())
    ax[0].set_xlabel('sample value')
    ax[0].set_ylabel('frequency')
    ax[0].set_title('')
    ax[0].grid(axis='both')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('sample value')
    ax[1].grid(axis='both')
    ax[1].set_title('')
    fig = ax[0].figure
    fig.suptitle(t_name)
    fig.savefig(dir_out_summ_fig + fname_analysis + '_arviz_stan_traceplot' + t_name + '.png')

