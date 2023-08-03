#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:35:27 2021

@author: glavrent
"""

#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#gmm libraries
import pygmm
#user-derfined functions
import pylib.pylib_gmm_eas as pylib_gmm_eas
import pylib.pylib_rvt as pylib_rvt

def ConvertPandasDf2NpArray(df_array):
    
    return df_array.values if isinstance(df_array, pd.DataFrame) or isinstance(df_array, pd.Series) else df_array

class NonErgPSAGMM(object):
    
    def __init__(self, per, c0, phi_0m1, phi_0m2, tau_0m1, tau_0m2,
                       mag_brk_phi, mag_brk_tau,
                       ErgEASGMM, NonErgEASGMM, ErgPSAGMM):
     
        #convert to numpy arrays if pandas arrays
        per     = ConvertPandasDf2NpArray(per)
        c0      = ConvertPandasDf2NpArray(c0)
        phi_0m1 = ConvertPandasDf2NpArray(phi_0m1)
        phi_0m2 = ConvertPandasDf2NpArray(phi_0m2)
        tau_0m1 = ConvertPandasDf2NpArray(tau_0m1)
        tau_0m2 = ConvertPandasDf2NpArray(tau_0m2)
        mag_brk_phi = ConvertPandasDf2NpArray(mag_brk_phi)
        mag_brk_tau = ConvertPandasDf2NpArray(mag_brk_tau)

        #psa periods
        self.psa_per = per
        #constant coefficient
        self.c0 = c0
        #phi0 and tau0 breaks
        self.phi_0m1 = phi_0m1
        self.phi_0m2 = phi_0m2
        self.tau_0m1 = tau_0m1
        self.tau_0m2 = tau_0m2
        #magnitude brakes
        self.mag_brk_phi = mag_brk_phi
        self.mag_brk_tau = mag_brk_tau

        # EAS GMM        
        self.ErgEASGMM    = ErgEASGMM
        self.NonErgEASGMM = NonErgEASGMM
        # PSa GMM
        self.ErgPSAGMM = ErgPSAGMM



    def CalcNonErgPSa(self, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                            mag, dip, sof, width, ztor, 
                            rrup, rrjb, rx, ry0,
                            vs30, z10=None, z25=None, regid=1,
                            flag_as=None, flag_hw=None, crjb=None,
                            ssn=None, 
                            nsamp=0, flag_samp_unqloc=True, flag_ifreq=True, flag_sloc=False,
                            fmin_r=0.5, fmax_r=1.5,
                            flag_include_c0NS=True,
                            flag_flatten=True, 
                            PF='BT15', sdur=0.85, per=None):


        #number of points
        npt = len(mag)
        
        #psa perios
        if per is None: per = self.psa_per.copy()
    
        # Non-ergodic Factor
        # ----   ----  ----  ----  ----
        per, rpsa_nerg, _, _, freq_eas, reas_nerg, eas_erg, eas_nerg = self.CalcNonErgRatios(eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                                                                                             mag, sof, ztor, rrup, vs30, z10, regid,
                                                                                             ssn, 
                                                                                             nsamp, flag_samp_unqloc, flag_ifreq, flag_sloc,
                                                                                             fmin_r, fmax_r,
                                                                                             flag_include_c0NS,
                                                                                             flag_psa_const = True,
                                                                                             PF = PF, sdur = sdur, per = per)
      
        # PSa
        # ----   ----  ----  ----  ----
        #ergodic PSa
        per, psa_erg = self.CalcErgPSa(mag, dip, sof, width, ztor, 
                                       rrup, rrjb, rx, ry0,
                                       vs30, z10, z25, regid,
                                       flag_as, flag_hw, crjb,
                                       per)
        #non-ergodic PSa
        psa_nerg = (psa_erg[:,:,np.newaxis] if nsamp > 0 else psa_erg) * np.exp(rpsa_nerg)
        
        # Aleatory Std
        # ----   ----  ----  ----  ----
        tau0 = self.GetTau0(mag, per)
        phi0 = self.GetTau0(mag, per)  
        psa_sig = np.sqrt(tau0**2 + phi0**2)
        # Summarize Data
        # ----   ----  ----  ----  ----
        #flatten arrays if only one datapoint
        if psa_nerg.shape[0] == 1 and flag_flatten:
            import pdb; pdb.set_trace()
            psa_nerg  = psa_nerg[0,:].T
            rpsa_nerg = rpsa_nerg[0,:].T
            psa_sig = psa_sig
            eas_nerg = eas_nerg
            reas_nerg = reas_nerg
            
        return per, psa_nerg, psa_sig, psa_erg, rpsa_nerg, freq_eas, eas_nerg, eas_erg, reas_nerg
    
    # Ergodic PSA
    #-------------------------------     
    def CalcErgPSa(self, mag, dip, sof, width, ztor, 
                         rrup, rrjb, rx, ry0,
                         vs30, z10=None, z25=None, regid = 1,
                         flag_as=None, flag_hw=None, crjb=None,
                         per=None):

        #number of points
        npt = len(mag)
        #psa perios
        if per is None: per = self.psa_per.copy()
        
        #inialize input
        region = 'california' if regid == 1 else 'global'
        z10     = np.full(mag.shape, np.nan) if z10     is None else z10
        z25     = np.full(mag.shape, np.nan) if z25     is None else z25
        rx      = np.full(mag.shape, np.nan) if rx      is None else rx
        ry0     = np.full(mag.shape, np.nan) if ry0     is None else ry0
        if crjb is None: 
            assert(flag_as is None),'Error. Inconsistent crjb and flag_as'
            crjb = np.full(mag.shape, np.nan)
        flag_as = np.full(mag.shape, False)  if flag_as  is None else flag_as
        flag_hw = np.full(mag.shape, False)  if flag_hw is None else flag_hw
        
        #evaulate base PSa model 
        psa_erg = np.full([npt,len(per)], np.nan)
        for j in range(npt):
            #recording info
            mech  = 'SS' if np.abs(sof[j]) < 0.49 else ('RS' if sof[j] > 0 else 'NS')
            
            #define scenario GMM
            if   not (np.isnan(z10[j]) or np.isnan(z25[j])):
                gmm_scen = pygmm.model.Scenario(mag=mag[j], dip=dip[j], mechanism=mech, width=width[j], depth_tor=ztor[j],
                                                is_aftershock=flag_as[j],
                                                dist_rup=rrup[j], dist_jb=rrjb[j], dist_crjb=crjb[j],
                                                dist_x=rx[j], dist_y0=ry0[j], 
                                                v_s30=vs30[j], depth_1_0=z10[j], depth_2_5=z25[j], 
                                                on_hanging_wall=flag_hw[j], region=region )
            elif not np.isnan(z10[j]):
                gmm_scen = pygmm.model.Scenario(mag=mag[j], dip=dip[j], mechanism=mech, width=width[j], depth_tor=ztor[j],
                                                is_aftershock=flag_as[j],
                                                dist_rup=rrup[j], dist_jb=rrjb[j], dist_crjb=crjb[j],
                                                dist_x=rx[j], dist_y0=ry0[j], 
                                                v_s30=vs30[j], depth_1_0=z10[j],
                                                on_hanging_wall=flag_hw[j], region=region )
            elif not np.isnan(z25[j]):
                gmm_scen = pygmm.model.Scenario(mag=mag[j], dip=dip[j], mechanism=mech, width=width[j], depth_tor=ztor[j],
                                                is_aftershock=flag_as[j],
                                                dist_rup=rrup[j], dist_jb=rrjb[j], dist_crjb=crjb[j],
                                                dist_x=rx[j], dist_y0=ry0[j], 
                                                v_s30=vs30[j], depth_2_5=z25[j],
                                                on_hanging_wall=flag_hw[j], region=region )
            else:
                gmm_scen = pygmm.model.Scenario(mag=mag[j], dip=dip[j], mechanism=mech, width=width[j], depth_tor=ztor[j],
                                                is_aftershock=flag_as[j],
                                                dist_rup=rrup[j], dist_jb=rrjb[j], dist_crjb=crjb[j],
                                                dist_x=rx[j], dist_y0=ry0[j], 
                                                v_s30=vs30[j], 
                                                on_hanging_wall=flag_hw[j], region=region )

            #evaluate PSa
            psa_erg[j,:] = self.ErgPSAGMM(gmm_scen).interp_spec_accels(periods=per)
            assert(not np.isnan(psa_erg[j,:]).any()),'Error. nan in PSa evaluations in %i record'%j

        return per, psa_erg

    # PSA and EAS Non-ergodic factors
    #------------------------------- 
    def CalcNonErgRatios(self, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                               mag, sof, ztor, rrup, vs30, z10=None, regid = 1,
                               ssn = None, 
                               nsamp = 0, flag_samp_unqloc = True, flag_ifreq = True, flag_sloc = False,
                               fmin_r=0.5, fmax_r=1.5,
                               flag_include_c0NS = True, 
                               flag_psa_const = True,
                               PF = 'BT15', sdur = 0.85, per = None):

        #number of points
        npt = len(mag)

        #psa perios
        if per is None: per = self.psa_per.copy()

        # EAS
        # ----   ----  ----  ----  ----
        fnorm = np.abs( [max(-1*sof[j],0) for j in range(npt)] )
        #evalute ergodic EAS GMM
        freq_eas_erg, eas_erg = self.ErgEASGMM.Eas(mag, rrup, vs30, ztor, fnorm, z10, regid)[:2]
        #evalute non-ergodic EAS GMM
        freq_eas_nerg, eas_nerg, _, _, rrup2 = self.NonErgEASGMM.Eas(eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                                                                     mag, vs30, ztor, fnorm, z10, regid,
                                                                     ssn, 
                                                                     nsamp, flag_samp_unqloc, flag_ifreq, flag_sloc,
                                                                     fmin_r, fmax_r,
                                                                     flag_include_c0NS, flag_flatten=False)
        
        #EAS frequencies
        assert((np.abs(freq_eas_erg-freq_eas_nerg)<1e-9).all()),'Error. Inconsistent ergodic and non-ergodic EAS frequencies'
        freq_eas = freq_eas_erg
        #check consistency of coordinates and rupture distance
        assert((np.abs(rrup-rrup2)<5e-2).all()),'Error. Inconsistent coordinates and rupture distance'
        
        #non-ergodic eas ratios
        reas_nerg =  np.log(eas_nerg) - np.log(eas_erg[:,:,np.newaxis] if nsamp > 0 else eas_erg)     
        
        # RVT
        # ----   ----  ----  ----  ----
        #calcualte psa through rvt
        psa_erg_rvt = pylib_rvt.CalcPSaRVTMultScen(1/per, freq_eas_erg, eas_erg, mag, rrup, vs30,
                                                   dur_interval=sdur, m_PF=PF)[1]
        if nsamp > 0:
            psa_nerg_rvt = np.full([npt,len(per), nsamp], np.nan)
            for j in range(nsamp):
                psa_nerg_rvt[:,:,j] = pylib_rvt.CalcPSaRVTMultScen(1/per, freq_eas_nerg, eas_nerg[:,:,j], mag, rrup, vs30,
                                                                    dur_interval=sdur, m_PF=PF)[1]
        else:
            psa_nerg_rvt = pylib_rvt.CalcPSaRVTMultScen(1/per, freq_eas_nerg, eas_nerg, mag, rrup, vs30,
                                                        dur_interval=sdur, m_PF=PF)[1]
        
        #intercept
        c0 = self.GetC0(per)
        #non-ergodic psa ratios
        rpsa_nerg = np.log(psa_nerg_rvt) - np.log(psa_erg_rvt[:,:,np.newaxis] if nsamp > 0 else psa_erg_rvt) 
        if flag_psa_const:
            if nsamp > 0: rpsa_nerg += c0[np.newaxis,:,np.newaxis]
            else:         rpsa_nerg += c0[np.newaxis,:]
        
        
        return per, rpsa_nerg, psa_erg_rvt, psa_nerg_rvt, freq_eas, reas_nerg, eas_erg, eas_nerg

    # Non-ergodic PSA constant and aleatory terms
    #-------------------------------
    def GetC0(self, per):
        
        return np.interp(np.log(per), np.log(self.psa_per), self.c0)
     
    def GetTau0(self, mag, per):
    
        tau_0m1 = np.interp(np.log(per), np.log(self.psa_per), self.tau_0m1)
        tau_0m2 = np.interp(np.log(per), np.log(self.psa_per), self.tau_0m2)
    
        return np.vstack( [np.interp(mag, self.mag_brk_tau, [t_0m1, t_0m2]) for t_0m1, t_0m2 in zip(tau_0m1, tau_0m2)] ).T
    
    def GetPhi0(self, mag, per):
    
        phi_0m1 = np.interp(np.log(per), np.log(self.psa_per), self.phi_0m1)
        phi_0m2 = np.interp(np.log(per), np.log(self.psa_per), self.phi_0m2)

        return np.hstack( [np.interp(mag, self.mag_brk_phi, [p_0m1, p_0m2]) for p_0m1, p_0m2 in zip(phi_0m1, phi_0m2)] ).T
   
    def GetSig0(self, mag, per):     
    
        tau0 = self.GetTau0(mag, per)
        phi0 = self.GetTau0(mag, per)  
        
        return np.sqrt(tau0**2 + phi0**2)
    