# ba18.py
# Conversion of Jeff Bayless' MATLAB code to Python
# Including class ba18
# I've tried to avoid mixed UPPER and lower case variable names
#   e.g. Mbreak, Rrup, Vsref 

#arithmetic libraries
import numpy as np
import numpy.matlib
from scipy import linalg as scipylalg
from scipy import sparse as scipysp
from sksparse.cholmod import cholesky as spchol
#geographic coordinates
import pyproj
#statistics libraries
import pandas as pd
#geometric libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly
#user-derfined functions
import pylib.pylib_cell_dist as pycelldist
import pylib.pylib_GP_model  as pygp
import pylib.pylib_stats     as pystats

def SlicingSparceMat(mat_sp, i_rows, j_col):
    '''Slice sparse matrix'''

    return np.array([mat_sp.getcol(i_r).toarray().flatten()[j_col] for i_r in i_rows])  

def QuartCos(per, x0, x, flag_left = False):
    
    y = np.cos( 2.*np.pi*(x-x0)/per )
    if flag_left: y[np.logical_or(x < x0-per/4, x > x0)]     = 0.
    else:         y[np.logical_or(x < x0,       x > x0+per/4)] = 0.
    
    return y

def QuadCosTapper(freq, freq_nerg):
    
    #boxcar at intermediate frequencies
    i_box = np.logical_and(freq >= freq_nerg.min(), freq <= freq_nerg.max()) 
    y_box        = np.zeros(len(freq))
    y_box[i_box] = 1.
    #quarter cosine left taper
    per   = 2 * freq_nerg.min()
    y_tpl = QuartCos(per, freq_nerg.min(), freq, flag_left=True)
    #quarter cosine right taper
    per   = 2 * freq_nerg.max()
    y_tpr = QuartCos(per, freq_nerg.max(), freq)
    #combined tapering function
    y_tapper = np.array([y_box, y_tpl, y_tpr]).max(axis=0)

    return y_tapper

def TriagTapper(freq, freq_nerg):
    
    fn_min = freq_nerg.min()
    fn_max = freq_nerg.max()
    
    #triangular window
    f_win = np.array([0.5*fn_min, fn_min, fn_max, 1.5*fn_max])
    y_win = np.array([0.,          1.,     1.,     0.])
    
    #triangular tapering function
    y_tapper = np.interp(np.log(freq), np.log(f_win), y_win)

    return y_tapper

def ConvertPandasDf2NpArray(df_array):
    
    array = df_array.values if isinstance(df_array, pd.DataFrame) or isinstance(df_array, pd.Series) else df_array
    
    return array

    

class BA18:
    def __init__(self, file=None):
        '''
        Constructor for this class
        Read CSV file of BA18 coefficients, frequency range: 0.1 - 100 Hz

        Parameters
        ----------
        file : string, optional
            file name for coefficients. The default is None.
        '''

        if file is None:
                #file = '/mnt/halcloud_nfs/glavrent/Research/NonErgodic_gmpes/Analyses/Python_lib/ground_motions/Bayless_ModelCoefs.csv'
                file = '/mnt/halcloud_nfs/glavrent/Research/Nonerg_CA_GMM/Analyses/Python_lib/ground_motions/Bayless_ModelCoefs.csv'
        df = pd.read_csv(file, index_col=0)
        df = df.head(301)
        # Frequencies 0.1 - 24 Hz
        self.freq = df.index.values
        # Median FAS parameters
        self.b1 = df.c1.values
        self.b2 = df.c2.values
        self.b3quantity = df['(c2-c3)/cn'].values
        self.b3 = df.c3.values
        self.bn = df.cn.values
        self.bm = df.cM .values
        self.b4 = df.c4.values
        self.b5 = df.c5.values
        self.b6 = df.c6.values
        self.bhm = df.chm.values
        self.b7 = df.c7.values
        self.b8 = df.c8.values
        self.b9 = df.c9.values
        self.b10 = df.c10.values
        self.b11a = df.c11a.values 
        self.b11b = df.c11b.values
        self.b11c = df.c11c.values
        self.b11d = df.c11d.values
        self.b1a = df.c1a.values
        self.b1a[239:] = 0
        # Non-linear site parameters
        self.f3 = df.f3.values
        self.f4 = df.f4.values
        self.f5 = df.f5.values
        # Aleatory variability parameters
        self.s1 = df.s1.values
        self.s2 = df.s2.values
        self.s3 = df.s3.values
        self.s4 = df.s4.values
        self.s5 = df.s5.values
        self.s6 = df.s6.values
        # Constants
        self.b4a = -0.5
        self.vsref = 1000
        self.mbreak = 6.0
        #bedrock anelastic attenuation
        self.b7rock = self.b7.copy()
        #frequency limits
        # self.maxfreq = 23.988321
        self.maxfreq = self.freq.max()
        self.minfreq = self.freq.min()
    
    def EasBase(self, mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7 = True):
                        
        # note Z1 must be provided in km
        z1ref = (1/1000) * np.exp(-7.67/4 * np.log((vs30**4+610**4)/(1360**4+610**4)) )
        if vs30<=200:
            self.b11 = self.b11a
        if vs30>200 and vs30<=300:
            self.b11 = self.b11b
        if vs30>300 and vs30<=500:
            self.b11 = self.b11c
        if vs30>500:
            self.b11 = self.b11d
    
        if z1 is None or np.isnan(z1):
            z1 = self.Z1(vs30, regid=1)
    
        # Compute lnFAS by summing contributions, including linear site response
        lnfas = self.b1 + self.b2*(mag-self.mbreak)
        lnfas += self.b3quantity*np.log(1+np.exp(self.bn*(self.bm-mag))) 
        lnfas += self.b4*np.log(rrup+self.b5*np.cosh(self.b6*np.maximum(mag-self.bhm,0)))
        lnfas += (self.b4a-self.b4) * np.log( np.sqrt(rrup**2+50**2) ) 
        lnfas += self.b7 * rrup if flag_keep_b7 else 0.
        lnfas += self.b8 * np.log( min(vs30,1000) / self.vsref ) 
        lnfas += self.b9 * min(ztor,20) 
        lnfas += self.b10 * fnorm 
        lnfas += self.b11 * np.log( (min(z1,2) + 0.01) / (z1ref + 0.01) )
        # this is the linear spectrum up to maxfreq=23.988321 Hz
        maxfreq = 23.988321
        imax = np.where(self.freq==maxfreq)[0][0]
        fas_lin = np.exp(lnfas)
        # Extrapolate to 100 Hz
        fas_maxfreq = fas_lin[imax]
        # Kappa
        kappa = np.exp(-0.4*np.log(vs30/760)-3.5)
        # Diminuition operator
        D = np.exp(-np.pi*kappa*(self.freq[imax:] - maxfreq))
        fas_lin = np.append(fas_lin[:imax], fas_maxfreq * D)
        
        # Compute non-linear site response
        # get the EAS_rock at 5 Hz (no c8, c11 terms)
        vref=760
        #row = df.iloc[df.index == 5.011872]
        i5 = np.where(self.freq==5.011872)
        lnfasrock5Hz = self.b1[i5]
        lnfasrock5Hz += self.b2[i5]*(mag-self.mbreak) 
        lnfasrock5Hz += self.b3quantity[i5]*np.log(1+np.exp(self.bn[i5]*(self.bm[i5]-mag))) 
        lnfasrock5Hz += self.b4[i5]*np.log(rrup+self.b5[i5]*np.cosh(self.b6[i5]*max(mag-self.bhm[i5],0)))
        lnfasrock5Hz += (self.b4a-self.b4[i5])*np.log(np.sqrt(rrup**2+50**2)) 
        lnfasrock5Hz += self.b7rock[i5]*rrup 
        lnfasrock5Hz += self.b9[i5]*min(ztor,20) 
        lnfasrock5Hz += self.b10[i5]*fnorm
        # Compute PGA_rock extimate from 5 Hz FAS
        IR = np.exp(1.238+0.846*lnfasrock5Hz)
        # apply the modified Hashash model
        self.f2 = self.f4*( np.exp(self.f5*(min(vs30,vref)-360)) - np.exp(self.f5*(vref-360)) )
        fnl0 = self.f2 * np.log((IR+self.f3)/self.f3)
        fnl0[np.where(fnl0==min(fnl0))[0][0]:] = min(fnl0)
        fas_nlin = np.exp( np.log(fas_lin) + fnl0 )
        
        # Aleatory variability
        if mag<4:
            tau = self.s1
            phi_s2s = self.s3
            phi_ss = self.s5
        if mag>6:
            tau = self.s2
            phi_s2s = self.s4
            phi_ss = self.s6
        if mag >= 4 and mag <= 6:
            tau = self.s1 + ((self.s2-self.s1)/2)*(mag-4)
            phi_s2s = self.s3 + ((self.s4-self.s3)/2)*(mag-4)
            phi_ss = self.s5 + ((self.s6-self.s5)/2)*(mag-4)
        sigma = np.sqrt(tau**2 + phi_s2s**2 + phi_ss**2 + self.b1a**2);
        
        return self.freq, fas_nlin, fas_lin, sigma

    def EasBaseArray(self, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7=True):
        
        #convert eq parameters to np.arrays   
        mag   = np.array([mag]).flatten()
        rrup  = np.array([rrup]).flatten()
        vs30  = np.array([vs30]).flatten()
        ztor  = np.array([ztor]).flatten()
        fnorm = np.array([fnorm]).flatten()
        z1    = np.array([self.Z1(vs, regid) for vs in vs30]) if z1 is None else np.array([z1]).flatten()
        
        #number of scenarios
        npt = len(mag)
        #input assertions
        assert( np.all(npt == np.array([len(rrup),len(vs30),len(ztor),len(fnorm),len(z1)])) ),'Error. Inconsistent number of gmm parameters'
        
        #compute fas for all scenarios
        fas_nlin = list()
        fas_lin  = list()
        sigma    = list()
        for k, (m, r, vs, zt, fn, z_1) in enumerate(zip(mag, rrup, vs30, ztor, fnorm, z1)):
            ba18_base = self.EasBase(m, r, vs, zt, fn, z_1, regid, flag_keep_b7)[1:]
            fas_nlin.append(ba18_base[0])
            fas_lin.append(ba18_base[1])
            sigma.append(ba18_base[2])
        #combine them to np.arrays
        fas_nlin = np.vstack(fas_nlin)
        fas_lin  = np.vstack(fas_lin)
        sigma    = np.vstack(sigma)
        
        # if npt == 1 and flag_flatten:
        #    fas_nlin = fas_nlin.flatten()
        #    fas_lin  = fas_lin.flatten()
        #    sigma    = sigma.flatten() 
        
        #return self.EasBase(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        return self.freq, fas_nlin, fas_lin, sigma    

    def Eas(self, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7=True, flag_flatten=True):
        '''
        Computes BA18 EAS GMM for all frequencies

        Parameters
        ----------
        mag : real
            moment magnitude [3-8].
        rrup : real
            Rupture distance in kilometers (km) [0-300].
        vs30 : real
            site-specific Vs30 = slowness-averaged shear wavespeed of upper 30 m (m/s) [120-1500].
        ztor : real
            depth to top of rupture (km) [0-20].
        fnorm : real
            1 for normal faults and 0 for all other faulting types (no units) [0 or 1].
        z1 : real, optional
            site-specific depth to shear wavespeed of 1 km/s (km) [0-2]. The default is =None.
        regid : int, optional
            DESCRIPTION. The default is =1.

        Returns
        -------
        freq : np.array
            frequency array.
        fas_nlin : np.array
            fas array with nonlinear site response.
        fas_lin : np.array
            fas array with linear site response.
        sigma : np.array
            standard deviation array.
        '''
        
        #return self.EasBase(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        # return self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7, flag_flatten)
        
        freq, fas_nlin, fas_lin, sigma = self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        
        #flatten arrays if only one datapoint
        if fas_nlin.shape[0] == 1 and flag_flatten:
            fas_nlin = fas_nlin.flatten()
            fas_lin  = fas_lin.flatten()
            sigma    = sigma.flatten()         
        
        return freq, fas_nlin, fas_lin, sigma 

    
    def EasF(self, freq, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7 = True, flag_flatten=True):
        '''
        Computes BA18 EAS GMM for frequency of interest

        Parameters
        ----------
        mag : real
            moment magnitude [3-8].
        rrup : real
            Rupture distance in kilometers (km) [0-300].
        vs30 : real
            site-specific Vs30 = slowness-averaged shear wavespeed of upper 30 m (m/s) [120-1500].
        ztor : real
            depth to top of rupture (km) [0-20].
        fnorm : real
            1 for normal faults and 0 for all other faulting types (no units) [0 or 1].
        z1 : real, optional
            site-specific depth to shear wavespeed of 1 km/s (km) [0-2]. The default is =None.
        regid : int, optional
            DESCRIPTION. The default is =1.

        Returns
        -------
        freq : real
            frequency of interest.
        fas_nlin : real
            fas with nonlinear site response for frequency of interest.
        fas_lin : real
            fas with linear site response for frequency of interest.
        sigma : real
            standard deviation of frequency of interest.
        '''
        
        #convert freq to numpy array
        freq = np.array([freq]).flatten()
        
        #frequency tolerance
        f_tol = 1e-4
        #compute fas for all frequencies
        freq_all, fas_all, fas_lin_all, sig_all = self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        
        #find eas for frequency of interest
        if np.all([np.isclose(f, freq_all, f_tol).any() for f in freq]):
            # i_f     = np.array([np.where(np.isclose(f, freq_all, f_tol))[0] for f in freq]).flatten()
            i_f     = np.array([np.argmin(np.abs(f-freq_all)) for f in freq]).flatten()
            freq    = freq_all[i_f]
            fas     = fas_all[:,i_f]
            fas_lin = fas_lin_all[:,i_f]
            sigma   = sig_all[:,i_f]
        else:
            fas     = np.vstack([np.exp(np.interp(np.log(np.abs(freq)), np.log(freq_all), np.log(fas),   left=-np.nan, right=-np.nan)) for fas   in fas_all])
            fas_lin = np.vstack([np.exp(np.interp(np.log(np.abs(freq)), np.log(freq_all), np.log(fas_l), left=-np.nan, right=-np.nan)) for fas_l in fas_lin_all])
            sigma   = np.vstack([       np.interp(np.log(np.abs(freq)), np.log(freq_all), sig,           left=-np.nan, right=-np.nan)  for sig   in sig_all])

        #if one scenario flatten arrays        
        if fas.shape[0] == 1 and flag_flatten:
            fas     = fas.flatten()
            fas_lin = fas_lin.flatten()
            sigma   = sigma.flatten()

        return fas, fas_lin, sigma
    
    def GetFreq(self):
        
        return np.array(self.freq)
    
    def Z1(self, vs30, regid=1):
        '''
        Compute Z1.0 based on Vs30 for CA and JP

        Parameters
        ----------
        vs30 : real
            Time average shear-wave velocity.
        regid : int, optional
            Region ID. The default is 1.

        Returns
        -------
        real
            Depth to a shear wave velocity of 1000m/sec.
        '''
        
        if regid == 1:    #CA
            z_1 = -7.67/4. * np.log((vs30**4+610.**4)/(1360.**4+610.**4))        
        elif regid == 10: #JP
            z_1 = -5.23/4. * np.log((vs30**4+412.**4)/(1360.**4+412.**4))
    
        return 1/1000*np.exp(z_1)
    
#%% Non-ergodic GMM
### -------------------------------------------------------------------------
class NonErgEASGMMCoeffBase(BA18):
    
    def __init__(self, eqid=None, ssn=None, cA_id = None, s_Xutm = None,
                 zone_utm=None, grid_Xutm=None, c1a_Xutm = None, c1b_Xutm = None, cA_Xutm = None, cAmpt_Xutm = None,
                 fname_ba18_coeffs=None, flg_sparse_cov = True, sp_tol=1e-3):
        
        
        #if ids and coordinates are pandas data-frame convert to numpy arrays
        eqid       = ConvertPandasDf2NpArray(eqid)
        ssn        = ConvertPandasDf2NpArray(ssn)
        s_Xutm     = ConvertPandasDf2NpArray(s_Xutm)
        cA_id      = ConvertPandasDf2NpArray(cA_id)
        grid_Xutm  = ConvertPandasDf2NpArray(grid_Xutm)
        c1a_Xutm   = ConvertPandasDf2NpArray(c1a_Xutm)
        c1b_Xutm   = ConvertPandasDf2NpArray(c1b_Xutm)
        cA_Xutm    = ConvertPandasDf2NpArray(cA_Xutm)
        cAmpt_Xutm = ConvertPandasDf2NpArray(cAmpt_Xutm)
        
        
        #initialize object fields
        self.delta     = 1e-5
        self.delta_corr = 5e-2
        self.ftol      = 1e-4
        self.Xtol      = 1e-3
        self.ssn_Xtol  = 0.2 #km
        self.nerg_freq = []
        #spatially varying coefficients        
        self.c1a_mu  = {}
        self.c1a_cov = {}
        self.c1b_mu  = {}
        self.c1b_cov = {}
        #delta S2S info
        self.ssn      = ssn
        self.dS2S_mu  = {}
        self.dS2S_sig = {}
        #anelastic cell attenuation
        self.cell_X  = cA_Xutm
        self.cmpt_X  = cAmpt_Xutm
        self.cell_id = cA_id 
        #anelastic coefficients
        self.cA_mu  = {}
        self.cA_cov = {}
        #aleatory terms
        self.eqid    = eqid
        self.dBe_mu  = {}
        self.dBe_sig = {}       
        #hyper-parameters
        self.hyp = pd.DataFrame(columns=['c0', 'c0_N', 'c0_S', 
                                         'rho_c1a', 'theta_c1a', 'rho_c1b', 'theta_c1b', 
                                         'mu_cA', 'rho_cA', 'theta_cA', 'sigma_cA', 'pi_cA',
                                         'phi_S2S', 'tau_0', 'phi_0'])
        self.hyp_ifreq = pd.DataFrame(columns=['A', 'B', 'C', 'D'], index=['c1a','c1b','dS2S','cA','aleat'])

        #spatial info
        self.zone_utm  = zone_utm
        #import pdb; pdb.set_trace()
        #if not (self.zone_utm is None): self.proj_utm  = pyproj.Proj("+proj=utm +zone="+zone_utm+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        if not (self.zone_utm is None): self.proj_utm  = pyproj.Proj("+proj=utm +zone="+zone_utm+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        #grid coordinates for spatially varying coefficients
        self.grid_X = grid_Xutm
        #coefficient coordinates if not specified on grid
        if not c1a_Xutm is None:
            assert(grid_Xutm is None),'Error. c1a_Xutm and grid_Xutm cannot be specified simultaneously'
            self.c1a_mu  = pd.DataFrame({'x':c1a_Xutm[:,0], 'y':c1a_Xutm[:,1]})
            self.c1a_cov = pd.DataFrame({'x':c1a_Xutm[:,0], 'y':c1a_Xutm[:,1]})
        if not c1b_Xutm is None:
            assert(grid_Xutm is None),'Error. c1b_Xutm and grid_Xutm cannot be specified simultaneously'
            self.c1b_mu  = pd.DataFrame({'x':c1a_Xutm[:,0], 'y':c1a_Xutm[:,1]})
            self.c1b_cov = pd.DataFrame({'x':c1a_Xutm[:,0], 'y':c1a_Xutm[:,1]})
        if not s_Xutm is None:
            assert(len(ssn) == s_Xutm.shape[0]),'Error. Inconsistent size of ssn and sXutm'
            self.site_info = pd.DataFrame({'id':ssn, 'x':s_Xutm[:,0], 'y':s_Xutm[:,1]})
        else:
            self.s_info = None            
        #read ergodic coefficients from file
        super().__init__(fname_ba18_coeffs)
        #sparse covariance options
        self.sp_cov = flg_sparse_cov
        self.sp_tol = sp_tol
        #sub-region
        self.sregN_latlon = np.array([[34.5175, -121.5251],[39.8384, -125.2341],[41.3595, -124.1684],[41.3995,-120.7227],[37.9773,-116.6227]])
        self.sregS_latlon = np.array([[37.9773, -116.6226],[35.2944, -113.4142],[31.4772, -115.0250],[31.0082,-117.6898],[34.5173,-121.5249]])
        self.sregN_poly = None
        self.sregS_poly = None
        self.sreg_mag   = 5
        
    def SampCoeffF(self, freq, mag, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                   ssn=None, eqid=None,
                   nsamp = 0, flag_samp_unqloc = False, flag_ifreq = False, flag_sloc = False):

        #convert freq to numpy array
        freq = np.array([freq]).flatten()

        #if coordinates are pandas data-frame convert to numpy arrays
        eqid          = ConvertPandasDf2NpArray(eqid)
        ssn           = ConvertPandasDf2NpArray(ssn)
        eq_latlon     = ConvertPandasDf2NpArray(eq_latlon)
        eqclst_latlon = ConvertPandasDf2NpArray(eqclst_latlon)
        sta_latlon    = ConvertPandasDf2NpArray(sta_latlon)
        eqclst_z      = ConvertPandasDf2NpArray(eqclst_z)
        
        #convert eq and sta latlon to two dim array 
        if eq_latlon.ndim     == 1: eq_latlon     = eq_latlon.reshape(1,2)
        if eqclst_latlon.ndim == 1: eqclst_latlon = eqclst_latlon.reshape(1,2)
        if sta_latlon.ndim    == 1: sta_latlon    = sta_latlon.reshape(1,2)
        
        #number of points
        npt = len(eq_latlon)
        #number for frequencies
        nf  = len(freq)
        
        #assign -999 if eqid or ssn are unspecified
        if eqid is None: 
            eqid = -1 - np.arange(npt)
        else:
            eqid = ConvertPandasDf2NpArray(eqid)
            eqid[np.isnan(eqid)] = -1 - np.arange(np.isnan(eqid).sum())
        if ssn is None:
            ssn  = -1 - np.arange(npt)
        else:
            ssn = ConvertPandasDf2NpArray(ssn)
            ssn[np.isnan(ssn)] = -1 - np.arange(np.isnan(ssn).sum())
        
        #convert eqid and ssn to np arrays        
        ssn  = np.array([ssn]).flatten()
        eqid = np.array([eqid]).flatten()
        #ssn index to nan
        ssn[np.isnan(ssn)] = -1 * np.arange(np.isnan(ssn).sum())
        # convert eqid and ssn to integers
        eqid = eqid.astype(int)
        ssn  = ssn.astype(int)
        
        #input assertions
        assert( np.all(np.isin(freq, self.nerg_freq)) ), 'Error. Unavailable frequency'
        assert( np.all(npt == np.array([len(eq_latlon), len(sta_latlon), 
                                        len(ssn), len(eqid), len(eqclst_z)])) ),'Error. Inconsistent number of gmm parameters'
        
        #map earthquake and site at utm coordinates
        eq_X_all     = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(eq_latlon[:,0],      eq_latlon[:,1])])     / 1000
        eqclst_X_all = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(eqclst_latlon[:,0],  eqclst_latlon[:,1])]) / 1000
        sta_X_all    = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(sta_latlon[:,0],     sta_latlon[:,1])])    / 1000

        #update ssn based on site coordinates
        if flag_sloc:
            assert(not self.site_info is None),'Error. Site info is not defined'
            ssn = self.FindSSN(self.site_info, ssn, sta_X_all, self.ssn_Xtol)

        #if true sample coefficients only for unique locations
        if flag_samp_unqloc:
            eq_X,  eq_idx,   eq_inv   = np.unique(eq_X_all.round(2),  return_index=True, return_inverse=True, axis=0)
            sta_X, sta_idx,  sta_inv  = np.unique(sta_X_all.round(2), return_index=True, return_inverse=True, axis=0)
            eqid,  eqid_idx, eqid_inv = np.unique(eqid,               return_index=True, return_inverse=True, axis=0)
            ssn,   ssn_idx,  ssn_inv  = np.unique(ssn,                return_index=True, return_inverse=True, axis=0)
        else:
            #coordinates
            eq_X  = eq_X_all
            sta_X = sta_X_all
            #inverse indices
            eq_inv   = np.arange(npt)
            sta_inv  = np.arange(npt)
            eqid_inv = np.arange(npt)
            ssn_inv  = np.arange(npt)
        #number of coefficient for c1a, c1b and dS2S   
        nc1a  = len(eq_X)
        nc1b  = len(sta_X)
        neqid = len(eqid)
        ndS2S = len(ssn)
        
        #inverse indices of all coefficients for all frequencies
        eq_inv_allf   = np.hstack([eq_inv   + nc1a  * k for k in range(nf)])
        sta_inv_allf  = np.hstack([sta_inv  + nc1b  * k for k in range(nf)])
        eqid_inv_allf = np.hstack([eqid_inv + neqid * k for k in range(nf)])
        ssn_inv_allf  = np.hstack([ssn_inv  + ndS2S * k for k in range(nf)])
        
        #constants
        c0_mu   = self.Getc0(freq, npt, flag_consol=False)
        c0NS_mu = self.Getc0NS(freq, mag, eq_X_all, flag_consol=False)
        #sample spatialy varying coefficients
        #c1a
        c1a_mu, c1a_cov = self.Getc1a(freq, eq_X,  flag_consol=False)
        #c1b
        c1b_mu, c1b_cov = self.Getc1b(freq, sta_X, flag_consol=False)
        #station terms
        dS2S_mu,   dS2S_cov = self.GetdS2S(freq, ssn, flag_consol=False)
        #sample anelastic attenuation coefficients
        cA_mu,  cA_cov, _, L_c, nc  = self.GetcA(freq, eqclst_X_all, sta_X_all, eqclst_z, flag_consol=False)
        #cell paths for all frequencies
        L_cells = [L_c] * nf
        L_cells = scipysp.block_diag(L_cells, format='csc') if self.sp_cov else scipylalg.block_diag(*L_cells)

        #compute inter-frequency correlation of epistemic uncertainty terms
        if flag_ifreq:
            c1a_corr  = self.GetIfreqCorrc1a(freq,  1) 
            c1b_corr  = self.GetIfreqCorrc1b(freq,  1) 
            dS2S_corr = self.GetIfreqCorrdS2S(freq, 1) 
            cA_corr   = self.GetIfreqCorrcA(freq,   1)
        else:
            c1a_corr  = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            c1b_corr  = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            dS2S_corr = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            cA_corr   = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
        
        #rupture distance
        rrup = L_c.sum(axis=1)
        #add ranomization if number of samples > 0
        if nsamp:
            #sample spatially varying coefficients
            c0   = np.matlib.repmat(np.hstack(c0_mu),   nsamp,1).T
            c0NS = np.matlib.repmat(np.hstack(c0NS_mu), nsamp,1).T
            #earthquake constant
            c1a  = np.random.normal(size=[nc1a,nf,nsamp])  
            c1a  = np.array([pystats.MVNRnd(None, c1a_cov[k],  seed= c1a[:,k,:],                         flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            c1a  = np.array( pystats.MVNRnd(None, c1a_corr,    seed=[c1a[k,:,:]  for k in range(nc1a)],  flag_sp=self.sp_cov, flag_list=True) )
            #site constant
            c1b  = np.random.normal(size=[nc1b,nf,nsamp])
            c1b  = np.array([pystats.MVNRnd(None, c1b_cov[k],  seed= c1b[:,k,:],                         flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            c1b  = np.array( pystats.MVNRnd(None, c1b_corr,    seed=[c1b[k,:,:]  for k in range(nc1b)],  flag_sp=self.sp_cov, flag_list=True) )
            #site term
            dS2S = np.random.normal(size=[ndS2S,nf,nsamp])
            dS2S = np.array([pystats.MVNRnd(None, dS2S_cov[k], seed= dS2S[:,k,:],                        flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            dS2S = np.array( pystats.MVNRnd(None, dS2S_corr,   seed=[dS2S[k,:,:] for k in range(ndS2S)], flag_sp=self.sp_cov, flag_list=True) )
            #cell specific anelastic attenuation
            cA   = np.random.normal(size=[nc,nf,nsamp])
            cA   = np.array([pystats.MVNRnd(None, cA_cov[k],   seed= cA[:,k,:],                          flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            cA   = np.array( pystats.MVNRnd(None, cA_corr,     seed=[cA[k,:,:]   for k in range(nc)],    flag_sp=self.sp_cov, flag_list=True) )
            #correct random samples for unbiased mean
            c1a  = c1a  - c1a.mean(axis=2)[:, :, np.newaxis]
            c1b  = c1b  - c1b.mean(axis=2)[:, :, np.newaxis]
            dS2S = dS2S - dS2S.mean(axis=2)[:, :, np.newaxis]
            cA   = cA   - cA.mean(axis=2)[:, :, np.newaxis]
            #convert coeffs to two dimensional arrays 
            c1a  = np.vstack([c1a[:,k,:]  + c1a_mu[k][:,np.newaxis]  for k in range(nf)])
            c1b  = np.vstack([c1b[:,k,:]  + c1b_mu[k][:,np.newaxis]  for k in range(nf)])
            dS2S = np.vstack([dS2S[:,k,:] + dS2S_mu[k][:,np.newaxis] for k in range(nf)])
            cA   = np.vstack([cA[:,k,:]   + cA_mu[k][:,np.newaxis]   for k in range(nf)])
            #assign coefficinets for all recordings
            c1a  = c1a[eq_inv_allf,:]
            c1b  = c1b[sta_inv_allf,:]
            dS2S = dS2S[ssn_inv_allf,:]
        else:
            #assign coefficinets for all recordings
            c0   = c0_mu
            c0NS = c0NS_mu
            c1a  = c1a[eq_inv]
            c1b  = c1b[sta_inv]
            dS2S = dS2S[ssn_inv]
            cA   = cA_mu            

        return(c0, c0NS, c1a, c1b, dS2S, cA, rrup, L_cells)

    def SampCoeffAll(self, mag, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                     ssn=None, eqid=None,
                     nsamp = 0, flag_samp_unqloc = False, flag_ifreq = False,
                     flag_sloc = False,
                     fmin_r=0.5, fmax_r=1.5):

        #if coordinates are pandas data-frame convert to numpy arrays
        eq_latlon     = ConvertPandasDf2NpArray(eq_latlon)
        eqclst_latlon = ConvertPandasDf2NpArray(eqclst_latlon)
        sta_latlon    = ConvertPandasDf2NpArray(sta_latlon)
        eqclst_z      = ConvertPandasDf2NpArray(eqclst_z)
        
        #convert eq and sta latlon to two dim array 
        if eq_latlon.ndim     == 1: eq_latlon     = eq_latlon.reshape(1,2)
        if eqclst_latlon.ndim == 1: eqclst_latlon = eqclst_latlon.reshape(1,2)
        if sta_latlon.ndim    == 1: sta_latlon    = sta_latlon.reshape(1,2)
        
        #number of points
        npt = len(eq_latlon)
        #assign -999 if eqid or ssn are unspecified
        if eqid is None: 
            eqid = -1 - np.arange(npt)
        else:
            eqid = ConvertPandasDf2NpArray(eqid)
            eqid[np.isnan(eqid)] = -1 - np.arange(np.isnan(eqid).sum())
        if ssn is None:
            ssn  = -1 - np.arange(npt)
        else:
            ssn = ConvertPandasDf2NpArray(ssn)
            ssn[np.isnan(ssn)] = -1 - np.arange(np.isnan(ssn).sum())

        #convert eqid and ssn to np arrays        
        ssn  = np.array([ssn]).flatten()
        eqid = np.array([eqid]).flatten()
        #convert eqid and ssn to integers
        eqid = eqid.astype(int)
        ssn  = ssn.astype(int)
        
        #input assertions
        assert( np.all(npt == np.array([len(eq_latlon), len(sta_latlon), 
                                        len(ssn), len(eqid), len(eqclst_z)])) ),'Error. Inconsistent number of gmm parameters.'
        assert( fmin_r <=1 and fmax_r>=1 ),'Error. Invalid frequency ratios.'
        
        #map earthquake and site at utm coordinates
        eq_X_all     = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(eq_latlon[:,0],      eq_latlon[:,1])])     / 1000
        eqclst_X_all = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(eqclst_latlon[:,0],  eqclst_latlon[:,1])]) / 1000
        sta_X_all    = np.array([self.proj_utm(pt_lon, pt_lat) for pt_lat, pt_lon in zip(sta_latlon[:,0],     sta_latlon[:,1])])    / 1000

        #frequencies for GMM
        freq   = self.freq
        c7_erg = self.b7
        c7_erg[np.isnan(c7_erg)] = c7_erg[~np.isnan(c7_erg)][-1] #extend attenuation beyond 24hz
        nf     = len(freq)
        #non-ergodic frequencies info
        freq_nerg = np.array(self.nerg_freq).flatten()
        c_freq_nerg = [self.GetFreqName(f) for f in freq_nerg]
        c_fn_min    = self.GetFreqName(freq_nerg.min())
        c_fn_max    = self.GetFreqName(freq_nerg.max())
        #extened non-ergodic frequencies
        freq_min_ext = np.max([self.minfreq, fmin_r * freq_nerg.min() ])
        freq_max_ext = np.min([self.maxfreq, fmax_r * freq_nerg.max() ])
        freq_nerg_ext =  np.concatenate([ freq_nerg, [freq_min_ext, freq_max_ext] ])
        
        #update ssn based on site coordinates
        if flag_sloc:
            assert(not self.site_info is None),'Error. Site info is not defined'
            ssn = self.FindSSN(self.site_info, ssn, sta_X_all, self.ssn_Xtol)

        #if true sample coefficients only for unique locations
        if flag_samp_unqloc:
            eq_X,  eq_idx,   eq_inv   = np.unique(eq_X_all.round(2),  return_index=True, return_inverse=True, axis=0)
            sta_X, sta_idx,  sta_inv  = np.unique(sta_X_all.round(2), return_index=True, return_inverse=True, axis=0)
            eqid,  eqid_idx, eqid_inv = np.unique(eqid,               return_index=True, return_inverse=True, axis=0)
            ssn,   ssn_idx,  ssn_inv  = np.unique(ssn,                return_index=True, return_inverse=True, axis=0)
        else:
            #coordinates
            eq_X  = eq_X_all
            sta_X = sta_X_all
            #inverse indices
            eq_inv   = np.arange(npt)
            sta_inv  = np.arange(npt)
            eqid_inv = np.arange(npt)
            ssn_inv  = np.arange(npt)
        #number of coefficient for c1a, c1b and dS2S   
        nc1a  = len(eq_X)
        nc1b  = len(sta_X)
        neqid = len(eqid)
        ndS2S = len(ssn)
        
        #inverse indices of all coefficients for all frequencies
        eq_inv_allf   = np.hstack([eq_inv   + nc1a  * k for k in range(nf)])
        sta_inv_allf  = np.hstack([sta_inv  + nc1b  * k for k in range(nf)])
        eqid_inv_allf = np.hstack([eqid_inv + neqid * k for k in range(nf)])
        ssn_inv_allf  = np.hstack([ssn_inv  + ndS2S * k for k in range(nf)])
            
        #constants
        c0_mu   = self.Getc0(freq_nerg, npt, flag_consol=False)
        c0_mu.append(np.zeros(c0_mu[0].shape))
        c0_mu.append(np.zeros(c0_mu[0].shape))
        c0NS_mu = self.Getc0NS(freq_nerg, mag, eq_X_all, flag_consol=False)
        c0NS_mu.append(np.zeros(c0NS_mu[0].shape))
        c0NS_mu.append(np.zeros(c0NS_mu[0].shape))
        #sample spatialy varying coefficients
        #c1a
        c1a_mu, c1a_cov = self.Getc1a(freq_nerg, eq_X,  flag_consol=False)
        #extended it low and high frequencies
        c1a_mu.append(  np.zeros(c1a_mu[0].shape) )
        c1a_cov.append( self.hyp.theta_c1a[c_fn_min]**2 * (scipysp.eye(*c1a_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*c1a_cov[0].shape)) )
        c1a_mu.append( np.zeros(c1a_mu[0].shape) )
        c1a_cov.append( self.hyp.theta_c1a[c_fn_max]**2 * (scipysp.eye(*c1a_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*c1a_cov[0].shape)) )
        #c1b
        c1b_mu, c1b_cov = self.Getc1b(freq_nerg, sta_X, flag_consol=False)
        c1b_mu.append(  np.zeros(c1b_mu[0].shape) )
        c1b_cov.append( self.hyp.theta_c1a[c_fn_min]**2 * (scipysp.eye(*c1b_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*c1b_cov[0].shape)) )
        c1b_mu.append( np.zeros(c1b_mu[0].shape) )
        c1b_cov.append( self.hyp.theta_c1a[c_fn_max]**2 * (scipysp.eye(*c1b_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*c1b_cov[0].shape)) )
        #station terms
        dS2S_mu,   dS2S_cov = self.GetdS2S(freq_nerg, ssn, flag_consol=False)
        dS2S_mu.append( np.zeros(dS2S_mu[0].shape) )
        dS2S_cov.append( self.hyp.phi_S2S[c_fn_min]**2 * (scipysp.eye(*dS2S_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dS2S_cov[0].shape)) )
        dS2S_mu.append( np.zeros(dS2S_mu[0].shape) )
        dS2S_cov.append( self.hyp.phi_S2S[c_fn_max]**2 * (scipysp.eye(*dS2S_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dS2S_cov[0].shape)) )
        #sample anelastic attenuation coefficients
        cA_mu,  cA_cov, _, L_c, nc  = self.GetcA(freq_nerg, eqclst_X_all, sta_X_all, eqclst_z, flag_consol=False)
        dcA_mu  = [cA - self.hyp.loc[c_f,'mu_cA'] for c_f, cA in zip(c_freq_nerg, cA_mu)]
        dcA_mu.append( np.zeros(dcA_mu[0].shape) )
        cA_cov.append( (self.hyp.loc[c_fn_min,['theta_cA','sigma_cA']]**2).sum() * (scipysp.eye(*cA_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*cA_cov[0].shape)) )
        dcA_mu.append( np.zeros(dcA_mu[0].shape) )
        cA_cov.append( (self.hyp.loc[c_fn_max,['theta_cA','sigma_cA']]**2).sum() * (scipysp.eye(*cA_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*cA_cov[0].shape)) )
        #cell paths for all frequencies
        L_cells = [L_c] * nf
        L_cells = scipysp.block_diag(L_cells, format='csc') if self.sp_cov else scipylalg.block_diag(*L_cells)
        
        #compute weights for new frequencies
        freq_wt = self.GetFreqWt( freq_nerg_ext, freq)
        
        #compute mean coeffs and covariance at all frequencies
        c0_mu             = self.GetWeightedMeanCov(freq_wt, c0_mu,             flag_consol=False)[0]
        c0NS_mu           = self.GetWeightedMeanCov(freq_wt, c0NS_mu,           flag_consol=False)[0]
        c1a_mu,  c1a_cov  = self.GetWeightedMeanCov(freq_wt, c1a_mu,  c1a_cov,  flag_consol=False)
        c1b_mu,  c1b_cov  = self.GetWeightedMeanCov(freq_wt, c1b_mu,  c1b_cov,  flag_consol=False)
        dS2S_mu, dS2S_cov = self.GetWeightedMeanCov(freq_wt, dS2S_mu, dS2S_cov, flag_consol=False)
        dcA_mu,  cA_cov   = self.GetWeightedMeanCov(freq_wt, dcA_mu,  cA_cov,   flag_consol=False)
        cA_base           = [np.full(nc, c7) for c7          in c7_erg]
        cA_mu             = [cA_b + dcA_m    for cA_b, dcA_m in zip(cA_base, dcA_mu)]
        # import pdb; pdb.set_trace()
        #compute inter-frequency correlation of epistemic uncertainty terms
        if flag_ifreq:
            c1a_corr  = self.GetIfreqCorrc1a(freq,  1) 
            c1b_corr  = self.GetIfreqCorrc1b(freq,  1) 
            dS2S_corr = self.GetIfreqCorrdS2S(freq, 1) 
            cA_corr   = self.GetIfreqCorrcA(freq,   1)
        else:
            c1a_corr  = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            c1b_corr  = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            dS2S_corr = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)
            cA_corr   = scipysp.eye(nf, format='csc') if self.sp_cov else np.eye(nf)

        #rupture distance
        rrup = np.array(L_c.sum(axis=1)).flatten()
        #add ranomization if number of samples > 0
        # import pdb; pdb.set_trace()
        if nsamp:
            #sample spatially varying coefficients
            c0   = np.matlib.repmat(np.hstack(c0_mu),   nsamp,1).T
            c0NS = np.matlib.repmat(np.hstack(c0NS_mu), nsamp,1).T
            #earthquake constant
            c1a  = np.random.normal(size=[nc1a,nf,nsamp])  
            c1a  = np.array([pystats.MVNRnd(None, c1a_cov[k],  seed= c1a[:,k,:],                         flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            c1a  = np.array( pystats.MVNRnd(None, c1a_corr,    seed=[c1a[k,:,:]  for k in range(nc1a)],  flag_sp=self.sp_cov, flag_list=True) )
            #site constant
            c1b  = np.random.normal(size=[nc1b,nf,nsamp])
            c1b  = np.array([pystats.MVNRnd(None, c1b_cov[k],  seed= c1b[:,k,:],                         flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            c1b  = np.array( pystats.MVNRnd(None, c1b_corr,    seed=[c1b[k,:,:]  for k in range(nc1b)],  flag_sp=self.sp_cov, flag_list=True) )
            #site term
            dS2S = np.random.normal(size=[ndS2S,nf,nsamp])
            dS2S = np.array([pystats.MVNRnd(None, dS2S_cov[k], seed= dS2S[:,k,:],                        flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            dS2S = np.array( pystats.MVNRnd(None, dS2S_corr,   seed=[dS2S[k,:,:] for k in range(ndS2S)], flag_sp=self.sp_cov, flag_list=True) )
            #cell specific anelastic attenuation
            cA   = np.random.normal(size=[nc,nf,nsamp])
            cA   = np.array([pystats.MVNRnd(None, cA_cov[k],   seed= cA[:,k,:],                          flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            cA   = np.array( pystats.MVNRnd(None, cA_corr,     seed=[cA[k,:,:]   for k in range(nc)],    flag_sp=self.sp_cov, flag_list=True) )
            #correct random samples for unbiased mean
            c1a  = c1a  - c1a.mean(axis=2)[:, :, np.newaxis]
            c1b  = c1b  - c1b.mean(axis=2)[:, :, np.newaxis]
            dS2S = dS2S - dS2S.mean(axis=2)[:, :, np.newaxis]
            cA   = cA   - cA.mean(axis=2)[:, :, np.newaxis]
            #testing 
            # c1a[:,:,:] = 0
            # c1b[:,:,:] = 0
            # dS2S[:,:,:] = 0
            # cA[:,:,:] = 0
            #convert coeffs to two dimensional arrays 
            c1a  = np.vstack([c1a[:,k,:]  + c1a_mu[k][:,np.newaxis]  for k in range(nf)])
            c1b  = np.vstack([c1b[:,k,:]  + c1b_mu[k][:,np.newaxis]  for k in range(nf)])
            dS2S = np.vstack([dS2S[:,k,:] + dS2S_mu[k][:,np.newaxis] for k in range(nf)])
            cA   = np.vstack([cA[:,k,:]   + cA_mu[k][:,np.newaxis]   for k in range(nf)])
            # #testing 
            # c0[:,:]   = 0
            # c0NS[:,:] = 0
            # c1a[:,:]  = 0
            # c1b[:,:]  = 0
            # dS2S[:,:] = 0
            # cA[:,:] = 0
            # cA   = np.vstack([cA_base[k][:,np.newaxis]   for k in range(nf)])
            #assign coefficinets for all recordings
            c1a  = c1a[eq_inv_allf,  :]
            c1b  = c1b[sta_inv_allf, :]
            dS2S = dS2S[ssn_inv_allf,:]
        else:
            #assign coefficinets for all recordings
            c0   = np.hstack(c0_mu)
            c0NS = np.hstack(c0NS_mu)
            c1a  = np.hstack(c1a_mu)[eq_inv_allf]
            c1b  = np.hstack(c1b_mu)[sta_inv_allf]
            dS2S = np.hstack(dS2S_mu)[ssn_inv_allf]
            cA   = np.hstack(cA_mu)  
            
        # import pdb; pdb.set_trace()

        return(c0, c0NS, c1a, c1b, dS2S, cA, rrup, L_cells)
    
    def EasF(self, freq, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                   mag, vs30, ztor, fnorm, z1=None, regid=1,
                   ssn=None, eqid=None, 
                   nsamp = 0, flag_samp_unqloc = False, flag_ifreq = False, flag_sloc = False,
                   flag_include_c0NS = False, flag_condense=False):

        #convert eq and sta latlon to two dim array 
        if eq_latlon.ndim == 1:  eq_latlon  = eq_latlon.reshape(1,2)
        if sta_latlon.ndim == 1: sta_latlon = sta_latlon.reshape(1,2)
        #if unavailable compute z1 based on vs30
        if z1 is None: z1 = [self.Z1(vs, regid) for vs in vs30]
        #convert gmm parameters to numpy 
        mag   = np.array([mag]).flatten()
        vs30  = np.array([vs30]).flatten()
        ztor  = np.array([ztor]).flatten()
        fnorm = np.array([fnorm]).flatten()
        z1    = np.array([z1]).flatten()
        
        #convert freq to numpy array
        freq = np.array([freq]).flatten()
        
        #number of points
        npt = len(mag)
        #number for frequencies
        nf  = len(freq)
        
        #input assertions
        assert( np.all(np.isin(freq, self.nerg_freq)) ), 'Error. Unavailable frequency'
        assert( np.all(npt == np.array([len(eq_latlon), len(sta_latlon), 
                                        len(mag), len(vs30), len(ztor), len(fnorm)])) ),'Error. Inconsistent number of gmm parameters'
        
        #sample epistemic uncertainty coefficients
        c0, c0NS, c1a, c1b, dS2S, cA, rrup, L_cells = self.SampCoeffF(freq, mag, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                                                                      ssn, eqid, nsamp, flag_samp_unqloc,
                                                                      flag_ifreq=flag_ifreq, flag_sloc=flag_sloc)

        y_ne = c0 + c0NS + c1a + c1b + dS2S + L_cells.dot(cA) if flag_include_c0NS else c0 + c1a + c1b + dS2S + L_cells.dot(cA)
        
        #compute base EAS without anelastic attenuation
        fas_base, fas_base_lin, _ = super().EasF(freq, mag, rrup, vs30, ztor, fnorm, z1, flag_keep_b7=False, flag_flatten=False)
        #log of base gm
        y     = np.log(fas_base).flatten('F')
        y_lin = np.log(fas_base_lin).flatten('F')
        #add non-ergodic coefficients
        if nsamp:
            y     = y[:,np.newaxis]     + y_ne
            y_lin = y_lin[:,np.newaxis] + y_ne 
        else:
            y     = y     + y_ne
            y_lin = y_lin + y_ne
            
        #compute median non-ergodic ground motions
        fas_nlin = np.exp(y)
        fas_lin  = np.exp(y_lin) 
        
        #aleatory terms
        dBe_mu, dBe_cov = self.GetdBe(freq, eqid, npt=npt)
        dWe_cov         = self.GetdWe(freq, eqid, npt=npt)
        #increase covariance conditioning number
        dBe_cov += self.delta * (scipysp.eye(*dBe_cov.shape,  format='csc')  if self.sp_cov else np.eye(*dBe_cov.shape))
        #aleatory covariance
        aleat_cov = dWe_cov + dBe_cov;
        aleat_sig = np.sqrt( aleat_cov.diagonal() )
        
        if flag_condense:
            fas_nlin  = fas_nlin.reshape((npt,nf,nsamp), order='F') if nsamp > 0 else fas_nlin.reshape((npt,nf), order='F') 
            fas_lin   = fas_lin.reshape((npt,nf,nsamp),  order='F') if nsamp > 0 else fas_lin.reshape((npt,nf),  order='F')
            dBe_mu    = dBe_mu.reshape((npt,nf), order='F')
            aleat_sig = aleat_sig.reshape((npt,nf), order='F')
            aleat_cov = None

        return(fas_nlin, fas_lin, dBe_mu, aleat_sig, aleat_cov, rrup)
       
    def Eas(self, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                  mag, vs30, ztor, fnorm, z1 = None, regid = 1,
                  ssn = None, 
                  nsamp = 0, flag_samp_unqloc = True, flag_ifreq = False, flag_sloc = False,
                  fmin_r=0.5, fmax_r=1.5,
                  flag_include_c0NS = True, flag_flatten=True):
    
        #if unavailable compute z1 based on vs30
        if z1 is None: z1 = np.array([self.Z1(vs, regid) for vs in vs30])
                
        #convert gmm parameters to numpy 
        mag   = np.array([mag]).flatten()
        vs30  = np.array([vs30]).flatten()
        ztor  = np.array([ztor]).flatten()
        fnorm = np.array([fnorm]).flatten()
        z1    = np.array([z1]).flatten()
        #convert eq and sta latlon to two dim array 
        if eq_latlon.ndim == 1:  eq_latlon  = eq_latlon.reshape(1,2)
        if sta_latlon.ndim == 1: sta_latlon = sta_latlon.reshape(1,2)
        
        #number of points
        npt = len(mag)
        
        #frequencies for GMM
        freq   = self.freq
        nf     = len(freq)
        #non-ergodic frequencies info
        freq_nerg = np.array(self.nerg_freq).flatten()
        c_freq_nerg = [self.GetFreqName(f) for f in freq_nerg]
        c_fn_min    = self.GetFreqName(freq_nerg.min())
        c_fn_max    = self.GetFreqName(freq_nerg.max())
        #extened non-ergodic frequencies
        #extened non-ergodic frequencies
        freq_min_ext = np.max([self.minfreq, fmin_r * freq_nerg.min() ])
        freq_max_ext = np.min([self.maxfreq, fmax_r * freq_nerg.max() ])
        freq_nerg_ext =  np.concatenate([ freq_nerg, [freq_min_ext, freq_max_ext] ])
        
        #input assertions
        assert( np.all(np.isin(freq_nerg, self.nerg_freq)) ), 'Error. Unavailable frequency'
        assert( np.all(npt == np.array([len(eq_latlon), len(sta_latlon),
                                        len(mag), len(vs30), len(ztor), len(z1), len(fnorm)])) ),'Error. Inconsistent number of gmm parameters'
        
        #sample epistemic uncertainty coefficients
        c0, c0NS, c1a, c1b, dS2S, cA, rrup, L_cells = self.SampCoeffAll(mag, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                                                                        ssn, eqid=None, nsamp=nsamp, 
                                                                        flag_samp_unqloc=flag_samp_unqloc, flag_ifreq=flag_ifreq, 
                                                                        flag_sloc = flag_sloc,
                                                                        fmin_r=fmin_r, fmax_r=fmax_r)

        #compute base EAS without anelastic attenuation
        _, fas_base, fas_base_lin, _ = super().Eas(mag, rrup, vs30, ztor, fnorm, z1, flag_keep_b7=False, flag_flatten=False)
        #log of base gm
        y_base     = np.log(fas_base)
        y_base_lin = np.log(fas_base_lin)

        #sum of non-ergodic effects at non-ergodic frequencies
        y_ne = c0 + c0NS + c1a + c1b + dS2S + L_cells.dot(cA) if flag_include_c0NS else c0 + c1a + c1b + dS2S + L_cells.dot(cA)
        # import pdb; pdb.set_trace()
        #reshape non-ergodic adjustments ( scenarios x frequencies )
        y_ne = y_ne.reshape((npt,nf,nsamp), order='F') if nsamp > 0 else y_ne.reshape((npt,nf), order='F') 

        #aleatory terms
        #dBe
        dBe_cov = self.GetdBe(freq_nerg, npt=npt, flag_consol=False)[1]
        dBe_cov.append( self.hyp.tau_0[c_fn_min]**2 * (scipysp.eye(*dBe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dBe_cov[0].shape)) )
        dBe_cov.append( self.hyp.tau_0[c_fn_max]**2 * (scipysp.eye(*dBe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dBe_cov[0].shape)) )
        #dWe
        dWe_cov = self.GetdWe(freq_nerg, npt=npt, flag_consol=False)
        dWe_cov.append( self.hyp.phi_0[c_fn_min]**2 * (scipysp.eye(*dWe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dWe_cov[0].shape)) )
        dWe_cov.append( self.hyp.phi_0[c_fn_max]**2 * (scipysp.eye(*dWe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dWe_cov[0].shape)) )
        
        #compute weights for new frequencies
        freq_wt = self.GetFreqWt( freq_nerg_ext, freq)
        
        #compute mean coeffs and covariance at all frequencies
        dBe_cov = self.GetWeightedMeanCov(freq_wt, forig_cov = dBe_cov)[1]
        dWe_cov = self.GetWeightedMeanCov(freq_wt, forig_cov = dWe_cov)[1]
               
        #full non-ergodic aleatory covariance
        y_ne_cov = dWe_cov + dBe_cov;
        y_ne_sig = np.sqrt( y_ne_cov.diagonal() )
        #reshape aleatory std ( scenarios x frequencies )
        y_ne_sig = y_ne_sig.reshape((npt,nf), order='F')
        
        #compute final ground motion sum of base erg gmm and non-ergodic adjustments
        if nsamp:
            fas_nlin = y_base[:,:,np.newaxis]     + y_ne
            fas_lin  = y_base_lin[:,:,np.newaxis] + y_ne
        else:
            fas_nlin = y_base     + y_ne
            fas_lin  = y_base_lin + y_ne
        #comptue ground motion
        fas_nlin = np.exp(fas_nlin)
        fas_lin  = np.exp(fas_lin)  
        #flatten arrays if only one datapoint
        if fas_nlin.shape[0] == 1 and flag_flatten:
            fas_nlin = fas_nlin[0,:].T
            fas_lin  = fas_lin[0,:].T
            # fas_nlin = fas_nlin.squeeze().T if nsamp else fas_nlin[0,:]
            # fas_lin  = fas_lin[0,:,:]  if nsamp else fas_lin[0,:]
        #aleatory 
        fas_sig  =  y_ne_sig
        
        return freq, fas_nlin, fas_lin, fas_sig, rrup
    
    def EasSampAleat(self, eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                     mag, vs30, ztor, fnorm, z1 = None, regid = 1,
                     ssn = None, eqid=None,
                     nsamp = 0, flag_samp_unqloc = True, flag_sloc = False,
                     fmin_r=0.5, fmax_r=1.5,
                     flag_include_c0NS = True, flag_flatten=True):
        
        #number of points
        npt = len(mag)
        #assign -999 if eqid or ssn are unspecified
        if eqid is None: 
            eqid = -1 - np.arange(npt)
        else:
            eqid = ConvertPandasDf2NpArray(eqid)
            eqid[np.isnan(eqid)] = -1 - np.arange(np.isnan(eqid).sum())

        #convert eqid to np arrays and integers   
        eqid = np.array([eqid]).flatten()
        eqid = eqid.astype(int)
        
        #input assertions
        assert( np.all(npt == np.array([len(eq_latlon), len(sta_latlon), 
                                        len(ssn), len(eqid), len(eqclst_z)])) ),'Error. Inconsistent number of gmm parameters.'
        assert( fmin_r <=1 and fmax_r>=1 ),'Error. Invalid frequency ratios.'
        
        #if true sample coefficients only for unique locations
        if flag_samp_unqloc:
            eqid,  eqid_idx, eqid_inv = np.unique(eqid, return_index=True, return_inverse=True, axis=0)
        else:
            #inverse indices
            eqid_inv = np.arange(npt)
        #number of coefficient for c1a, c1b and dS2S   
        neqid = len(eqid)
        
        #sample ergodic coefficients
        freq, fas_nlin, fas_lin, fas_sig, rrup = self.Eas(eq_latlon, sta_latlon, eqclst_latlon, eqclst_z,
                                                          mag, vs30, ztor, fnorm, z1 = None, regid = 1,
                                                          ssn = None, 
                                                          nsamp = 0, flag_samp_unqloc = True, flag_ifreq = True, flag_sloc = False,
                                                          fmin_r=0.5, fmax_r=1.5,
                                                          flag_include_c0NS = True, flag_flatten=True)
        
        #frequencies for GMM
        freq   = self.freq
        nf     = len(freq)
        #non-ergodic frequencies info
        freq_nerg = np.array(self.nerg_freq).flatten()
        c_fn_min    = self.GetFreqName(freq_nerg.min())
        c_fn_max    = self.GetFreqName(freq_nerg.max())
        #extened non-ergodic frequencies
        #extened non-ergodic frequencies
        freq_min_ext = np.max([self.minfreq, fmin_r * freq_nerg.min() ])
        freq_max_ext = np.min([self.maxfreq, fmax_r * freq_nerg.max() ])
        freq_nerg_ext =  np.concatenate([ freq_nerg, [freq_min_ext, freq_max_ext] ])
        
        import pdb; pdb.set_trace()
        
        #inverse indices of all coefficients for all frequencies
        eqid_inv_allf = np.hstack([eqid_inv + neqid * k for k in range(nf)])
        
        #aleatory terms
        #dBe
        dBe_cov = self.GetdBe(freq_nerg, npt=npt, flag_consol=False)[1]
        dBe_cov.append( self.hyp.tau_0[c_fn_min]**2 * (scipysp.eye(*dBe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dBe_cov[0].shape)) )
        dBe_cov.append( self.hyp.tau_0[c_fn_max]**2 * (scipysp.eye(*dBe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dBe_cov[0].shape)) )
        #dWe
        dWe_cov = self.GetdWe(freq_nerg, npt=npt, flag_consol=False)
        dWe_cov.append( self.hyp.phi_0[c_fn_min]**2 * (scipysp.eye(*dWe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dWe_cov[0].shape)) )
        dWe_cov.append( self.hyp.phi_0[c_fn_max]**2 * (scipysp.eye(*dWe_cov[0].shape,  format='csc') if self.sp_cov else np.eye(*dWe_cov[0].shape)) )
        
        #compute weights for new frequencies
        freq_wt = self.GetFreqWt( freq_nerg_ext, freq)
        
        #compute mean coeffs and covariance at all frequencies
        dBe_cov = self.GetWeightedMeanCov(freq_wt, forig_cov = dBe_cov, flag_consol=False)[1]
        dWe_cov = self.GetWeightedMeanCov(freq_wt, forig_cov = dWe_cov, flag_consol=False)[1]
        
        #compute inter-frequency correlation of epistemic uncertainty terms
        dBe_corr  = self.GetIfreqCorrcAleat(freq,  1) 
        dWe_corr  = self.GetIfreqCorrcAleat(freq,  1) 

        #
        if nsamp:
            #between event terms
            dBe  = np.random.normal(size=[npt,nf,nsamp])  
            dBe  = np.array([pystats.MVNRnd(None, dBe_cov[k],  seed= dBe[:,k,:],                        flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            dBe  = np.array( pystats.MVNRnd(None, dBe_corr,    seed=[dBe[k,:,:]  for k in range(npt)],  flag_sp=self.sp_cov, flag_list=True) )
            #within event terms
            dWe  = np.random.normal(size=[npt,nf,nsamp])
            dWe  = np.array([pystats.MVNRnd(None, dWe_cov[k],  seed= dWe[:,k,:],                        flag_sp=self.sp_cov) for k in range(nf)]).swapaxes(0,1)
            dWe  = np.array( pystats.MVNRnd(None, dWe_corr,    seed=[dWe[k,:,:]  for k in range(npt)],  flag_sp=self.sp_cov, flag_list=True) )
            #convert coeffs to two dimensional arrays 
            dBe  = np.vstack([dBe[:,k,:] for k in range(nf)])
            dWe  = np.vstack([dWe[:,k,:] for k in range(nf)])
            #assign coefficinets for all recordings
            dBe  = dBe[eqid_inv_allf, :]
            dWe  = dWe[eqid_inv_allf, :]
    
    
    #Add inter-frequency correlation coefficients
    #--------------------------------------
    def AddInterFreqCorrNCoeff(self, c1a_A,  c1a_B,  c1a_C,  c1a_D,  c1b_A, c1b_B, c1b_C, c1b_D, 
                               dS2S_A, dS2S_B, dS2S_C, dS2S_D, cA_A,  cA_B,  cA_C, cA_D):
    
    
        self.hyp_ifreq.loc['c1a',:]  = [c1a_A,  c1a_B,  c1a_C,  c1a_D]
        self.hyp_ifreq.loc['c1b',:]  = [c1b_A,  c1b_B,  c1b_C,  c1b_D]
        self.hyp_ifreq.loc['dS2S',:] = [dS2S_A, dS2S_B, dS2S_C, dS2S_D]
        self.hyp_ifreq.loc['cA',:]   = [cA_A,   cA_B,   cA_C,   cA_D]
        
    def AddInterFreqCorrAleat(self, aleat_A=1.25,  aleat_B=-0.75,  aleat_C=1.5,  aleat_D=-45):
        
        self.hyp_ifreq.loc['aleat',:]  = [aleat_A,  aleat_B,  aleat_C,  aleat_D]
        
    #Non-ergodic frequencies
    #--------------------------------------
    def GetNergFreq(self):
        '''
        Return non-ergodic frequencies

        Returns
        -------
        pt_xy : np.array(dim=[len(nerg_freq),])
            Non-ergodic frequencies
        '''        
        
        return np.array(self.nerg_freq)
    
    def GetFreqName(self, freq):
        
        nerg_freq = self.GetNergFreq()
        i_f = np.where(np.abs(nerg_freq - freq) < self.ftol)[0]
        assert(len(i_f) == 1),'Error. Undefined non-ergodic frequency'
        col_freq = 'f%i'% i_f
        
        return col_freq

    #Projection functions
    #--------------------------------------
    def ProjLatLon2UTM(self, pt_latlon):
        '''
        Compute UTM coordiantes from lat, lon based on GMM projection system.

        Parameters
        ----------
        pt_latlon : np.array(dim=[2,])
            Latitude and logitude point coordinates.

        Returns
        -------
        np.array(dim=[2,])
            UTM x&y point coordinates.
        '''
        
        return np.array(self.proj_utm(pt_latlon[1], pt_latlon[0])) / 1000
        
    def ProjUTM2LatLon(self, pt_xy):
        '''
         Compute lat, lon coordinates from UTM based on GMM projection system.
 

        Parameters
        ----------
        pt_xy : np.array(dim=[2,])
            UTM x&y coordinates.


        Returns
        -------
        np.array(dim=[2,])
            Latitude and logitude point coordinates.
        '''
             
        #convert from km to m
        pt_xy = pt_xy * 1000
        return np.flip(self.proj_utm(pt_xy[0], pt_xy[1], inverse=True))
    
    def CellPaths(self, eq_X, sta_X, eq_z):
    
        #compute anelastic cell pahts
        L_cells = np.zeros([len(eq_X), len(self.cell_X)])
        #import pdb; pdb.set_trace()
        for i, (pt1, pt2) in enumerate(zip(eq_X, sta_X)):
            #add depth info
            pt1 = np.append(pt1, eq_z[i])
            pt2 = np.append(pt2, 0)
            #compute cell paths
            try:
                dm = pycelldist.ComputeDistGridCells(pt1,pt2,self.cell_X, flagUTM=True)
            except ValueError:
                print('i: ',i)
                print('pt1: ', pt1)
                print('pt2: ', pt2)
                raise
            L_cells[i] = dm
            
        #valid cells with more than one path
        i_c_valid = np.where(L_cells.sum(axis=0) > 0)[0] #valid cells with more than one path
        L_cells   = L_cells[:,i_c_valid]
        n_c       = len(i_c_valid)
    
        return(i_c_valid, L_cells, n_c)
    
    def GetSubReg(self, eq_X):
        
        #create shapely objects for subregion polygons
        if self.sregN_poly is None:
            sregN_utm = np.array([self.ProjLatLon2UTM(pt) for pt in self.sregN_latlon])
            sregS_utm = np.array([self.ProjLatLon2UTM(pt) for pt in self.sregS_latlon])
            #North and South sub-region shapely polygons
            self.sregN_poly = shp_poly(sregN_utm)
            self.sregS_poly = shp_poly(sregS_utm)
            
        #true if earthquake belongs in North or South subregion polygon
        i_sregN = np.array([ shp_pt(eq_x).within(self.sregN_poly) for eq_x in eq_X ])
        i_sregS = np.array([ shp_pt(eq_x).within(self.sregS_poly) for eq_x in eq_X ])
        assert( ~np.vstack([i_sregS, i_sregN]).all(axis=0).any() ),'Error non-unique subregion'
        
        return i_sregN, i_sregS
    
    #Sample coefficients at intermediary frequencies
    #--------------------------------------
    def GetFreqWt(self, freq_orig, freq_new):
    
        #number of original frequencies
        n_forig = len(freq_orig)
        #sort original frequencies from smallest to largest
        forig_s, i_forig = np.unique(freq_orig, return_inverse=True)
        assert(len(forig_s) == n_forig),'Error. Duplicate original frequencies'
       
        #initialize weights matrix and iterate over all non-ergodic frequencies
        wt_mat = list()
        for j in range(n_forig):
            #weight fuction for current non-ergodic freq
            wt_f    = np.zeros(n_forig)
            wt_f[j] = 1    
            #weights at new frequencies
            wt_mat.append(np.interp(freq_new, forig_s, wt_f))
        #summarize weights at all new frequncies
        wt_mat = np.vstack(wt_mat)
        #reorder weights based on the original frequencies
        wt_mat = wt_mat[i_forig,:]
    
        return wt_mat
    
    def GetWeightedMeanCov(self, wt_mat, forig_mu=None, forig_cov=None, flag_consol=True):
        
        #number of new frequencies
        n_fnew = wt_mat.shape[1]

        #compute weighted mean
        if not forig_mu is None:
            fnew_mu = list()
            for j in range(n_fnew):
                #compute weighted mean array
                fn_mu = np.sum([fo_wt*fo_mu for fo_wt, fo_mu in zip(wt_mat[:,j], forig_mu)], axis=0)
                #summarize mean coefficients for all new frequencies
                fnew_mu.append(fn_mu)
            #consolidate mean arrays if specified
            if flag_consol:
                fnew_mu  = np.hstack(fnew_mu)
        #return none for mean if not provided in inputs
        else:
            fnew_mu = None
            
        #compute weighted covariance
        if not forig_cov is None:
            fnew_cov = list()
            for j in range(n_fnew):
                #compute weighted covariance matrix
                fn_cov = np.sum([fo_wt*fo_cov for fo_wt, fo_cov in zip(wt_mat[:,j], forig_cov)], axis=0)
                #summarize covariance matrix for all new frequencies
                fnew_cov.append(fn_cov)
            #consolidate mean arrays if specified
            if flag_consol:
                fnew_cov = scipysp.block_diag(fnew_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*fnew_cov)
        #return none for covariance if not provided in inputs
        else:
            fnew_cov = None
        
        return fnew_mu, fnew_cov
    
    #Find SSN based on coordinates
    #--------------------------------------
    def FindSSN(self, site_info, ssn, s_X, s_Xtol):

        #update coefficinet dataframe with new coeff at existing locations 
        i_site = list()
        for s_x in s_X: 
            #distance to stations
            dist2sta = scipylalg.norm(site_info[['x','y']] - s_x, axis=1)
            #find site index based on coordinates, set to nan if non avail coor in site_info
            i_s = dist2sta.argmin() if dist2sta.min() < s_Xtol else np.nan
            i_site.append(i_s)
        i_site = np.array(i_site)

        #find records with unspecified ssn
        i_ssn_empty = np.logical_or(np.isnan(ssn), ssn<0)
        #station sequence numbers to update
        i_ssn2upd = np.logical_and(i_ssn_empty , ~np.isnan(i_site))
        
        #update station sequence number
        if i_ssn2upd.sum() > 0:
            ssn[i_ssn2upd] = site_info['id'].values[ i_site[i_ssn2upd].astype(int) ]
        
        return ssn
    
    #Get constant coefficients
    #--------------------------------------
    def Getc0(self, freqs, npt, flag_consol = True):
        
        #sample constant for all frequencies
        c0_allfreq_mu  = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            c0_mu = np.ones(npt) * self.hyp.loc[c_f,'c0']
            #summarize c0 for different frequencies
            c0_allfreq_mu.append(c0_mu)
        #convert mean to an array
        if flag_consol:
            c0_allfreq_mu  = np.hstack(c0_allfreq_mu)

        return c0_allfreq_mu
    
    def Getc0NS(self, freqs, mag, eq_X, flag_consol = True):
        
        #number of points per frequency
        npt = len(mag)        
        #sample constant for all frequencies
        c0NS_allfreq_mu  = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #threshold magnitude and North/South region indices
            i_mag = mag <= self.sreg_mag
            i_sregN, i_sregS = self.GetSubReg(eq_X)
            #regional constant
            c0NS_mu = np.zeros(npt) 
            c0NS_mu[np.logical_and(i_mag,i_sregN)] = self.hyp.loc[c_f,'c0_N']
            c0NS_mu[np.logical_and(i_mag,i_sregS)] = self.hyp.loc[c_f,'c0_S']
            #summarize c0NS for different frequencies
            c0NS_allfreq_mu.append(c0NS_mu)
        #convert mean to an array
        if flag_consol:
            c0NS_allfreq_mu  = np.hstack(c0NS_allfreq_mu)

        return c0NS_allfreq_mu

    #Get covariance matrix inter-frequency correlation
    #--------------------------------------
    def GetIfreqCorr(self, freq, npt, A, B, C, D):
        
        #correlation between frequencies
        cov_ifreq_corr = list()
        for j1, f1 in enumerate(freq):
            cov_ifreq_corr.append([]) 
            for j2, f2 in enumerate(freq):
                fr  = np.abs(np.log(f1/f2))
                fm  = np.min([f1,f2])
                rho = np.tanh( A*np.exp(-B*fr) + C*np.exp(-D*fr) )
                cov_f1f2_corr = 1 if np.abs(f1-f2) < 1e-6 else rho
                #
                cov_f1f2_corr = cov_f1f2_corr * scipysp.eye(npt, format='csc')  if self.sp_cov else cov_f1f2_corr * np.eye(npt) 
                #add correlation betwen f1 and f2
                cov_ifreq_corr[j1].append(cov_f1f2_corr)

        #create correlation matrix for all frequencies       
        cov_ifreq_corr = scipysp.bmat(cov_ifreq_corr, format='csc') if self.sp_cov else np.block(cov_ifreq_corr)
        
        return cov_ifreq_corr

    def GetIfreqCorrc1a(self, freq, n_eq):
        #coefficients of c1a inter-frequency correlation
        A_c1a, B_c1a, C_c1a, D_c1a = self.hyp_ifreq.loc['c1a',:]
        return self.GetIfreqCorr(freq, n_eq, A_c1a, B_c1a, C_c1a, D_c1a)
        
    def GetIfreqCorrc1b(self, freq, n_sta):
        #coefficients of c1b inter-frequency correlation
        A_c1b, B_c1b, C_c1b, D_c1b = self.hyp_ifreq.loc['c1b',:]
        return self.GetIfreqCorr(freq, n_sta, A_c1b, B_c1b, C_c1b, D_c1b)

    def GetIfreqCorrdS2S(self, freq, n_sta):
        #coefficients of dS2S inter-frequency correlation
        A_dS2S, B_dS2S, C_dS2S, D_dS2S = self.hyp_ifreq.loc['dS2S',:]
        return self.GetIfreqCorr(freq, n_sta, A_dS2S, B_dS2S, C_dS2S, D_dS2S)

    def GetIfreqCorrcA(self, freq, n_c):
        #coefficients of cA inter-frequency correlation
        A_cA, B_cA, C_cA, D_cA = self.hyp_ifreq.loc['cA',:]
        return self.GetIfreqCorr(freq, n_c, A_cA, B_cA, C_cA, D_cA)
   
    def GetIfreqCorrcAleat(self, freq, n_gm):
        #coefficients of c1a inter-frequency correlation
        A_aleat, B_aleat, C_aleat, D_aleat = self.hyp_ifreq.loc['aleat',:]
        return self.GetIfreqCorr(freq, n_gm, A_aleat, B_aleat, C_aleat, D_aleat)

    #Auxiliary functions
    #--------------------------------------
    def Convert2SparseCovMat(self, cov_mat, frmt = 'csc'):

        #set to zero entries bellow the tolerance
        cov_mat[np.abs(cov_mat) < self.sp_tol*np.abs(cov_mat).max()] = 0

        #convert to spase matrix
        if   frmt == 'csc': cov_mat = scipysp.csc_matrix(cov_mat, copy=True)
        elif frmt == 'csr': cov_mat = scipysp.csr_matrix(cov_mat, copy=True)
        else: raise RuntimeError('Invalid sparse matrix format')

        return cov_mat

## Compute non-ergodic ground motion based on grid coefficients
## ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
class NonErgEASGMMCoeffGrid(NonErgEASGMMCoeffBase):
    
    def AddFreq(self, freq, zone_utm, g_Xutm, cA_Xutm, ssn, eqid,
                      c1a_mean, c1a_cov, c1b_mean, c1b_cov, 
                      dS2S_mean, dS2S_sig, dBe_mean, dBe_sig,
                      cA_mean, cA_cov,
                      c0, c0_N, c0_S, rho_c1a, theta_c1a, rho_c1b, theta_c1b, 
                      mu_cA, rho_cA,  theta_cA, sigma_cA,  pi_cA,
                      phi_S2S, tau_0, phi_0):
        
        #convert freq to numpy array
        freq = np.array([freq]).flatten()

        #if ids and coordinates are pandas data-frame convert to numpy arrays
        eqid      = ConvertPandasDf2NpArray(eqid)
        ssn       = ConvertPandasDf2NpArray(ssn)
        g_Xutm    = ConvertPandasDf2NpArray(g_Xutm)
        c1a_mean  = ConvertPandasDf2NpArray(c1a_mean)
        c1a_cov   = ConvertPandasDf2NpArray(c1a_cov)
        c1b_mean  = ConvertPandasDf2NpArray(c1b_mean)
        c1b_cov   = ConvertPandasDf2NpArray(c1b_cov)
        dS2S_mean = ConvertPandasDf2NpArray(dS2S_mean)
        dS2S_sig  = ConvertPandasDf2NpArray(dS2S_sig)
        dBe_mean  = ConvertPandasDf2NpArray(dBe_mean)
        dBe_sig   = ConvertPandasDf2NpArray(dBe_sig)
        cA_mean   = ConvertPandasDf2NpArray(cA_mean)
        cA_cov    = ConvertPandasDf2NpArray(cA_cov)
        
        #convert eqid and ssn to integers
        eqid = eqid.astype(int)
        ssn  = ssn.astype(int)
        #reserve eqid = -999 for new earthquake and station
        assert( ~(eqid < 0).any()),'Error. Negative EQ_IDs are not valid'
        assert( ~(ssn  < 0).any()),'Error. Negative SSNs are not valid'
        
        #update ids and coordinates if not defined
        if self.eqid is None: self.eqid = eqid
        if self.ssn  is None: self.ssn = ssn 
        if self.zone_utm is None: 
            self.zone_utm = zone_utm
            self.proj_utm  = pyproj.Proj("+proj=utm +zone="+zone_utm+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        if self.grid_X   is None: self.grid_X = g_Xutm 
        if self.cell_X   is None: self.cell_X = cA_Xutm
        
        #check consistency of grid coefficient and cell coordinates
        assert(self.zone_utm == zone_utm),           'Error. Inconsistent UTM zones'
        assert(np.array_equal(self.grid_X, g_Xutm)), 'Error. Inconsistent grid coordinates'
        assert(np.array_equal(self.cell_X, cA_Xutm)),'Error. Inconsistent cell coordinates'

        #indices for earthquake and station ids
        i_eqid = np.array([np.where(self.eqid == e_id) if (self.eqid == e_id).any() else np.array([[np.nan]]) for e_id in  eqid]).flatten()
        i_ssn  = np.array([np.where(self.ssn  == s_id) if (self.ssn  == s_id).any() else np.array([[np.nan]]) for s_id in  ssn]).flatten()
        #remove nan
        i_eqid_val = ~np.isnan(i_eqid)
        i_ssn_val  = ~np.isnan(i_ssn)
        i_eqid = i_eqid[i_eqid_val].astype(int)
        i_ssn  = i_ssn[i_ssn_val].astype(int)

        #add new frequency if doesn’t exist in non-ergodic frequencies
        if (not freq in self.nerg_freq): self.nerg_freq.append(freq)
         
        #if specifed convert to space matrix
        if self.sp_cov:
            c1a_cov = self.Convert2SparseCovMat( c1a_cov )
            c1b_cov = self.Convert2SparseCovMat( c1b_cov )
            cA_cov  = self.Convert2SparseCovMat( cA_cov )
                   
        #frequency column
        c_f = self.GetFreqName(freq)
        #spatially varying coefficients
        self.c1a_mu[c_f]  = c1a_mean
        self.c1b_mu[c_f]  = c1b_mean
        self.c1a_cov[c_f] = c1a_cov
        self.c1b_cov[c_f] = c1b_cov
        #station terms
        self.dS2S_mu[c_f]         = np.zeros(self.ssn.shape)
        self.dS2S_sig[c_f]        = np.full(self.ssn.shape,  phi_S2S)
        self.dS2S_mu[c_f][i_ssn]  = dS2S_mean[i_ssn_val]
        self.dS2S_sig[c_f][i_ssn] = dS2S_sig[i_ssn_val]
        #anelastic attenuation cells
        self.cA_mu[c_f]  = cA_mean
        self.cA_cov[c_f] = cA_cov
        #aleatory terms
        self.dBe_mu[c_f]          = np.zeros(self.eqid.shape)
        self.dBe_sig[c_f]         = np.full(self.eqid.shape, tau_0)
        self.dBe_mu[c_f][i_eqid]  = dBe_mean[i_eqid_val]
        self.dBe_sig[c_f][i_eqid] = dBe_sig[i_eqid_val]
        #hyper-parameters
        self.hyp.loc[c_f] = [c0, c0_N, c0_S, rho_c1a, theta_c1a, rho_c1b, theta_c1b, 
                             mu_cA, rho_cA,  theta_cA, sigma_cA,  pi_cA,
                             phi_S2S, tau_0, phi_0 ]
        
    #Get non-ergodic coefficinets on coordinates of interest
    #--------------------------------------
    def Getc1a(self, freqs, eq_X, flag_consol = True):
        
        #grid point indices for earthquake
        i_g_eq  = [scipylalg.norm(self.grid_X - eq_x, axis=1).argmin()  for eq_x in eq_X]
        #sample mean and covariance of earthquake coefficient
        c1a_allfreq_mu  = list()
        c1a_allfreq_cov = list()
        for f in freqs:
            c1a_mu = self.c1a_mu[f][i_g_eq]
            if self.sp_cov: c1a_cov = SlicingSparceMat(self.c1a_cov[f],i_g_eq,i_g_eq)
            else:           c1a_cov = self.c1a_cov[f][i_g_eq,:][:,i_g_eq]
            #summarize c1a for different frequencies
            c1a_allfreq_mu.append(c1a_mu)
            c1a_allfreq_cov.append(c1a_cov)
        #convert mean and covariance to array and matrix
        if flag_consol:
            c1a_allfreq_mu  = np.hstack(c1a_allfreq_mu)
            c1a_allfreq_cov = scipysp.block_diag(c1a_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*c1a_allfreq_cov)
        
        return c1a_allfreq_mu, c1a_allfreq_cov
    
    def Getc1b(self, freqs, sta_X, flag_consol = True):
        
        #grid point indices for site
        i_q_sta = [scipylalg.norm(self.grid_X - sta_x, axis=1).argmin() for sta_x in sta_X]
        #sample mean and covariance of station coefficient
        c1b_allfreq_mu  = list()
        c1b_allfreq_cov = list()
        for f in freqs:
            c1b_mu = self.c1b_mu[f][i_q_sta]
            if self.sp_cov: c1b_cov = SlicingSparceMat(self.c1b_cov[f],i_q_sta,i_q_sta)
            else:           c1b_cov = self.c1b_cov[f][i_q_sta,:][:,i_q_sta]
            #summarize c1b for different frequencies
            c1b_allfreq_mu.append(c1b_mu)
            c1b_allfreq_cov.append(c1b_cov)
        #convert mean and covariance to array and matrix
        if flag_consol:
            c1b_allfreq_mu  = np.hstack(c1b_allfreq_mu)
            c1b_allfreq_cov = scipysp.block_diag(c1b_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*c1b_allfreq_cov)
        
        return c1b_allfreq_mu, c1b_allfreq_cov
    
    def GetdS2S(self, freqs, ssn, flag_consol = True):
        
        #sample mean and covariance of deltaS2S
        dS2S_allfreq_mu  = list()
        dS2S_allfreq_cov = list()
        for f in freqs:
            dS2S_mu, _, dS2S_cov = pygp.SampledS2S(ssn, self.ssn, self.dS2S_mu[f], self.hyp.phi_S2S[f], self.dS2S_sig[f])
            #summarize dS2S for different frequencies
            dS2S_allfreq_mu.append(dS2S_mu)
            dS2S_allfreq_cov.append(dS2S_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dS2S_allfreq_mu  = np.hstack(dS2S_allfreq_mu)
            dS2S_allfreq_cov = scipysp.block_diag(dS2S_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dS2S_allfreq_cov)
        
        return dS2S_allfreq_mu, dS2S_allfreq_cov

    def GetcA(self, freqs, eq_X, sta_X, eq_z, flag_consol = True):
        
        #cell indices and path lengths
        i_c_valid, L_cells, n_c = self.CellPaths(eq_X, sta_X, eq_z, flag_consol = True)
        #sample mean and covariance of cell attenuation coefficients
        cA_allfreq_mu  = list()
        cA_allfreq_cov = list()
        for f in freqs:
            cA_mu = self.cA_mu[f][i_c_valid]
            if self.sp_cov: cA_cov = SlicingSparceMat(self.cA_cov[f],i_c_valid,i_c_valid)
            else:           cA_cov = self.cA_cov[f][i_c_valid,:][:,i_c_valid]
            #summarize cA for different frequencies
            cA_allfreq_mu.append(cA_mu)
            cA_allfreq_cov.append(cA_cov)     
        L_allfreq_cells = [L_cells]*len(freqs)
        #convert mean and covariance to array and matrix
        if flag_consol:
            cA_allfreq_mu   = np.hstack(cA_allfreq_mu)
            cA_allfreq_cov  = scipysp.block_diag(cA_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*cA_allfreq_cov)
            L_allfreq_cells = scipysp.block_diag(L_allfreq_cells, format='csc') if self.sp_cov else scipylalg.block_diag(*L_allfreq_cells)

        return cA_allfreq_mu, cA_allfreq_cov, L_allfreq_cells, L_cells, n_c
    
    #Get aleatory terms
    #--------------------------------------
    def GetdBe(self, freqs, eqid=None, npt=None, flag_consol = True):
        
        #earthquake id if undefined
        eqid = -1 - np.arange(npt) if eqid is None else eqid
        #sample mean and covariance of deltaS2S
        dBe_allfreq_mu  = list()
        dBe_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            dBe_mu,  _, dBe_cov = pygp.SampledBe(eqid, self.eqid, self.dBe_mu[c_f], self.hyp.tau_0[c_f], self.dBe_sig[c_f])
            dBe_cov = self.Convert2SparseCovMat(dBe_cov) if self.sp_cov else dBe_cov
            #summarize dBe for different frequencies
            dBe_allfreq_mu.append(dBe_mu)
            dBe_allfreq_cov.append(dBe_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dBe_allfreq_mu  = np.hstack(dBe_allfreq_mu)
            dBe_allfreq_cov = scipysp.block_diag(dBe_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dBe_allfreq_cov)
        
        return dBe_allfreq_mu, dBe_allfreq_cov

    def GetdWe(self, freqs, eqid=None, npt=None, flag_consol = True):
        
        #covariance of within event residual
        npt = npt if eqid is None else len(eqid)
        dWe_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            dWe_cov = self.hyp.phi_0[c_f]**2 * np.eye(npt)
            dWe_cov = self.Convert2SparseCovMat(dWe_cov) if self.sp_cov else dWe_cov
            #summarize dBe for different frequencies
            dWe_allfreq_cov.append(dWe_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dWe_allfreq_cov = scipysp.block_diag(dWe_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dWe_allfreq_cov)
        
        return dWe_allfreq_cov
    
## Compute non-ergodic ground motion conditioned on coefficients at known locations
## ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
class NonErgEASGMMCoeffCond(NonErgEASGMMCoeffBase):
    
    def AddFreq(self, freq, zone_utm, c1a_Xutm, c1b_Xutm, cAmpt_Xutm, ssn, eqid, cA_id,
                      c1a_mean, c1a_sig, c1b_mean, c1b_sig, 
                      dS2S_mean, dS2S_sig, dBe_mean, dBe_sig,
                      cA_mean, cA_sig,
                      c0, c0_N, c0_S, 
                      rho_c1a, theta_c1a, rho_c1b, theta_c1b, 
                      mu_cA, rho_cA,  theta_cA, sigma_cA,  pi_cA,
                      phi_S2S, tau_0, phi_0):
        
        #convert freq to numpy array
        freq = np.array([freq]).flatten()

        #if ids and coordinates are pandas data-frame convert to numpy arrays
        eqid       = ConvertPandasDf2NpArray(eqid)
        ssn        = ConvertPandasDf2NpArray(ssn)
        cA_id      = ConvertPandasDf2NpArray(cA_id)
        c1a_Xutm   = ConvertPandasDf2NpArray(c1a_Xutm)
        c1b_Xutm   = ConvertPandasDf2NpArray(c1b_Xutm)
        cAmpt_Xutm = ConvertPandasDf2NpArray(cAmpt_Xutm)
        c1a_mean   = ConvertPandasDf2NpArray(c1a_mean)
        c1a_sig    = ConvertPandasDf2NpArray(c1a_sig)
        c1b_mean   = ConvertPandasDf2NpArray(c1b_mean)
        c1b_sig    = ConvertPandasDf2NpArray(c1b_sig)
        dS2S_mean  = ConvertPandasDf2NpArray(dS2S_mean)
        dS2S_sig   = ConvertPandasDf2NpArray(dS2S_sig)
        dBe_mean   = ConvertPandasDf2NpArray(dBe_mean)
        dBe_sig    = ConvertPandasDf2NpArray(dBe_sig)
        cA_mean    = ConvertPandasDf2NpArray(cA_mean)
        cA_sig     = ConvertPandasDf2NpArray(cA_sig)
        
        #convert eqid, ssn and cell_id to integers
        eqid  = eqid.astype(int)
        ssn   = ssn.astype(int)
        cA_id = cA_id.astype(int)
        #reserve eqid = -999 for new earthquake and station
        assert( ~(eqid < 0).any()),'Error. Negative EQ_IDs are not valid'
        assert( ~(ssn  < 0).any()),'Error. Negative SSNs are not valid'
        
        #initalization
        #initialize projection utm zone
        if self.zone_utm is None: 
            self.zone_utm = zone_utm
            self.proj_utm  = pyproj.Proj("+proj=utm +zone="+zone_utm+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        #initilalize station terms
        if not len(self.dS2S_mu):  self.dS2S_mu  = pd.DataFrame(ssn, columns=['id'])
        if not len(self.dS2S_sig): self.dS2S_sig = pd.DataFrame(ssn, columns=['id'])
        #initialize aleatory terms
        if not len(self.dBe_mu):  self.dBe_mu  = pd.DataFrame(eqid, columns=['id'])
        if not len(self.dBe_sig): self.dBe_sig = pd.DataFrame(eqid, columns=['id'])
        #initialize coefficient variables if empty
        if not len(self.c1a_mu):  self.c1a_mu  = pd.DataFrame(c1a_Xutm, columns=['x','y'])
        if not len(self.c1a_cov): self.c1a_cov = pd.DataFrame(c1a_Xutm, columns=['x','y'])
        if not len(self.c1b_mu):  self.c1b_mu  = pd.DataFrame(c1b_Xutm, columns=['x','y'])
        if not len(self.c1b_cov): self.c1b_cov = pd.DataFrame(c1b_Xutm, columns=['x','y'])
        #cell attenuation
        assert(not self.cell_id is None),'Error. Undefined cells'
        if not len(self.cA_mu):   self.cA_mu  = pd.DataFrame(np.vstack([cA_id,cAmpt_Xutm.T]).T, columns=['id','x','y'])
        if not len(self.cA_cov):  self.cA_cov = pd.DataFrame(np.vstack([cA_id,cAmpt_Xutm.T]).T, columns=['id','x','y'])
        #check consistency of existing frequencies with new one
        assert(self.zone_utm == zone_utm),'Error. Inconsistent UTM zones'
        assert( np.isin(cA_id, self.cell_id).all()),'Error. Inconsistent cell ids'
                
        #add new frequency if doesn’t exist in non-ergodic frequencies
        if not np.any( np.abs(self.nerg_freq - freq) < self.ftol ): self.nerg_freq.append(freq)
        c_f = self.GetFreqName(freq)
        
        #update coefficients with new frequency
        #station terms
        self.dS2S_mu  = self.UpdIndexCoeffData(self.dS2S_mu,  dS2S_mean, ssn,  c_f)
        self.dS2S_sig = self.UpdIndexCoeffData(self.dS2S_sig, dS2S_sig,  ssn,  c_f)
        #station terms
        self.dBe_mu   = self.UpdIndexCoeffData(self.dBe_mu,   dBe_mean,  eqid, c_f)
        self.dBe_sig  = self.UpdIndexCoeffData(self.dBe_sig,  dBe_sig,   eqid, c_f)
        #statially varying coefficients
        self.c1a_mu  = self.UpdSpatialCoeffData(self.c1a_mu,  c1a_mean, c1a_Xutm, c_f)
        self.c1a_cov = self.UpdSpatialCoeffData(self.c1a_cov, c1a_sig,  c1a_Xutm, c_f)
        self.c1b_mu  = self.UpdSpatialCoeffData(self.c1b_mu,  c1b_mean, c1b_Xutm, c_f)
        self.c1b_cov = self.UpdSpatialCoeffData(self.c1b_cov, c1b_sig,  c1b_Xutm, c_f)
        #anelastic attenuation
        self.cA_mu  = self.UpdCellCoeffData(self.cA_mu,  cA_mean, cA_id, cAmpt_Xutm, c_f)
        self.cA_cov = self.UpdCellCoeffData(self.cA_cov, cA_sig,  cA_id, cAmpt_Xutm, c_f)
        
        #udate hyper-parameters
        self.hyp.loc[c_f] = [c0, c0_N, c0_S, rho_c1a, theta_c1a, rho_c1b, theta_c1b, 
                             mu_cA, rho_cA,  theta_cA, sigma_cA,  pi_cA,
                             phi_S2S, tau_0, phi_0 ]
        
        #check id and coordinate consistency in coefficient data-frame
        assert( scipylalg.norm(self.dS2S_mu.id        - self.dS2S_sig.id)         < 1e-6 )
        assert( scipylalg.norm(self.dBe_mu.id         - self.dBe_sig.id)          < 1e-6 )
        assert( scipylalg.norm(self.c1a_mu[['x','y']] - self.c1a_cov[['x','y']])  < 1e-6 )
        assert( scipylalg.norm(self.c1b_mu[['x','y']] - self.c1b_cov[['x','y']])  < 1e-6 )
        assert( scipylalg.norm(self.cA_mu[['x','y']]  - self.cA_cov[['x','y']])   < 1e-6 )
        
    #Functions for updating non-ergodic coefficient data
    #--------------------------------------
    def UpdSpatialCoeffData(self, coeff_data, c_data, c_X, c_name):
        
        #update coefficinet dataframe with new coeff at existing locations 
        i_coeff = list()
        for c_x in c_X: 
            #distance to coeff coordinates
            dist2coeff = scipylalg.norm(coeff_data[['x','y']] - c_x, axis=1)
            #find coeff index based on coordinates, set to nan if non avail coor in coeff_data
            i_c = dist2coeff.argmin() if dist2coeff.min() < self.Xtol else np.nan
            i_coeff.append(i_c)
        i_coeff = np.array(i_coeff) #convert to numpy array
        #data on existing coordinates
        i_avail =~np.isnan(i_coeff)
        
        #update coefficient data-frame
        i_coeff = i_coeff[i_avail]
        coeff_data.loc[i_coeff,c_name] = c_data[i_avail]
        
        #new locations data-frame
        coeff_data_new = pd.DataFrame({'x':c_X[~i_avail,0], 'y':c_X[~i_avail,1], c_name: c_data[~i_avail]})
        
        #merge old and new locations data-frame
        coeff_data = pd.concat([coeff_data, coeff_data_new], axis=0).reset_index(drop=True)
        
        return coeff_data

    def UpdIndexCoeffData(self, coeff_data, c_data, c_ID, c_name):
        
        #update coefficinet dataframe with new coeff at existing locations for 
        i_coeff = list()
        for c_id in c_ID: 
            #find coeff index based on ids
            i_c = np.where(np.isin(coeff_data['id'], c_id))[0]
            i_c = np.nan if not i_c.size else i_c[0] #set to nan if non avail id in coeff_data
            i_coeff.append(i_c)
        i_coeff = np.array(i_coeff) #convert to numpy array
        #data on existing ids
        i_avail =~np.isnan(i_coeff)
        
        #update coefficient data-frame
        i_coeff = i_coeff[i_avail]
        coeff_data.loc[i_coeff,c_name] = c_data[i_avail]
        
        #new locations data-frame
        coeff_data_new = pd.DataFrame({'id':c_ID[~i_avail], c_name: c_data[~i_avail]})
        
        #merge old and new locations data-frame
        coeff_data = pd.concat([coeff_data, coeff_data_new], axis=0).reset_index(drop=True)
        
        return coeff_data
    
    def UpdCellCoeffData(self, cellcoeff_data, c_data, c_id, c_X, c_name):
        
        #update coefficinet dataframe with new coeff at existing locations 
        i_coeff = list()
        for c_i in c_id: 
            #find coeff index based on coordinates
            i_c = np.where(np.isin(cellcoeff_data['id'], c_i))[0]
            i_c = np.nan if not i_c.size else i_c[0] #set to nan if non avail coor in coeff_data
            i_coeff.append(i_c)

        i_coeff = np.array(i_coeff) #convert to numpy array
        #data on existing coordinates
        i_avail =~np.isnan(i_coeff)
        
        #update coefficient data-frame
        i_coeff = i_coeff[i_avail]
        cellcoeff_data.loc[i_coeff,c_name] = c_data[i_avail]
        
        #new locations data-frame
        cellcoeff_data_new = pd.DataFrame({'x':c_X[~i_avail,0], 'y':c_X[~i_avail,1], c_name: c_data[~i_avail]})
        
        #merge old and new locations data-frame
        cellcoeff_data = pd.concat([cellcoeff_data, cellcoeff_data_new], axis=0).reset_index(drop=True)
        
        return cellcoeff_data

    
    #Get non-ergodic coefficinets on coordinates of interest
    #--------------------------------------
    def Getc1a(self, freqs, eq_X, flag_consol = True):
        
        #sample mean and covariance of earthquake coefficient
        c1a_allfreq_mu  = list()
        c1a_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #hyper-parameters
            rho_eq_c   = self.hyp.rho_c1a[c_f]
            theta_eq_c = self.hyp.theta_c1a[c_f]
            #coordinates, mean and std of eq constant at known locations
            i_avail = ~np.isnan(self.c1a_mu[c_f].values)
            data_X       = self.c1a_mu[['x','y']].values[i_avail,:]
            data_c1a_mu  = self.c1a_mu[c_f].values[i_avail]
            data_c1a_sig = self.c1a_cov[c_f].values[i_avail]
            #predict earthquake constants at new locations
            c1a_mu, _, c1a_cov = pygp.SampleCoeffs(eq_X, data_X, c_data_mu=data_c1a_mu, c_data_sig=data_c1a_sig, hyp_rho=rho_eq_c, hyp_theta=theta_eq_c)
            c1a_cov = self.Convert2SparseCovMat(c1a_cov) if self.sp_cov else c1a_cov
            #summarize c1a for different frequencies
            c1a_allfreq_mu.append(c1a_mu)
            c1a_allfreq_cov.append(c1a_cov)
        #convert mean and covariance to array and matrix
        if flag_consol:
            c1a_allfreq_mu  = np.hstack(c1a_allfreq_mu)
            c1a_allfreq_cov = scipysp.block_diag(c1a_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*c1a_allfreq_cov)
        
        return c1a_allfreq_mu, c1a_allfreq_cov
    
    def Getc1b(self, freqs, sta_X, flag_consol = True):
        
        #sample mean and covariance of earthquake coefficient
        c1b_allfreq_mu  = list()
        c1b_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #hyper-parameters
            rho_sta_c   = self.hyp.rho_c1b[c_f]
            theta_sta_c = self.hyp.theta_c1b[c_f]
            #coordinates, mean and std of eq constant at known locations
            i_avail = ~np.isnan(self.c1b_mu[c_f].values)
            data_X       = self.c1b_mu[['x','y']].values[i_avail,:]
            data_c1b_mu  = self.c1b_mu[c_f].values[i_avail]
            data_c1b_sig = self.c1b_cov[c_f].values[i_avail]
            #predict station constants at new locations
            c1b_mu, _, c1b_cov = pygp.SampleCoeffs(sta_X, data_X, c_data_mu=data_c1b_mu, c_data_sig=data_c1b_sig, hyp_rho=rho_sta_c, hyp_theta=theta_sta_c)
            c1b_cov = self.Convert2SparseCovMat(c1b_cov) if self.sp_cov else c1b_cov
            #summarize c1a for different frequencies
            c1b_allfreq_mu.append(c1b_mu)
            c1b_allfreq_cov.append(c1b_cov)
        #convert mean and covariance to array and matrix
        if flag_consol:
            c1b_allfreq_mu  = np.hstack(c1b_allfreq_mu)
            c1b_allfreq_cov = scipysp.block_diag(c1b_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*c1b_allfreq_cov)
        
        return c1b_allfreq_mu, c1b_allfreq_cov
    
    def GetdS2S(self, freqs, ssn, flag_consol = True):
        
        #sample mean and covariance of deltaS2S
        dS2S_allfreq_mu  = list()
        dS2S_allfreq_cov = list()
        for f in freqs:
            # import pdb; pdb.set_trace()
            c_f = self.GetFreqName(f)
            #hyper-parameters
            phi_S2S   = self.hyp.phi_S2S[c_f]
            #coordinates, mean and std of sta constant at known locations
            i_avail = ~np.isnan(self.dS2S_mu[c_f].values)
            data_ssn      = self.dS2S_mu['id'].values[i_avail]
            data_dS2S_mu  = self.dS2S_mu[c_f].values[i_avail]
            data_dS2S_sig = self.dS2S_sig[c_f].values[i_avail]
            #perdic station term
            dS2S_mu, _, dS2S_cov = pygp.SampledS2S(ssn, data_ssn, dS2S_data_mu=data_dS2S_mu, dS2S_data_sig=data_dS2S_sig, phi_S2S=phi_S2S)
            dS2S_cov = self.Convert2SparseCovMat(dS2S_cov) if self.sp_cov else dS2S_cov
            #summarize dS2S for different frequencies
            dS2S_allfreq_mu.append(dS2S_mu)
            dS2S_allfreq_cov.append(dS2S_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dS2S_allfreq_mu  = np.hstack(dS2S_allfreq_mu)
            dS2S_allfreq_cov = scipysp.block_diag(dS2S_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dS2S_allfreq_cov)
        
        return dS2S_allfreq_mu, dS2S_allfreq_cov

    def GetcA(self, freqs, eq_X, sta_X, eq_z, flag_consol = True):
        #cell indices and path lengths
        i_cA_valid, L_cells, n_c = self.CellPaths(eq_X, sta_X, eq_z)
        cA_X    = self.cell_X[i_cA_valid,:]
        cAmpt_X = self.cmpt_X[i_cA_valid,:]
        L_cells = self.Convert2SparseCovMat(L_cells) if self.sp_cov else L_cells
        #sample mean and covariance of cell attenuation coefficients
        cA_allfreq_mu  = list()
        cA_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #hyper-parameters
            mu_cA    = self.hyp.mu_cA[c_f]
            rho_cA   = self.hyp.rho_cA[c_f]
            theta_cA = self.hyp.theta_cA[c_f]
            sigma_cA = self.hyp.sigma_cA[c_f]
            pi_cA    = 0.
            #coordinates, mean and std of sta constant at known locations
            i_avail = ~np.isnan(self.cA_mu[c_f].values)
            data_X      = self.cA_mu[['x','y']].values[i_avail,:]
            data_cA_mu  = self.cA_mu[c_f].values[i_avail]
            data_cA_sig = self.cA_cov[c_f].values[i_avail]
            #predict anelastic attenuation
            cA_mu, _, cA_cov = pygp.SampleAttenCoeffsNegExp(cAmpt_X, data_X,
                                                            cA_data_mu=data_cA_mu, cA_data_sig=data_cA_sig,
                                                            mu_cA=mu_cA, rho_cA=rho_cA, theta_cA=theta_cA, 
                                                            sigma_cA=sigma_cA, pi_cA=pi_cA)
            cA_mu[cA_mu>0] = 0.
            cA_cov = self.Convert2SparseCovMat(cA_cov) if self.sp_cov else cA_cov
            #summarize cA for different frequencies
            cA_allfreq_mu.append(cA_mu)
            cA_allfreq_cov.append(cA_cov)
        L_allfreq_cells = [L_cells]*len(freqs)
        #convert mean and covariance to array and matrix
        if flag_consol:
            cA_allfreq_mu   = np.hstack(cA_allfreq_mu)
            cA_allfreq_cov  = scipysp.block_diag(cA_allfreq_cov,  format='csc') if self.sp_cov else scipylalg.block_diag(*cA_allfreq_cov)
            L_allfreq_cells = scipysp.block_diag(L_allfreq_cells, format='csc') if self.sp_cov else scipylalg.block_diag(*L_allfreq_cells)

        return cA_allfreq_mu, cA_allfreq_cov, L_allfreq_cells, L_cells, n_c

    #Get aleatory terms
    #--------------------------------------
    def GetdBe(self, freqs, eqid=None, npt=None, flag_consol = True):
        
        #earthquake id if undefined
        eqid = -1 - np.arange(npt) if eqid is None else eqid
        #sample mean and covariance of deltaS2S
        dBe_allfreq_mu  = list()
        dBe_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #hyper-parameters
            tau_0 = self.hyp.tau_0[c_f]
            #
            i_avail = ~np.isnan(self.dBe_mu[c_f].values)
            data_eqid    = self.dBe_mu['id'].values[i_avail]
            data_dBe_mu  = self.dBe_mu[c_f].values[i_avail]
            data_dBe_sig = self.dBe_sig[c_f].values[i_avail]
            #predict anelastic attenuation
            dBe_mu,  _, dBe_cov = pygp.SampledBe(eqid, data_eqid, dB_data_mu=data_dBe_mu, tau_0=tau_0)
            dBe_cov = self.Convert2SparseCovMat(dBe_cov) if self.sp_cov else dBe_cov
            #summarize dBe for different frequencies
            dBe_allfreq_mu.append(dBe_mu)
            dBe_allfreq_cov.append(dBe_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dBe_allfreq_mu  = np.hstack(dBe_allfreq_mu)
            dBe_allfreq_cov = scipysp.block_diag(dBe_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dBe_allfreq_cov)
        
        return dBe_allfreq_mu, dBe_allfreq_cov

    def GetdWe(self, freqs, eqid=None, npt=None, flag_consol = True):
        
        #covariance of within event residual
        npt = npt if eqid is None else len(eqid)
        dWe_allfreq_cov = list()
        for f in freqs:
            c_f = self.GetFreqName(f)
            #hyper-parameters
            phi_0 = self.hyp.phi_0[c_f]
            #within event covariance matrices
            dWe_cov = phi_0**2 * np.eye(npt)
            dWe_cov = self.Convert2SparseCovMat(dWe_cov) if self.sp_cov else dWe_cov
            #summarize dBe for different frequencies
            dWe_allfreq_cov.append(dWe_cov)       
        #convert mean and covariance to array and matrix
        if flag_consol:
            dWe_allfreq_cov = scipysp.block_diag(dWe_allfreq_cov, format='csc') if self.sp_cov else scipylalg.block_diag(*dWe_allfreq_cov)
        
        return dWe_allfreq_cov

