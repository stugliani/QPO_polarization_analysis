import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pf
import os
from scipy.optimize import curve_fit
import math
from itertools import chain
from tqdm import tqdm
from scipy.stats import t

import argparse

from stingray.base import StingrayTimeseries
from stingray import Powerspectrum
from ixpeobssim.binning.misc import xBinnedLightCurve
import ixpeobssim.core.pipeline as pipeline
from ixpeobssim.evt.event import xEventFile


class TimeBase():
    
    def __init__(self, files, du_id):
        self.du_id = du_id-1
        self.file_list = files
        self.tstart = self.get_TIME()[0]
        self.tstop = self.get_TIME()[1]

    def get_TIME(self):
        tstart = []
        tstop = []
        for f in self.file_list:
            event_file = xEventFile(f)
            tstart.append(event_file.start_met())
            tstop.append(event_file.stop_met())
        return np.min(tstart), np.max(tstop)
    
    def readsimfitsfile(self):
        """
            Function reads fits file
            it returns the events and the GTI
        """
        data_f = pf.open(self.file_list[self.du_id])
        data_f.info()
        events = data_f['EVENTS'].data
        GTI = data_f['GTI'].data
        return events, GTI

    def EventTimes(self):
        events_ = self.readsimfitsfile()[0]
        return events_['TIME']

    def GTI(self):
        GTI_ = self.readsimfitsfile()[1]
        # I build the GTI array from the GTI of the observation 
        # in such a way that stingray can use and understand it
        gti=[]
        for i in range(len(GTI_)):
            gti.append([GTI_[i][0],GTI_[i][1]])
        return gti
    
    def Livetime(self):
        ef = xEventFile(self.file_list[self.du_id])
        return ef.livetime()



class Times(TimeBase):

    def __init__(self, energy_binning, grayfilter_bool, acceptance_correction, files, tbins, du_id):
        TimeBase.__init__(self, files, du_id)
        self.energy_binning = energy_binning
        self.grayfilter = grayfilter_bool
        self.acceptcorr = acceptance_correction
        self.file_list = files
        self.tbins = tbins

    def get_TIME(self):
        tstart = []
        tstop = []
        for f in self.file_list:
            event_file = xEventFile(f)
            tstart.append(event_file.start_met())
            tstop.append(event_file.stop_met())
        return np.min(tstart), np.max(tstop)
    
    def LC(self):
        """
            Function that creates light curve _LC fits 
            files and returns an xBinnedLightCurve object
        """
        LC_LIST = pipeline.xpbin(*self.file_list, algorithm='LC',tbins=self.tbins, tmin=self.tstart, tmax=self.tstop, 
                    ebinning=self.energy_binning, overwrite=True, 
                    grayfilter=self.grayfilter,acceptcorr=self.acceptcorr)
        lightcurve = xBinnedLightCurve.from_file_list(LC_LIST)
        # os.remove(self.path+'ixpe02250901_det1_evt2_v01_src_lc.fits')
        # # os.remove(PATH+'ixpe02250901_det2_evt2_v01_src_lc.fits')
        # # os.remove(PATH+'ixpe02250901_det3_evt2_v01_src_lc.fits')
        return lightcurve

    def getLC(self):
        """
            Function that returns the LC rate, error and time from an
            xBinnedLightCurve object (lightcurve) and the LC bins (tbins)
        """
        light_curve, light_curve_error, time_LC = [], [], []
        for i in range(self.tbins-1):
            light_curve.append(self.LC().COUNTS[i]/self.LC().EXPOSURE[i])
            light_curve_error.append(self.LC().ERROR[i]/self.LC().EXPOSURE[i])
            time_LC.append(self.LC().TIME[i])
        return light_curve, light_curve_error, time_LC
    
    def rate(self):
        return self.getLC()[0]

    def rate_err(self):
        return self.getLC()[1]
    
    def time(self):
        return self.getLC()[2]

    def show_LC(self,plot_intervals):
        """
            Function that plots the Light Curve from an
            the LC rate, error and time
        """
        _, ax = plt.subplots()
        ax.errorbar(self.time(), self.rate(), yerr=self.rate_err,xerr=0,linestyle='',marker='.')
        if len(plot_intervals) !=0:
            for T in plot_intervals:
                ax.axvline(x=T, linestyle='--')
        ax.set_ylabel('Rate [Hz]')
        ax.set_xlabel('MET [s]')
        ax.grid(True)


class PowerSpectrumQPO:
    
    def __init__(self, stingray_ts, dt, segment_int, segment_tot):
        self.sting_ts = stingray_ts
        self.dt = dt
        self.segment_tot = segment_tot
        self.segment_int = segment_int
        self.A = 0.
        self.B = 0.
        self.C = 0.
        self.Freq = self.PowerSpectrum()[0]
        self.Power = self.PowerSpectrum()[1]
        self.Power_err = self.PowerSpectrum()[2]
        # # parameters = [norm1, center1, hwhm1, nomr2, center2, hwhm2, A, B, C]
        # self.parameters = parameters
        # self.norm1 = parameters[0]
        # self.center1 = parameters[1]
        # self.hwhm1 = parameters[2]
        # self.norm2 = parameters[3]
        # self.center2 = parameters[4]
        # self.hwhm2 = parameters[5]
        # self.A = parameters[]

    def Gaussian(self, x, A, sig, mu):
        return A/np.sqrt(2*math.pi)/sig*np.exp(-1/2*((x-mu)/sig)**2)
    
    def Exp(self, x, A, B, C):
        return A*np.exp(-B*x)+C

    def Lorentzian(self, x, norm, center, hwhm):
        return (1/np.pi)*(norm * hwhm / ((x - center) ** 2 + hwhm ** 2))

    def LorentzianExp(self, x, norm, center, hwhm, A, B, C):
        return (1/np.pi)*(norm * hwhm / ((x - center) ** 2 + hwhm ** 2)) + A*np.exp(-B*x) + C

    def DoubleLorentzianExp(self, x, norm1, center1, hwhm1, norm2, center2, hwhm2, A, B, C):
        return (1/np.pi)*(norm1 * hwhm1 / ((x - center1) ** 2 + hwhm1 ** 2)) + (1/np.pi)*(norm2 * hwhm2 ** 2 / ((x - center2) ** 2 + hwhm2 ** 2)) + A*np.exp(-B*x) + C

    def LorentzianExpFixed(self, x, norm, center, hwhm):
        return (1/np.pi)*(norm * hwhm / ((x - center) ** 2 + hwhm ** 2)) + self.A*np.exp(-self.B*x) + self.C
    
    def LorentzianBaseline(self, x, norm, center, hwhm):
        return (1/np.pi)*(norm * hwhm / ((x - center) ** 2 + hwhm ** 2)) + self.C

    def PowerSpectrumTS(self, ts):
        """
            Function that returns the frequencies and powers 
            of a power spectrum performed using stingray. 
            The time resolution dt and the segment size are 
            passed as arguments.
        """
        ps = Powerspectrum.from_time_array(ts.time, dt=self.dt, gti=ts.gti, segment_size=self.segment_int,norm="none") ## no normalization
        ####ps = Powerspectrum.from_time_array(ts.time, dt=self.dt, gti=ts.gti, segment_size=self.segment_int,norm="leahy")
        return ps.freq, ps.power
    
    def PowerSpectrum(self):
        """
            Function that returns the frequencies and powers 
            of a power spectrum performed using stingray. 
            The time resolution dt and the segment size are 
            passed as arguments.
        """
        ps = Powerspectrum.from_time_array(self.sting_ts.time, dt=self.dt, gti=self.sting_ts.gti, segment_size=self.segment_tot,norm="none") ## no normalization
        ####ps = Powerspectrum.from_time_array(self.sting_ts.time, dt=self.dt, gti=self.sting_ts.gti, segment_size=self.segment_tot,norm="leahy")
        return ps.freq, ps.power, ps.power_err
    
    def DoubleLorentzianExp_Fit(self, initial_guess, bounds):
        """
            Fit of the power spectrum with
            2 QPOs (Lorentzians) and an
            exponential term.
        """
        par = 0
        cov = 0
        try:
            par, cov = curve_fit(self.DoubleLorentzianExp, self.Freq, self.Power, p0=initial_guess, bounds=bounds)
        except RuntimeError:
            pass
        return par, cov
    
    def DoubleLorentzianExp_Fit_Norm(self, initial_guess, bounds, norm):
        """
            Fit of the power spectrum with
            2 QPOs (Lorentzians) and an
            exponential term. NORMALIZED case.
        """
        par = 0
        cov = 0
        try:
            par, cov = curve_fit(self.DoubleLorentzianExp, self.Freq, self.Power/norm, p0=initial_guess, bounds=bounds)
        except RuntimeError:
            pass
        return par, cov
    
    def LorentzianExpFixed_Fit(self, initial_guess, bounds):
        """
        """
        par = 0
        cov = 0
        try:
            par, cov = curve_fit(self.LorentzianExpFixed, self.Freq, self.Power, p0=initial_guess, bounds=bounds)
        except RuntimeError:
            pass
        return par, cov
    
    def LorentzianBaseline_Fit(self, initial_guess, bounds):
        """
        """
        par = 0
        cov = 0
        try:
            par, cov = curve_fit(self.LorentzianBaseline, self.Freq, self.Power, p0=initial_guess, bounds=bounds)
        except RuntimeError:
            pass
        return par, cov
    
    def LorentzianExpFixed_Fit_Norm(self, initial_guess, bounds, norm):
        """
        """
        par = 0
        cov = 0
        try:
            par, cov = curve_fit(self.LorentzianExpFixed, self.Freq, self.Power/norm, p0=initial_guess, bounds=bounds)
        except RuntimeError:
            pass
        return par, cov
    
    def FindMaxInInterval(self, X, Y, mu, sigma): 
        """ 
            Function that returns the maximum 
            power in a specified frequency range, 
            evaluated using the parameters peak 
            as center of the range and sigma to  
            calculate the edges. 
        """ 
        sup = mu + sigma 
        inf = mu - sigma 
        mask = np.where((X > inf) & (X < sup))[0] 

        if len(mask) > 0: 
            sup = mu + 0.1 * mu 
            inf = mu - 0.1 * mu 
            mask = np.where((X > inf) & (X < sup))[0] 
    
        max_Y = np.max(Y[mask]) 
    
        return max_Y



    
    def Compatibility(self,a,aErr,b,bErr,nsigma):
        delta = abs(a - b) / np.sqrt(aErr**2 + bErr**2)
        if delta <= nsigma:
            comp = True
        else:
            comp = False
        return delta, comp
    
    def MakeThresholds(self,array,cl_quantile,steps):
        q_sup = 1.-cl_quantile/2
        q_inf = 0.+cl_quantile/2
        quant_sup = np.quantile(array,q_sup)
        quant_inf = np.quantile(array,q_inf)
        quant_sup = np.round(quant_sup,2)
        quant_inf = np.round(quant_inf,2)
        thr_ = np.linspace(quant_inf,quant_sup,int((quant_sup-quant_inf)/steps))
        return np.round(thr_,3)
    
    def MakeNorme(self,freqs,powers,par,qpo,hwhm,norme,max_norm_array):
        if not isinstance(par, int):
            norm_ = par[0]/np.pi
            delta_, compat = self.Compatibility(par[1],par[2],qpo,hwhm,3)
            if compat == True:
                max_norm = self.FindMaxInInterval(freqs,powers,par[1],par[2])
            else:
                max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
        else:
            norm_ = par/np.pi
            max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
        max_norm_array.append(max_norm-self.C)
        norme.append(norm_)
        return norm_, max_norm-self.C  
    
    def MakeNorme_Norm(self,freqs,powers,par,qpo,hwhm,norme,max_norm_array,norm):
        #powers = powers
        if not isinstance(par, int):
            norm_ = par[0]#/np.pi
            delta_, compat = self.Compatibility(par[1],par[2],qpo,hwhm,3)
            if compat == True:
                max_norm = self.FindMaxInInterval(freqs,powers,par[1],par[2])
            else:
                max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
        else:
            norm_ = par#/np.pi
            max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
        max_norm = (max_norm-self.C)/norm
        max_norm_array.append(max_norm)
        norme.append((norm_)/norm)
        return (norm_)/norm, max_norm
    
    def MakeNormeAree_Norm(self,freqs,powers,par,qpo,hwhm,norme,max_norm_array,norm):
        if not isinstance(par, int):
            norm_ = par[0]
            delta_, compat = self.Compatibility(par[1],par[2],qpo,hwhm,3)
            if compat == True:
                max_norm = self.FindMaxInInterval(freqs,powers,par[1],par[2])
                max_norm = (max_norm-self.C)*np.pi*par[2]
            else:
                max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
                max_norm = (max_norm-self.C)*np.pi*hwhm
        else:
            norm_ = par
            max_norm = self.FindMaxInInterval(freqs,powers,qpo,hwhm)
            max_norm = (max_norm-self.C)*np.pi*hwhm
        max_norm_array.append(max_norm/norm)
        norme.append((norm_)/norm)
        return norm_/norm, max_norm/norm 

    def GetTimeMask(self,event_times, sel_times_array,fast):
        if fast==False:
            mask = np.zeros(len(event_times)).astype(bool)  # I create a mask array long as event the main array filled with False
            for i, t in enumerate(tqdm(sel_times_array)):
                #print(file_path, '  ',"%.2f" % float(i/len(sel_times_array)*100), ' %')
                iii = np.where(event_times == t)[0]         # iii is the array of the indices of the main array element equal to the t element of the selected array
                if len(iii) > 0:
        #            print('ok', t_[iii], t)
                    mask[iii] = bool(1)                     # if there are elements in the main array equal to the element t of the selected array, 
                else:                                       # I put the corresponding bool mask element equal to True
                    abs_differences = abs(event_times - t)
                    best_iii = np.argmin(abs_differences)
                    mask[best_iii] = bool(0)
        #            print('---- NOT ok', t_[best_iii], t)
            return mask
        if fast==True:
            mask_fast = np.isin(event_times, sel_times_array)
            return mask_fast
        else:
            print('Specify if fast or not')
        
    
    def GetSingleDUBooMask(self, du_id, event_time_array, du_id_array, du_id_array_selected, event_time_array_selected, fast):
    
        # I select the events in the main time serie of the detector unit du_id
        mask_du_before_selection = np.where(du_id_array==du_id)[0]
        events_du_before_selection = event_time_array[mask_du_before_selection]
        # I select the events in the selected time serie of the detector unit du_id
        mask_du_after_selection = np.where(du_id_array_selected==du_id)[0]
        events_du_after_selection = event_time_array_selected[mask_du_after_selection]
        # I create the mask_bool of putting True if the event is selected and False if the event is not selectes, always of the DU du_id
        mask_du_bool = self.GetTimeMask(event_times=events_du_before_selection,sel_times_array=events_du_after_selection,fast=fast)
        return mask_du_bool
    
    def MakeDir(self,dir,info=True):
        # creation of a folder if it doesn't exists
        if not os.path.exists(dir):  # check if the folder exists
            os.makedirs(dir)         # creation of a folder ONLY if it doesn't exists
            if info:
                print(f"[INFO] Creating folder {dir}")
                print(f"[INFO] Saving in folder {dir}")
        else:
            if info:
                print(f"[INFO] Saving in folder {dir}")
            else:
                pass


class T_TEST_STAT:
    def __init__(self, X_1, X_ERR_1, X_2, X_ERR_2, N_1, N_2, CL):
        self.X_1 = X_1
        self.X_2 = X_2
        self.N_1 = N_1
        self.N_2 = N_2
        self.X_ERR_1 = X_ERR_1*np.sqrt(self.N_1)
        self.X_ERR_2 = X_ERR_2*np.sqrt(self.N_2)
        self.CL = CL

    def t(self):
        num_ = self.X_1 - self.X_2
        den_ = self.X_ERR_1**2/self.N_1 + self.X_ERR_2**2/self.N_2
        return num_/np.sqrt(den_)

    def dof(self):
        num_ = self.X_ERR_1**2/self.N_1 + self.X_ERR_2**2/self.N_2
        den_ = (self.X_ERR_1**2/self.N_1)**2/(self.N_1-1) + (self.X_ERR_2**2/self.N_2)**2/(self.N_2-1)
        return num_**2/den_
    
    def p_value(self):
        p_value_ = 2 * (1 - t.cdf(abs(self.t()), self.dof()))
        return p_value_

class Polarization:
    def __init__(self, qn, qn_err, un, un_err):
        self.qn = qn
        self.un = un
        self.qn_err = qn_err
        self.un_err = un_err

    def PD(self):
        pd_ = self.qn**2 + self.un**2
        return np.sqrt(pd_)
    
    def PD_ERR(self):
        return self.qn_err
    
    def PA(self):
        psi = 0.5*np.arctan2(self.un,self.qn)
        return np.degrees(psi)
    
    def PA_ERR(self):
        arg = (self.qn*self.un_err)**2 + (self.un*self.qn_err)**2
        pd_ = self.PD()
        return (1/(2*pd_**2))*np.sqrt(arg)

    def QN(self):
        return self.qn

    def UN(self):
        return self.un 
    
    def QN_ERR(self):
        return self.qn

    def UN_ERR(self):
        return self.un 
        
    
