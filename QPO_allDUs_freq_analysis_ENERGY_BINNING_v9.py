# import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from QPO_classes import Times, PowerSpectrumQPO
import argparse
from stingray.base import StingrayTimeseries
from stingray import Powerspectrum
from tqdm import tqdm
import astropy.io.fits as pf
from ixpeobssim.evt.event import xEventFile ##per energia!!
import inspect
from ixpeobssim.evt import event
from uncertainties import ufloat 





# define flags to launch the code: segment size, method, thresholds (min,max,step), energy range
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
#parser.add_argument('-du','--DU', type=int,  help='DU number ypu want to analyze', required=True)
parser.add_argument('-seg','--seg-size', type=int,  help='Duration of the time segments in seconds', required=True)
parser.add_argument('-LC','--LightCurve', action='store_true',  help='Do you want to see the Light Curve ?', required=False, default=False)
parser.add_argument('-show','--show-seg-PS', action='store_true',  help='Show the Power Spectra for all the time segments ?', default=False, required=False)
parser.add_argument('-method','--method-selection', type=str,  help='method to select the good time intervals',choices=['max_norm','par_norm','hybrid'] , default='max_norm', required=True)
parser.add_argument('-s','--save', action='store_true',  help='Do you want to save the masks ?', required=False, default=False)
parser.add_argument('-min','--min-thr', type=float, help='minimum threshold value', required=False)
parser.add_argument('-max','--max-thr', type=float, help='maximum threshold value', required=False)
parser.add_argument('-step','--step-thr', type=float, help='threshold step', required=False)
parser.add_argument('-emin','--energy-min', type=float, help='Minimum energy in keV', required=True)
parser.add_argument('-emax','--energy-max', type=float, help='Maximum energy in keV', required=True)

args = parser.parse_args()

# there i give an example of CODE CALL:
# python QPO_allDUs_freq_analysis_ENERGY_BINNING_v6.py -seg 70 -method hybrid -min 0.1 -max 0.5 -step 0.01 -emin 2.0 -emax 3.5 -s
# MIND THE -s AT THE END TO SAVE EVERYTHING YOU NEED!

#n = args.DU
SEG_SIZE = args.seg_size
LightCurve = args.LightCurve
show_all = args.show_seg_PS
method = args.method_selection
SAVE = args.save
#min_thr = args.min_thr
#max_thr = args.max_thr
#step_thr = args.step_thr
#threshold_array = np.linspace(min_thr, max_thr, int((max_thr-min_thr)/step_thr)+1)
E_MIN = args.energy_min
E_MAX = args.energy_max
ENERGY_BIN_LABEL = f'E{E_MIN}-{E_MAX}keV'





# Insert here path to data files (change according to observation number!)
# for energy analysis
PATH = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'
# for standard analysis
#PATH = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_python_OLD_version/Obs1/'


# insert here the observation number (1,2,3,4,5)
n_obs = 1 

#comment/decomment next linea to upload the correct files according to the considered observation

# Obs1
DU = [PATH+'ixpe02250901_det1_evt2_v01_src.fits', PATH+'ixpe02250901_det2_evt2_v01_src.fits', PATH+'ixpe02250901_det3_evt2_v01_src.fits']

# Obs2
# DU = [PATH+'ixpe02251001_det1_evt2_v01_src.fits', PATH+'ixpe02251001_det2_evt2_v01_src.fits', PATH+'ixpe02251001_det3_evt2_v01_src.fits']


# upload numpies for your observation (we generated them in the previous code block "make_GTI")
event_times = np.load(PATH+'times_final.npy')
gti_starts = np.load(PATH+'gti_starts_merged.npy')
gti_stops = np.load(PATH+'gti_stops_merged.npy')
du_id_final = np.load(PATH+'du_id_final.npy')
energy_final = np.load(PATH+'energy_final.npy')





# don't touch these parameters (unless you know what you are doing :) )
# these are two key parameters in the temporal-frequency (STINGRAY) analysis
# DT is temporal binning for photons (n_bin=segment_size/dt)
DT = 0.02
# SEGMENT_SIZE defines the resolution of the Fourier analysis (Delta f = 1/ SEGMENT_SIZE)
# In other words, SEGMENT_SIZE defines the time domain used to compute power spectra with Stingray: the global power spectrum is the mean of these segments with the chosen SEGMENT_SIZE (there's too noise to calculate the power spectrum on the entire observation)
# NW! Don't confuse SEGMENT_SIZE that relates to the global analysis with -SEG (flag passed with the code call) that regards the various time segments
SEGMENT_SIZE = 10





# energy mask
mask_E = (energy_final >= E_MIN) & (energy_final <= E_MAX)
event_times = event_times[mask_E]
du_id_final = du_id_final[mask_E]






#f = pf.open(DU[0]) 
#events = f['EVENTS'].data

#ef = xEventFile(DU[0])
#print(ef.event_data.dtype.names)
#print(events.columns)  
#print([m for m in dir(ef) if not m.startswith('_')])

#print(f"PI: min={events['PI'].min()}, max={events['PI'].max()}")

#print(ef.event_data.dtype.names)
#if 'ENERGY' in ef.event_data.dtype.names:
 #   energy = ef.event_data['ENERGY']
  #  print(f"Energy: min={energy.min()}, max={energy.max()}")



# parameters for PCUBE and LC
# NW! This range isn't related to the energy analysis, but it's used for the global lightcurve computing (note that 2-8 keV it's IXPE energy band!)
ENERGY_BINNING = [2., 8.] 
# the next filter is a boolean filter used to filter IXPE photons specifically for ultraluminous sources like Swift
grayfilter_bool = True
# the next filter is for IXPE acceptance correction
acceptance_correction = False





# define the time object TIMES using 'Times' class (see definition in QPO_classes) and giving as input the DUs files list, energy band parameters and lightcurve resolution
# TIMES stores lightcurve parameters and configuration for later use
TIMES = Times(energy_binning=ENERGY_BINNING, grayfilter_bool=grayfilter_bool, acceptance_correction=acceptance_correction, files=DU, tbins=1000, du_id=1)





# create a list containing the couples of GTIs start and stop times (that are vectors generated in code block make_GTI by making the logical AND of the three DUs GTIs)
gti=[]
for i in range(len(gti_starts)):
    gti.append([gti_starts[i],gti_stops[i]])


# NW! The following passage is key in the analysis!
# produce the StingrayTimeseries object ts: giving as input event_times (all filtered photons) and GTIs, it's created the base time structure on which Stingray later computes all the Power Spectra of both the global observation and single segments
ts = StingrayTimeseries(time=event_times,gti=gti)





"""
    I set the initial guesses for the paramenters of the fit
    and the bounds, both for the single and double Lorentzian fit.
"""

# initial guesses and bounds for OBS1
norm1 = 45000
center1 = 1.35
hwhm1 = 0.1
norm2 = 15000
center2 = 2.68
hwhm2 = 0.1
A = 10
B = 1
C = 7500


initial_guesses2 = [norm1,center1,hwhm1,norm2,center2,hwhm2,A,B,C]
bounds2 = ([0.0, 1., 0.0, 0., 0., 0., 0., 0., 0.], [np.inf, 2., 1., np.inf, np.inf, np.inf, np.inf, 10., 10000])

# initial guesses and bounds for OBS2

#norm1 = 7852.85      
#center1 = 2.78        
#hwhm1 = 0.308
#norm2 = 3881.0       
#center2 = 5.464       # NW!: double of fundamental (~2 * center1)
#hwhm2 = 0.6          
#A = 12641.82
#B = 1.952
#C = 7572.17 

#max_k_guess = 16000    #initial intensity guess of qpo1

#initial_guesses2 = [norm1,center1,hwhm1,norm2,center2,hwhm2,A,B,C]
#bounds2 = (   [0.0, 2.0, 0.05, 0.0, 4.5, 0.10, 0.0, 0.0, 0.0],     [np.inf, 4.0, 0.5, np.inf, 7.0, 0.70, np.inf, 10.0, np.inf] # Limiti superiori (max: hwhm2 max = 0.80))





# NW!: Another key passage!
# PSQPO is an instance of the PowerSpectrumQPO class (see QPO_classes) that includes all the necessary items to model Power Spectra
# segment_int: used for intermediate power spectra, time-lenght of single segments used to compute "intermediate" Power Spectra that are mediated to return the global Power Spectrum.
# segment_tot: used for global power spectrum, time-lenght of Global Power Spectrum
PSQPO = PowerSpectrumQPO(stingray_ts=ts, dt=DT, segment_int=10, segment_tot=50)





# create directories to store output files
du_folder = PATH+f'ALLDU_v'
du_seg_folder = PATH+f'ALLDU_v/seg{SEG_SIZE}s_{ENERGY_BIN_LABEL}'
PSQPO.MakeDir(du_folder)
PSQPO.MakeDir(du_seg_folder)




# NW! Another key point of the analysis
# fit of Global Power Spectrum
# par: array containing best fit parameters 
# cov: covariance matrix of fit
par, cov = PSQPO.DoubleLorentzianExp_Fit(initial_guess=initial_guesses2, bounds=bounds2)





# all the parameters extracted from fit are converted in floats
qpo_norm1 = float("%.3f"%par[0])
qpo_norm1_err = float("%.3f"%np.sqrt(cov[0][0]))
qpo_norm1_true = float("%.3f"%(par[0]/np.pi))
qpo1 = float("%.3f"%par[1])
qpo1_err = float("%.3f"%np.sqrt(cov[1][1]))
qpo_hwhm1 = float("%.3f"%par[2])
qpo_hwhm1_err = float("%.3f"%np.sqrt(cov[2][2]))
qpo_norm2 = float("%.3f"%par[3])
qpo_norm2_err = float("%.3f"%np.sqrt(cov[3][3]))
qpo_norm2_true = float("%.3f"%(par[3]/np.pi))
qpo2 = float("%.3f"%par[4])
qpo2_err = float("%.3f"%np.sqrt(cov[4][4]))
qpo_hwhm2 = float("%.3f"%par[5])
qpo_hwhm2_err = float("%.3f"%np.sqrt(cov[5][5]))
exp_norm = float("%.3f"%par[6])
exp_norm_err = float("%.3f"%np.sqrt(cov[6][6]))
exp_tau = float("%.3f"%par[7])
exp_tau_err = float("%.3f"%np.sqrt(cov[7][7]))
exp_const = float("%.3f"%par[8])
exp_const_err = float("%.3f"%np.sqrt(cov[8][8]))

# save area (integrated norm) of first QPO found in global fit, later it's used to normalize the power spectrum
area_qpo = par[0] 

uqpo1 = ufloat(qpo1, qpo1_err)
uhwhm1 = ufloat(qpo_hwhm1,qpo_hwhm1_err)
unorm1 = ufloat(qpo_norm1,qpo_norm1_err)
uqpo2 = ufloat(qpo2, qpo2_err)
uhwhm2 = ufloat(qpo_hwhm2,qpo_hwhm2_err)
unorm2 = ufloat(qpo_norm2,qpo_norm2_err)
uA = ufloat(exp_norm,exp_norm_err)
uB = ufloat(exp_tau,exp_tau_err)
uC = ufloat(exp_const,exp_const_err)



# create mask of frequency interval between qpo and \pm 1HWHM
#mask_qpo_window = ((PSQPO.Freq >= qpo1 - 3*qpo_hwhm1) & (PSQPO.Freq <= qpo1 + 3*qpo_hwhm1))
# find max intensity in the aforementioned window
#max_qpo_power = float(np.max(PSQPO.Power[mask_qpo_window]))





# calculate percentage of energy selected events
n_events_tot = len(np.load(PATH + 'times_final.npy'))  
n_events_selected = len(event_times)  
perc_selected = (n_events_selected / n_events_tot) * 100





# plot of Global Power Spectrum 
plt.figure('POWER SPECTRUM',figsize=(10,7))
plt.plot(PSQPO.Freq,PSQPO.Power,marker='.',linestyle='')
x = np.linspace(np.min(PSQPO.Freq),np.max(PSQPO.Freq),1000)
plt.plot(x,PSQPO.DoubleLorentzianExp(x,par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8]),marker='',linestyle='-',label='(Lorentzian qpo1)+\n(Lorentzian qpo2)+\n(exp)\n')
plt.plot(x,PSQPO.Lorentzian(x,par[0],par[1],par[2])+par[8],marker='',linestyle='--',label=f'Lorentzian qpo1\n qpo = ({uqpo1:.1uP}) Hz\n hwhm = ({uhwhm1:.1uP}) Hz\n norm = {unorm1:.1uP}')
plt.plot(x,PSQPO.Lorentzian(x,par[3],par[4],par[5])+par[8],marker='',linestyle='--',label=f'Lorentzian qpo2\n qpo = ({uqpo2:.1uP}) Hz\n hwhm = ({uhwhm2:.1uP}) Hz\n norm = {unorm2:.1uP}')
plt.plot(x,PSQPO.Exp(x,par[6],par[7],par[8]),marker='',linestyle='--',label=f'exp: ' + r'$A e^{-B\nu}+C$' + f'\n A = {uA:.1uP} \n B = ({uB:.1uP}) Hz$^{{-1}}$ \n C = {uC:.1uP}')
plt.plot([],[], ' ', label=f'Events in {ENERGY_BIN_LABEL}: \n {n_events_selected:.2e} \n ({perc_selected:.1f}% of total)')
plt.title(f'Global fit of power spectrum: Obs {n_obs} - {ENERGY_BIN_LABEL}')
plt.xscale('log')
plt.grid('both')
plt.xlabel('frequency [Hz]')
plt.ylabel('power')
plt.xlim([np.min(PSQPO.Freq),np.max(PSQPO.Freq)])
plt.legend(fontsize=10)





# extract model: frequencies and global best fit (double lorentzian) parameters are given as input and a new array of the same lenght as PSQPO.Freq containing teoretical esteemed values is returned
model_power = PSQPO.DoubleLorentzianExp(PSQPO.Freq,par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8])
# create array containing 1 sigma errors on power for each frequency bin
sigma_ps = PSQPO.Power_err





# save Global Power Spectrum plot
figname_ps = f'PS_ALLDU.png'
PSQPO.MakeDir(du_seg_folder+'/images')
plt.savefig(du_seg_folder+'/images/'+figname_ps, dpi=300)
# clear figure
plt.close()





# breakpoint after the global power spectrum plot: check if the fit is ok, maybe you want to refine the initial guesses and bounds parameters
# if the fit is ok, type 'c' in your terminal to go on
breakpoint()





"""
    Now I can analyze the segments and fit the function 
    power_spectrum in each of them. Note that the function
    power_spectrum needs to have a StingrayTimeseries object
    as only argument.
"""

def power_spectrum(ts):
    """
        Function that returns the frequencies and powers 
        of a power spectrum performed using STINGRAY.
        In this case, the argument has to be ONLY a 
        STINGRAYTIMESERIES object ('ts'--> see code line 157) to be passed in the 
        analyze_segments function. 
        The time resolution dt and the segment size are 
        set by the global variables DT and SEGMENT SIZE.
        This spectrum is normalized with leahy normalization.
    """
    ps = Powerspectrum.from_time_array(ts.time, dt=DT, gti=ts.gti, segment_size=SEGMENT_SIZE,norm="leahy")
    # NW!: segment_size=SEGMENT_SIZE (=10 as set at the begginning of the code), therefore the "intermediate" power spectra to compute single segments power spectra is 10 s (this is important to understand line 421)
    return ps.freq, ps.power





def power_spectrum_NotNorm(ts):
     """
            Function that returns the frequencies and powers 
            of a power spectrum performed using STINGRAY.
            In this case, the argument has to be ONLY a 
            STINGRAYTIMESERIES object ('ts'--> see code line 157) to be passed in the 
            analyze_segments function. 
            The time resolution dt and the segment size are 
            set by the global variables DT and SEGMENT SIZE.
            This spectrum is NOT normalized.
        """
    ps = Powerspectrum.from_time_array(ts.time, dt=DT, gti=ts.gti, segment_size=SEGMENT_SIZE,norm="none")
    # NW!: segment_size=SEGMENT_SIZE (=10 as set at the begginning of the code), therefore the "intermediate" power spectra to compute single segments power spectra is 10 s (this is important to understand line 421)
    return ps.freq, ps.power





def Counts(ts):
    """
    calculate photon counts per time segment
    """
    return len(ts.time)





# analyze_segments from STINGRAY divides the observation in segments of duration set by SEG_SIZE (passed as a flag while launching the code) and for each of these segments uses the function power_spectrum_NotNorm, that computes the power spectrum, it gives start, stop and res arrays as output
# start: array with starting times (in MET!) for each segment
# stop: array with end times (in MET!) for each segment
# res: list containing two arrays r[0] and r[1] for each segment, respectively representing frequencies and powers of the segment power spectra 
start, stop, res = ts.analyze_segments(power_spectrum_NotNorm,SEG_SIZE)
# same as previous line, but counts the photon present in each segment
start_, stop_, counts = ts.analyze_segments(Counts,SEG_SIZE)

"""
    if LIGHTCURVE is True (i.e. the flag "-LC" is included while launching the code), I produce the lightcurve and 
    I plot it with the edges of the segments returned 
    by the analyze_segments function.
"""
# if LIGHTCURVE is True (i.e. the flag "-LC" is included while launching the code) extract all start and end times of segments and time-order them TIMES.show_LC(edges) is called to plot global lightcurve and identify segments time position with vertical bars
if LightCurve:
    edges=[]
    for i in range(len(start)):
        edges.append(start[i])
        edges.append(stop[i])
    edges = np.sort(edges)
    TIMES.show_LC(edges)





# noise components are fixed at the parameters extracted from global fit
PSQPO.A = par[6]
PSQPO.B = par[7]
PSQPO.C = par[8]/5. # NW!: why /5. ? Poisson noise is proportional to temporal acquisition window: since Global Power Spectrum uses a temporal segment size of 50 seconds to compute the power spectrum while for single segments the segment size is 10 seconds (see line 355), the C Poisson constant extracted in the global fit is to be divided by 5 





# guess for norm in the single segments power spectra using the double lorentian fitting function for nu = nu_qpo 
initial_guess_for_norm = (norm1-PSQPO.C)*np.pi*qpo_hwhm1     # the maximum is in x=nu_qpo and its value is norm/pi/hwhm + baseline
# guess parameters for signle segments power spectra
initial_guesses3 = [initial_guess_for_norm,qpo1,qpo_hwhm1]
norm_inf, norm_sup = 0, np.inf
# bounds for QPO frequency and hwhm in the single segments power spectra (\pm 3 hwhm)
nu_inf, nu_sup = qpo1-3*qpo_hwhm1, qpo1+3*qpo_hwhm1
h_inf, h_sup = qpo_hwhm1-3*qpo_hwhm1_err, qpo_hwhm1+3*qpo_hwhm1_err
bounds3 = ([norm_inf, nu_inf, h_inf], [norm_sup, nu_sup, h_sup])





# NW!: Another key passage!
# single segments fits using a 'for' cycle
# initialize cycle index and arrays
i=0
norme, max_norm_array = [], []
norme_test = []





# 'for' cycle to compute single power spectra: the cycle is tracked down with a progressing time bar (tqdm)
for r in tqdm(res):
    # passes frequency array for the given segment to PSQPO object
    PSQPO.Freq = r[0]
    # passes frequency array for the given segment to PSQPO object
    PSQPO.Power = r[1]
    # fit of Power Spectrum of the given segments with a double Lorentzian keeping all the background parameters fixed (with C corrected for segment size) and extract parameters array and covaraiance matrix
    par, cov = PSQPO.LorentzianExpFixed_Fit(initial_guess=initial_guesses3, bounds=bounds3)
    # NW! A key passage in the analysis
    # extract norm and max_norm
    # norm: norm of QPO from fit normalized with respect to global area 
    # max_norm: normalized maximum power of QPO in QPO window (\pm 1hwhm)
    norm, max_norm = PSQPO.MakeNormeAree_Norm(freqs=PSQPO.Freq,powers=PSQPO.Power,par=par,qpo=qpo1,hwhm=hwhm1,norme=norme,max_norm_array=max_norm_array,norm=area_qpo)
    # save plot of fitted Power Spectrum once per 5 segments
    if i%5==0:
        #segmenti temporali in cui salvo e guardo il PS che controllo online 
        plt.figure()
        plt.plot(r[0],r[1],marker='.',label=f'segment {i}')
        # verify if the fit succedeed: if isinstance(par, int) is True means the parameters array is a set of zero, which happens when fit fails (see LorentzianExpFixed_Fit in QPO classes); therefore we want if NOT isinstance(par, int), that means the fit worked
        if not isinstance(par, int):
            x = np.linspace(np.min(r[0]),np.max(r[0]),1000)
            plt.plot(x,PSQPO.LorentzianExpFixed(x, par[0], par[1], par[2]),marker='',linestyle='-', label=f'norm = {np.round(norm,2)} max = {np.round(max_norm,2)},\nqpo = {np.round(par[1],2)},\nhwhm = {np.round(par[2],2)}')
        plt.xscale('log')
        plt.grid('both')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('power')
        plt.legend()
        plt.xlim([np.min(r[0]),np.max(r[0])])
        figname_ps_seg = f'PS_seg{i}_alldus.png'
        figname_ps_seg_path = du_seg_folder+'/images/'+'seg_fits/'
        PSQPO.MakeDir(figname_ps_seg_path, info=False)
        plt.savefig(figname_ps_seg_path+figname_ps_seg, dpi=300)
        plt.close()  
    i = i+1

"""
    The array norme contains the norm/pi of the Lorentzian 
    of the fit of the qpo peak. 
    max_norm, on the other hand, contains the differences between 
    the max power in the range qpo +- hwhm from the fit and the 
    baseline given as parameter C of the global fit and 
    taken fixed for the fit of the segments.
    I plot of the norme and max_norm for each time segment.
"""
# create the array 'seg' containing integers corresponding to the segment number 
seg = np.arange(len(norme))
# Step plot comparing QPO strength per single segment
# 'norm from fit': Lorentzian area from segment fit
# 'power of the QPO': Peak power within QPO window MINUS Poisson noise C
fig, (ax1) = plt.subplots(1,1, sharex=True, figsize=(8, 6))
ax1.step(seg,norme,marker='.',where='mid',label='norm from fit')
ax1.step(seg,max_norm_array,marker='.',where='mid',label='power of the QPO')#label='max in range')
ax1.grid('both')
ax1.set_xlabel(f'segment number')
ax1.set_ylabel('norm')
ax1.legend()
# save plot
figname_norm = f'norms_alldu_seg{SEG_SIZE}s.png'
plt.savefig(du_seg_folder+'/images/'+figname_norm, dpi=300)
plt.close()





# break if norme and starts arrays don't match in lenght
if len(start)-len(max_norm_array)!=0:
    print('!!!!!!!!!\n   ERROR: len(starts) and len(stops) is different wrt the lenght of the max_norm_array: something wrong producing max_norm_array\n!!!!!!!!!')
    exit()

# show total livetime and total number of segments of the given observation
print(f'\n\n#####   livetime = {TIMES.Livetime()}\n')
print(f'#####   The number of segments produced using a segment size of {SEG_SIZE} seconds is {len(res)}\n')





# concatenate norm arrays with both methods (fit and max)
n_segments = len(res)
total_norms = np.concatenate((norme,max_norm_array))
n_segments_combined = len(total_norms)





# NW!: The following passage is key in the analysis
# build an array of threshold values ranging between the 2.5% and 97.5% quantiles of total_norms, stepping by 0.005 
threshold_array = PSQPO.MakeThresholds(array=total_norms,cl_quantile=0.05,steps=0.005)






#   I initialize the start and stop array of the segments where the max norm exceeds the threshold
for threshold in threshold_array:
    start_mask, stop_mask = [], []
    """
        In this loop I check in each segment if the condition 
        of max_norm (or norme) > threshold is satisfied and if it is, 
        I save in start_mask and stop_mask the start and stop 
        of the segment.
    """
    if method == 'max_norm':
        for i in range(len(max_norm_array)):
            if max_norm_array[i]>threshold:
                start_mask.append(start[i])
                stop_mask.append(stop[i])
    if method == 'par_norm':
        for i in range(len(norme)):
            if norme[i]>threshold:
                start_mask.append(start[i])
                stop_mask.append(stop[i])
    if method == 'hybrid':
        for i in range(len(max_norm_array)):
            if max_norm_array[i]>threshold or norme[i]>threshold:
                start_mask.append(start[i])
                stop_mask.append(stop[i])

    """
        Now I can finally save into the array good_times the
        events that belong to the good segments identified 
        by the start_mask and stop_mask.
    """
    good_times = []
    good_du_id = []
    for i in range(len(start_mask)):
        mask_time = np.where((event_times>start_mask[i])&(event_times<stop_mask[i]))[0]
        good_times.append(event_times[mask_time])
        good_du_id.append(du_id_final[mask_time])
    good_time_events = list(chain.from_iterable(good_times))
    good_du_id_index = list(chain.from_iterable(good_du_id))
    good_du_id_index = np.array(good_du_id_index)
    good_time_events = np.array(good_time_events)

    print(f'#####     The total number of events is {len(event_times)}.\n#####     The number of SELECTED events is {len(good_time_events)} in {len(start_mask)} selected time segments.\n\n')
    print(f'#####     len of good_time_events = {len(good_time_events)} and len of good_du_id_index = {len(good_du_id_index)}')

    mask_bool_selected = PSQPO.GetTimeMask(event_times, good_time_events, fast=True)
    print(len(mask_bool_selected))
    print(len(np.where(mask_bool_selected==True)[0]))
    mask_bool_not = np.logical_not(mask_bool_selected)

    mask_du1_bool_selected = PSQPO.GetSingleDUBooMask(du_id=1,event_time_array=event_times,
                                                     du_id_array=du_id_final,du_id_array_selected=good_du_id_index,
                                                     event_time_array_selected=good_time_events, fast=True)
    mask_du1_bool_not = np.logical_not(mask_du1_bool_selected)

    mask_du2_bool_selected = PSQPO.GetSingleDUBooMask(du_id=2,event_time_array=event_times,
                                                     du_id_array=du_id_final,du_id_array_selected=good_du_id_index,
                                                     event_time_array_selected=good_time_events, fast=True)
    mask_du2_bool_not = np.logical_not(mask_du2_bool_selected)

    mask_du3_bool_selected = PSQPO.GetSingleDUBooMask(du_id=3,event_time_array=event_times,
                                                     du_id_array=du_id_final,du_id_array_selected=good_du_id_index,
                                                     event_time_array_selected=good_time_events, fast=True)
    mask_du3_bool_not = np.logical_not(mask_du3_bool_selected)





    # is flag '-s' (i.e. 'save') is passed as the code is launched all the numpy arrays are saved
    if SAVE == True:
        print('\n[INFO] SAVING ARRAYS')
        np.save(du_seg_folder+f'/threshold_ALLdu_new.npy', good_time_events)
        np.save(du_seg_folder+f'/good_time_events_{method}_{SEG_SIZE}_{threshold}_ALLdu_new.npy', good_time_events)
        np.save(du_seg_folder+f'/mask_{method}_{SEG_SIZE}_{threshold}_ALLdu_new.npy',mask_bool_selected)
        np.save(du_seg_folder+f'/mask_not_{method}_{SEG_SIZE}_{threshold}_ALLdu_new.npy',mask_bool_not)
        np.save(du_seg_folder+f'/mask_{method}_{SEG_SIZE}_{threshold}_DU1_new.npy',mask_du1_bool_selected)
        np.save(du_seg_folder+f'/mask_not_{method}_{SEG_SIZE}_{threshold}_DU1_new.npy',mask_du1_bool_not)
        np.save(du_seg_folder+f'/mask_{method}_{SEG_SIZE}_{threshold}_DU2_new.npy',mask_du2_bool_selected)
        np.save(du_seg_folder+f'/mask_not_{method}_{SEG_SIZE}_{threshold}_DU2_new.npy',mask_du2_bool_not)
        np.save(du_seg_folder+f'/mask_{method}_{SEG_SIZE}_{threshold}_DU3_new.npy',mask_du3_bool_selected)
        np.save(du_seg_folder+f'/mask_not_{method}_{SEG_SIZE}_{threshold}_DU3_new.npy',mask_du3_bool_not)

    # plot the following sentence while saving so it's clear it's ok
if SAVE == True:
        print('\n[INFO] SAVING THRESHOLD ARRAY')
        np.save(du_seg_folder+f'/threshold_ALLdu_new.npy', threshold_array)