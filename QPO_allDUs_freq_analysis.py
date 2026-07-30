import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from QPO_classes import Times, PowerSpectrumQPO
import argparse
from stingray.base import StingrayTimeseries
from stingray import Powerspectrum
from tqdm import tqdm

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
# parser.add_argument('-du','--DU', type=int,  help='DU number ypu want to analyze', required=True)
parser.add_argument('-seg','--seg-size', type=int,  help='Duration of the time segments in seconds', required=True)
parser.add_argument('-LC','--LightCurve', action='store_true',  help='Do you want to see the Light Curve ?', required=False, default=False)
parser.add_argument('-show','--show-seg-PS', action='store_true',  help='Show the Power Spectra for all the time segments ?', default=False, required=False)
parser.add_argument('-method','--method-selection', type=str,  help='method to select the good time intervals',choices=['max_norm','par_norm','hybrid'] , default='max_norm', required=True)
parser.add_argument('-s','--save', action='store_true',  help='Do you want to save the masks ?', required=False, default=False)
parser.add_argument('-min','--min-thr', type=float, help='minimum threshold value', required=False)
parser.add_argument('-max','--max-thr', type=float, help='maximum threshold value', required=False)
parser.add_argument('-step','--step-thr', type=float, help='threshold step', required=False)

args = parser.parse_args()

# n = args.DU
SEG_SIZE = args.seg_size
LightCurve = args.LightCurve
show_all = args.show_seg_PS
method = args.method_selection
SAVE = args.save
# min_thr = args.min_thr
# max_thr = args.max_thr
# step_thr = args.step_thr
# threshold_array = np.linspace(min_thr, max_thr, int((max_thr-min_thr)/step_thr)+1)

#INSERIRE QUI PATH 
PATH = '/Users/stefanotugliani/Desktop/dati_ixpe/swift/event_files/02250901/'

event_times = np.load(PATH+'times_final.npy')
gti_starts = np.load(PATH+'gti_starts_merged.npy')
gti_stops = np.load(PATH+'gti_stops_merged.npy')
du_id_final = np.load(PATH+'du_id_final.npy')

DU = [PATH+'ixpe02250901_det1_evt2_v01_src.fits',
      PATH+'ixpe02250901_det2_evt2_v01_src.fits',
      PATH+'ixpe02250901_det3_evt2_v01_src.fits'
      ]

DT = 0.02
SEGMENT_SIZE = 10

#n_bin=segment_size/dt

# parameters for PCUBE and LC
ENERGY_BINNING = [2., 8.]
grayfilter_bool = True
acceptance_correction = False

TIMES = Times(energy_binning=ENERGY_BINNING, grayfilter_bool=grayfilter_bool, acceptance_correction=acceptance_correction, 
              files=DU, tbins=1000, du_id=1)

gti=[]
for i in range(len(gti_starts)):
    gti.append([gti_starts[i],gti_stops[i]])
# print(gti)

# I produce the StingrayTimeseries object ts
ts = StingrayTimeseries(time=event_times,gti=gti)


"""
    I set the initial guesses for the paramenters of the fit
    and the bounds, both for the single and double Lorentzian fit.
"""
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

PSQPO = PowerSpectrumQPO(stingray_ts=ts, dt=DT, segment_int=10, segment_tot=50)

du_folder = PATH+f'ALLDU_v'
du_seg_folder = PATH+f'ALLDU_v/seg{SEG_SIZE}s'
PSQPO.MakeDir(du_folder)
PSQPO.MakeDir(du_seg_folder)

par, cov = PSQPO.DoubleLorentzianExp_Fit(initial_guess=initial_guesses2, bounds=bounds2)

qpo_norm1 = float("%.3f"%par[0])
qpo_norm1_err = float("%.3f"%np.sqrt(cov[0][0]))
qpo_norm1_true = float("%.3f"%(par[0]/np.pi))
qpo1 = float("%.3f"%par[1])
qpo1_err = float("%.3f"%np.sqrt(cov[1][1]))
qpo_hwhm1 = float("%.3f"%par[2])
qpo_hwhm1_err = float("%.3f"%np.sqrt(cov[2][2]))
qpo_norm2 = float("%.3f"%par[3])
qpo_norm2_true = float("%.3f"%(par[3]/np.pi))
qpo2 = float("%.3f"%par[4])
qpo_hwhm2 = float("%.3f"%par[5])
exp_norm = float("%.3f"%par[6])
exp_tau = float("%.3f"%par[7])
exp_const = float("%.3f"%par[8])

area_qpo = par[0]


plt.figure('POWER SPECTRUM',figsize=(10,7))
plt.plot(PSQPO.Freq,PSQPO.Power,marker='.',linestyle='')
x = np.linspace(np.min(PSQPO.Freq),np.max(PSQPO.Freq),1000)
plt.plot(x,PSQPO.DoubleLorentzianExp(x,par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8]),marker='',linestyle='-',label='(Lorentzian qpo1)+\n(Lorentzian qpo2)+\n(exp)\n')
plt.plot(x,PSQPO.Lorentzian(x,par[0],par[1],par[2])+par[8],marker='',linestyle='--',label=f'Lorentzian qpo1\n qpo = {qpo1} Hz\n hwhm = {qpo_hwhm1} Hz\n norm = {qpo_norm1}')
plt.plot(x,PSQPO.Lorentzian(x,par[3],par[4],par[5])+par[8],marker='',linestyle='--',label=f'Lorentzian qpo2\n qpo = {qpo2} Hz\n hwhm = {qpo_hwhm2} Hz\n norm = {qpo_norm2}')
plt.plot(x,PSQPO.Exp(x,par[6],par[7],par[8]),marker='',linestyle='--',label=f'exp: ' + r'$A e^{-Bx}+C$' + f'\n A = {exp_norm} \n B = {exp_tau} \n C = {exp_const}')
# plt.yscale('log')
plt.xscale('log')
plt.grid('both')
plt.xlabel('frequency [Hz]')
plt.ylabel('power')
plt.xlim([np.min(PSQPO.Freq),np.max(PSQPO.Freq)])
plt.legend()
# plt.show()

model_power = PSQPO.DoubleLorentzianExp(PSQPO.Freq,par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8])
sigma_ps = PSQPO.Power_err

figname_ps = f'PS_ALLDU.png'
PSQPO.MakeDir(du_seg_folder+'/images')
plt.savefig(du_seg_folder+'/images/'+figname_ps, dpi=300)

breakpoint()

"""
    Now I can analyze the segments and fun the function 
    power_spectrum in each of them. Note that the function
    power_spectrum needs to have a StingrayTimeseries object
    as only argument.
"""

def power_spectrum(ts):
    """
        Function that returns the frequencies and powers 
        of a power spectrum performed using stingray.
        In this case, the argument has to be ONLY a 
        StingrayTimeseries object to be passed in the 
        analyze_segments function. 
        The time resolution dt and the segment size are 
        set by the global variables DT and SEGMENT SIZE.
    """
    ps = Powerspectrum.from_time_array(ts.time, dt=DT, gti=ts.gti, segment_size=SEGMENT_SIZE,norm="leahy")
    return ps.freq, ps.power

def power_spectrum_NotNorm(ts):
    """
        Function that returns the frequencies and powers 
        of a power spectrum performed using stingray.
        In this case, the argument has to be ONLY a 
        StingrayTimeseries object to be passed in the 
        analyze_segments function. 
        The time resolution dt and the segment size are 
        set by the global variables DT and SEGMENT SIZE.
    """
    ps = Powerspectrum.from_time_array(ts.time, dt=DT, gti=ts.gti, segment_size=SEGMENT_SIZE,norm="none")
    return ps.freq, ps.power

def Counts(ts):
    return len(ts.time)

# start, stop, res = ts.analyze_segments(power_spectrum,SEG_SIZE)
start, stop, res = ts.analyze_segments(power_spectrum_NotNorm,SEG_SIZE)

start_, stop_, counts = ts.analyze_segments(Counts,SEG_SIZE)

"""
    if LIGHTCURVE is True, I produce the lightcurve and 
    I plot it with the edges of the segments returned 
    by the analyze_segments function.
"""
if LightCurve:
    edges=[]
    for i in range(len(start)):
        edges.append(start[i])
        edges.append(stop[i])
    edges = np.sort(edges)
    TIMES.show_LC(edges)


### Lorentzian_exp(x, norm, center, hwhm, A, B, C)

PSQPO.A = par[6]
PSQPO.B = par[7]
PSQPO.C = par[8]/5. # /5. because frquencies = 0.5/DT*segment_tot = 0.5/0.02*50 = 1250, while for each segment we have 0.5/0.02*10 = 250, so fre_seg/freq_tot = 1/5

initial_guess_for_norm = (42000-PSQPO.C)*np.pi*qpo_hwhm1     # the maximum is in x=nu_qpo and its value is norm/pi/hwhm + baseline

initial_guesses3 = [initial_guess_for_norm,qpo1,qpo_hwhm1]
norm_inf, norm_sup = 0, np.inf
nu_inf, nu_sup = qpo1-3*qpo_hwhm1, qpo1+3*qpo_hwhm1
h_inf, h_sup = qpo_hwhm1-3*qpo_hwhm1_err, qpo_hwhm1+3*qpo_hwhm1_err
bounds3 = ([norm_inf, nu_inf, h_inf], [norm_sup, nu_sup, h_sup])

i=0
norme, max_norm_array = [], []
norme_test = []

for r in tqdm(res):
    PSQPO.Freq = r[0]
    PSQPO.Power = r[1]
    par, cov = PSQPO.LorentzianExpFixed_Fit(initial_guess=initial_guesses3, bounds=bounds3)
    # norm, max_norm = PSQPO.MakeNorme(freqs=PSQPO.Freq,powers=PSQPO.Power,par=par,qpo=qpo1,hwhm=hwhm1,norme=norme,max_norm_array=max_norm_array)
    norm, max_norm = PSQPO.MakeNormeAree_Norm(freqs=PSQPO.Freq,powers=PSQPO.Power,par=par,qpo=qpo1,hwhm=hwhm1,norme=norme,max_norm_array=max_norm_array,norm=area_qpo)
    if i%5==0:
        #segmenti temporali in cui salvo e guardo il PS che controllo online 
        plt.figure()
        plt.plot(r[0],r[1],marker='.',label=f'segment {i}')
        if not isinstance(par, int):
            x = np.linspace(np.min(r[0]),np.max(r[0]),1000)
            # plt.plot(x,PSQPO.LorentzianExpFixed(x, par[0], par[1], par[2]),marker='',linestyle='-')#######,label=f'norm = {par[0]/np.pi} max = {max_norm-C_fixed},\nqpo = {par[1]},\nhwhm = {par[2]}')
            plt.plot(x,PSQPO.LorentzianExpFixed(x, par[0], par[1], par[2]),marker='',linestyle='-',
                     label=f'norm = {np.round(norm,2)} max = {np.round(max_norm,2)},\nqpo = {np.round(par[1],2)},\nhwhm = {np.round(par[2],2)}')
        # plt.yscale('log')
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
seg = np.arange(len(norme))
# fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(8, 6))
fig, (ax1) = plt.subplots(1,1, sharex=True, figsize=(8, 6))
#ax1.step(seg,norme,marker='.',where='mid',label='norm from fit')
ax1.step(seg,norme,marker='.',where='mid',label='norm from fit')
ax1.step(seg,max_norm_array,marker='.',where='mid',label='power of the QPO')#label='max in range')
ax1.grid('both')
ax1.set_xlabel(f'segment number')
ax1.set_ylabel('norm')
ax1.legend()
# plt.show()

# breakpoint()

figname_norm = f'norms_alldu_seg{SEG_SIZE}s.png'
plt.savefig(du_seg_folder+'/images/'+figname_norm, dpi=300)

# ax2.step(seg,max_norm_array,marker='.',where='mid')
# ax2.grid('both')
# ax2.set_xlabel(f'segment number')
# ax2.set_ylabel('max from selection')
# breakpoint()
#   check if len(starts) and len(stops) are equal to len(max_norm_array): 
#   this is a way to check if something went wrong producing max_norm_array.
if len(start)-len(max_norm_array)!=0:
    print('!!!!!!!!!\n   ERROR: len(starts) and len(stops) is different wrt the lenght of the max_norm_array: something wrong producing max_norm_array\n!!!!!!!!!')
    exit()


print(f'\n\n#####   livetime = {TIMES.Livetime()}\n')
print(f'#####   The number of segments produced using a segment size of {SEG_SIZE} seconds is {len(res)}\n')

n_segments = len(res)
total_norms = np.concatenate((norme,max_norm_array))
n_segments_combined = len(total_norms)

threshold_array = PSQPO.MakeThresholds(array=total_norms,cl_quantile=0.05,steps=0.005)

#   I initialize the start and stop array of the segments
#   where the max norm exceeds the threshold

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
# plt.show()
if SAVE == True:
        print('\n[INFO] SAVING THRESHOLD ARRAY')
        np.save(du_seg_folder+f'/threshold_ALLdu_new.npy', threshold_array)
