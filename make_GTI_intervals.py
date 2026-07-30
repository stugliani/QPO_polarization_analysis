"""
import useful libraries

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pf
import os
from scipy.optimize import curve_fit
import math
from tqdm import tqdm

import argparse
from ixpeobssim.binning.polarization import xBinnedPolarizationCube, xBinnedCountSpectrum
from ixpeobssim.binning.misc import xBinnedLightCurve
import ixpeobssim.core.pipeline as pipeline
from ixpeobssim.evt.event import xEventFile
"""
to get DU number from command line

"""
# formatter = argparse.ArgumentDefaultsHelpFormatter
# parser = argparse.ArgumentParser(formatter_class=formatter)
# parser.add_argument('-du','--DU', type=int,  help='DU number ypu want to analyze', required=True)

# args = parser.parse_args()

# n = args.DU


ENERGY_BINNING = [2., 8.] #energy interval in keV
grayfilter_bool = True # IXPE "gray filter" ACTIVE in binning for this source (swift is very luminous!)
acceptance_correction = False # acceptance correction disabled


# insert YOUR PATH HERE (where the source files are stored)
PATH = '/Users/stefanotugliani/Desktop/dati_ixpe/swift/event_files/02250901/'
# and each DU file
DU = [PATH+'ixpe02250901_det1_evt2_v01_src.fits',
      PATH+'ixpe02250901_det2_evt2_v01_src.fits',
      PATH+'ixpe02250901_det3_evt2_v01_src.fits'
      ]

"""
NW! THIS PARAMETER IS KEY IN THE ANALYSIS

DT: to choose temporal resolution
SEGMENT_SIZE: size of temporal segments 

"""

DT = 0.02
SEGMENT_SIZE = 10

def readsimfitsfile(file_path):
    """
        Function reads fits file
        it returns the events and the GTI
    """
    data_f = pf.open(file_path)
    data_f.info()
    events = data_f['EVENTS'].data
    GTI = data_f['GTI'].data
    
    return events, GTI



def get_TIME(FILE_LIST):
    
    """
    computes global tstart (min time) and tstop (max time) among all the source files
   
    """
    tstart = []
    tstop = []
    for f in FILE_LIST:
        event_file = xEventFile(f)
        tstart.append(event_file.start_met())
        tstop.append(event_file.stop_met())
    
    return np.min(tstart), np.max(tstop)



def LC(FILE_LIST,tbins, ENERGY_BINNING, TSTART, TSTOP):
    """
        Function that creates light curve _LC fits 
        files and returns an xBinnedLightCurve object
    """
    # create time-binned files from FITS file 
    LC_LIST = pipeline.xpbin(*FILE_LIST, algorithm='LC',tbins=tbins, tmin=TSTART, tmax=TSTOP, 
                   ebinning=ENERGY_BINNING, overwrite=True, 
                   grayfilter=grayfilter_bool,acceptcorr=acceptance_correction)
    # time-binned files list stored as xBinnedLightCurve object 
    lightcurve = xBinnedLightCurve.from_file_list(LC_LIST)
    # os.remove(PATH+'ixpe02250901_det1_evt2_v01_src_lc.fits')
    # os.remove(PATH+'ixpe02250901_det2_evt2_v01_src_lc.fits')
    # os.remove(PATH+'ixpe02250901_det3_evt2_v01_src_lc.fits')

    return lightcurve

def get_LC(lightcurve, tbins):
    """
        Function that returns the LC rate, error and time from an
        xBinnedLightCurve object (lightcurve) and the LC bins (tbins)
    """
    light_curve, light_curve_error, time_LC = [], [], []
    for i in range(tbins-1):
        light_curve.append(lightcurve.COUNTS[i]/lightcurve.EXPOSURE[i])
        light_curve_error.append(lightcurve.ERROR[i]/lightcurve.EXPOSURE[i])
        time_LC.append(lightcurve.TIME[i])

    return light_curve, light_curve_error, time_LC

def adjust_lc_rate_from_array(light_curve, light_curve_error, time_LC):
    rate, rate_err, t = [], [], []
    for i in range(len(light_curve)):
        if not math.isnan(light_curve[i]):
            rate.append(light_curve[i])
            rate_err.append(light_curve_error[i])
            t.append(time_LC[i])
    return np.array(rate), np.array(rate_err), np.array(t)

def show_LC(light_curve, light_curve_error, time_LC, plot_intervals=None, gti=None,ax=None):#, plot_rate_value):
    """
        Function that plots the Light Curve from an
        the LC rate, error and time
    """
    if ax is not None:
        ax = ax
    else:
        fig_LC, ax = plt.subplots()

    light_curve, light_curve_error, time_LC = adjust_lc_rate_from_array(light_curve, light_curve_error, time_LC)

    ax.errorbar(time_LC,light_curve,yerr=light_curve_error,xerr=0,linestyle='',marker='.')

    if plot_intervals is not None:
        for T in plot_intervals:
            ax.axvline(x=T, linestyle='--')
    
    if gti is not None:
        starts = gti[0]
        stops = gti[1]
        for i in range(len(starts)):
            ax.fill_betweenx(y=np.linspace(0,np.max(light_curve)+3*np.max(light_curve_error),1000),x1=starts[i], x2=stops[i], alpha=0.25,color='cyan')
        for ss in starts:
            ax.axvline(x=ss, linestyle='--',color='green')
        for sss in stops:
            ax.axvline(x=sss, linestyle='--',color='red')

    ax.set_ylabel('Rate [Hz]')
    ax.set_xlabel('MET [s]')
    ax.grid(True)
    return ax


def time_analysis_single_DU(DU_int,ax=None):
    n = DU_int-1
    # I obtain the event and GTI from DU number n
    events, GTI = readsimfitsfile(DU[n])
    event_times = events['TIME'] # I get the event times
    # I build the GTI array from the GTI of the observation 
    gti=[]
    gti_starts, gti_stops = [], []
    for i in range(len(GTI)):
        gti.append([GTI[i][0],GTI[i][1]])
        gti_starts.append(GTI[i][0])
        gti_stops.append(GTI[i][1])

    TSTART, TSTOP = get_TIME([DU[n]])
    light_curve, light_curve_error, time_LC = get_LC(LC([DU[n]],1000, ENERGY_BINNING, TSTART, TSTOP),1000)
    show_LC(light_curve, light_curve_error, time_LC,gti=[gti_starts,gti_stops],ax=ax)#, plot_intervals, plot_rate_value)
    return np.array(event_times), np.array(gti), np.array(gti_starts), np.array(gti_stops)

def times_corrected(times, du_id, gti_starts, gti_stops):
    T = []
    DU_ = []
    for t in range(len(times)):
        for i in range(len(gti_starts)):
            if times[t]>=gti_starts[i] and times[t]<=gti_stops[i]:
                T.append(times[t])
                DU_.append(du_id[t])
    return np.array(T), np.array(DU_)




# times_1 = times_corrected(times=event_times_1, gti_starts=gti_starts_1, gti_stops=gti_stops_1)
# times_2 = times_corrected(times=event_times_2, gti_starts=gti_starts_2, gti_stops=gti_stops_2)
# times_3 = times_corrected(times=event_times_3, gti_starts=gti_starts_3, gti_stops=gti_stops_3)

# gti_starts_12 = np.sort(np.concatenate((gti_starts_1,gti_starts_2)))
# gti_stops_12 = np.sort(np.concatenate((gti_stops_1,gti_stops_2)))

def create_merged_GTI_couple(gti_starts_1,gti_starts_2,gti_stops_1,gti_stops_2):

    N_GTI_1 = len(gti_starts_1)
    N_GTI_2 = len(gti_starts_2)
    start_12 = []
    stop_12 = []
    for i in range(N_GTI_1):
        for j in range(N_GTI_2):
            if gti_starts_2[j]>gti_starts_1[i] and  gti_starts_2[j]<gti_stops_1[i] and gti_stops_2[j]>gti_starts_1[i] and  gti_stops_2[j]<gti_stops_1[i]:
                start_12.append(gti_starts_2[j])
                stop_12.append(gti_stops_2[j])

            if gti_starts_2[j]<gti_starts_1[i] and gti_stops_2[j]>gti_stops_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_1[i])

            if gti_starts_2[j]>gti_starts_1[i] and gti_starts_2[j]<gti_stops_1[i] and gti_stops_2[j]>gti_stops_1[i]:
                start_12.append(gti_starts_2[j])
                stop_12.append(gti_stops_1[i])

            if gti_stops_2[j]>gti_starts_1[i] and gti_stops_2[j]<gti_stops_1[i] and gti_starts_2[j]<gti_starts_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_2[j])

            if gti_starts_1[i]==gti_starts_2[j] and gti_stops_2[j]>gti_starts_1[i] and gti_stops_2[j]<gti_stops_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_2[j])

            if gti_starts_1[i]==gti_starts_2[j] and gti_stops_2[j]>gti_stops_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_1[i])

            if gti_starts_1[i]==gti_starts_2[j] and gti_stops_2[j]==gti_stops_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_1[i])

            if gti_stops_1[i]==gti_stops_2[j] and gti_starts_2[j]>gti_starts_1[i] and gti_starts_2[j]<gti_stops_1[i]:
                start_12.append(gti_starts_2[j])
                stop_12.append(gti_stops_1[i])

            if gti_stops_1[i]==gti_stops_2[j] and gti_starts_2[j]<gti_starts_1[i]:
                start_12.append(gti_starts_1[i])
                stop_12.append(gti_stops_1[i])

            if gti_starts_2[j]<gti_starts_1[i] and gti_stops_2[j]<gti_starts_1[i]:
                continue

            if gti_starts_2[j]>gti_stops_1[i] and gti_stops_2[j]>gti_stops_1[i]:
                continue
    
    start_12 = np.array(list(dict.fromkeys(start_12)))
    stop_12 = np.array(list(dict.fromkeys(stop_12)))

    return start_12, stop_12

def get_mask(event_times, sel_times_array):

    mask = np.zeros(len(event_times)).astype(bool)  # I create a mask array long as event the main array filled with False
    for i, t in enumerate(tqdm(sel_times_array)):
        iii = np.where(event_times == t)[0]         # iii is the array of the indices of the main array element equal to the t element of the selected array
        if len(iii) > 0:
            mask[iii] = bool(1)                     # if there are elements in the main array equal to the element t of the selected array, 
        else:                                       # I put the corresponding bool mask element equal to True
            abs_differences = abs(event_times - t)
            best_iii = np.argmin(abs_differences)
            mask[best_iii] = bool(0)
    return mask

fig_LC, (ax1, ax2, ax3) = plt.subplots(3,1,sharex = True)
event_times_1, gti_1, gti_starts_1, gti_stops_1 = time_analysis_single_DU(DU_int=1,ax=ax1)
event_times_2, gti_2, gti_starts_2, gti_stops_2 = time_analysis_single_DU(DU_int=2,ax=ax2)
event_times_3, gti_3, gti_starts_3, gti_stops_3 = time_analysis_single_DU(DU_int=3,ax=ax3)

start_12, stop_12 = create_merged_GTI_couple(gti_starts_1,gti_starts_2,gti_stops_1,gti_stops_2)
start_gti, stop_gti = create_merged_GTI_couple(gti_starts_1=start_12,gti_starts_2=gti_starts_3,gti_stops_1=stop_12,gti_stops_2=gti_stops_3)

du1_id = np.array([1]*len(event_times_1))
du2_id = np.array([2]*len(event_times_2))
du3_id = np.array([3]*len(event_times_3))

du_id = np.concatenate((du1_id,du2_id,du3_id))
event_times = np.concatenate((event_times_1,event_times_2,event_times_3))

combined = np.array(sorted(zip(event_times, du_id), key=lambda x: x[0]))

times = combined[:,0]
du_id_array = combined[:,1]


times_final, du_id_final = times_corrected(times, du_id_array, start_gti, stop_gti)

mask_du_1 = np.where(du_id_final==1)[0]
event_times_1_final = times_final[mask_du_1]
mask_bool_1 = get_mask(event_times=event_times_1, sel_times_array=event_times_1_final)

mask_du_2 = np.where(du_id_final==2)[0]
event_times_2_final = times_final[mask_du_2]
mask_bool_2 = get_mask(event_times=event_times_2, sel_times_array=event_times_2_final)

mask_du_3 = np.where(du_id_final==3)[0]
event_times_3_final = times_final[mask_du_3]
mask_bool_3 = get_mask(event_times=event_times_3, sel_times_array=event_times_3_final)


# times_final_1 = times_corrected(event_times_1,start_gti,stop_gti)
# times_final_2 = times_corrected(event_times_2,start_gti,stop_gti)
# times_final_3 = times_corrected(event_times_3,start_gti,stop_gti)

TSTART, TSTOP = get_TIME(DU)
light_curve, light_curve_error, time_LC = get_LC(LC(DU,1000, ENERGY_BINNING, TSTART, TSTOP),1000)
fig_LC_, ax = plt.subplots(1,1,sharex = True)
show_LC(light_curve, light_curve_error, time_LC,gti=[start_gti,stop_gti],ax=ax)

print(f'NUMBER OF EVENTS: {len(times)}')
print(f'NUMBER OF EVENTS AFTER MERGED GTIs: {len(times_final)}')



##### UNCOMMENT IF YOU WANT TO TEST THE MERGING OF THE GTIs
'''
fig_, (ax1, ax2) = plt.subplots(2,1,sharex = True)
for i in range(len(gti_starts_1)):        
    ax1.axvline(x=gti_starts_1[i],color='green')
    ax1.axvline(x=gti_stops_1[i],color='red')
    ax1.fill_betweenx(y=np.linspace(0,100,1000),x1=gti_starts_1[i], x2=gti_stops_1[i], alpha=0.25,color='gray')
for i in range(len(gti_starts_2)):        
    ax1.axvline(x=gti_starts_2[i],color='blue')
    ax1.axvline(x=gti_stops_2[i],color='orange')
    ax1.fill_betweenx(y=np.linspace(0,100,1000),x1=gti_starts_2[i], x2=gti_stops_2[i], alpha=0.25,color='gray')
for i in range(len(start_12)):        
    ax2.axvline(x=start_12[i],color='purple')
    ax2.axvline(x=stop_12[i],color='pink')
    ax2.fill_betweenx(y=np.linspace(0,100,1000),x1=start_12[i], x2=stop_12[i], alpha=0.25,color='gray')
'''

np.save(PATH+'times_final.npy',times_final)
np.save(PATH+'du_id_final.npy',du_id_final)
np.save(PATH+'gti_starts_merged.npy',start_gti)
np.save(PATH+'gti_stops_merged.npy',stop_gti)
np.save(PATH+'mask_1_allDU.npy',mask_bool_1)
np.save(PATH+'mask_2_allDU.npy',mask_bool_2)
np.save(PATH+'mask_3_allDU.npy',mask_bool_3)


plt.show()

# ef = xEventFile(DU[n])
# livetime = ef.livetime()
# print(f'\n\n#####     livetime = {livetime}\n')
# print(f'#####     The number of segments produced using a segment size of {SIZE} seconds is {len(res)}\n')

