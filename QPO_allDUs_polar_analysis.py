import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pf
import os
import math
import pandas

# the following functions are imported from 
# other python scripts 
#sys.path.insert(0,'/Users/stefanotugliani/Desktop/analisi/analysis_functions')
sys.path.insert(0,'/Users/gabri/OneDrive/Fisica/TESI/QPO_python/analisi')
from polarization_plots import *
from general import round_to_significant_figures


from ixpeobssim.binning.polarization import xBinnedPolarizationCube, xBinnedCountSpectrum
from ixpeobssim.binning.misc import xBinnedLightCurve
import ixpeobssim.core.pipeline as pipeline
from ixpeobssim.evt.event import xEventFile

import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-seg','--seg_size', type=str,  help='segment size', required=True)
parser.add_argument('-m','--method', type=str,  help='method for selection', required=True)
parser.add_argument('-s','--save', action='store_true',  help='Do you want to save the figures ?', required=False, default=False)
parser.add_argument('-sf','--save_out_file', action='store_true',  help='Do you want to save the output file ?', required=False, default=False)
parser.add_argument('-st','--save_table', action='store_true',  help='Do you want to save the output csv table ?', required=False, default=False)

args = parser.parse_args()

seg_size = args.seg_size # segment size in seconds
method = args.method # the method used to select the events
save = args.save # to save the figures 
save_out_file = args.save_out_file # to save the polarization results on an output file
save_table = args.save_table # to save the results and useful data on an output csv table

PATH = '/Users/stefanotugliani/Desktop/dati_ixpe/swift/event_files/02250901/'

DU = [f'{PATH}ixpe02250901_det1_evt2_v01_src.fits',
      f'{PATH}ixpe02250901_det2_evt2_v01_src.fits',
      f'{PATH}ixpe02250901_det3_evt2_v01_src.fits'
      ]
path_thr = f'{PATH}ALLDU_v/seg{seg_size}s/threshold_ALLdu_new.npy'
THR = np.load(path_thr)

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

def run_xpselect(file_path, custom_mask, suffix):
    # breakpoint()
    pipeline.xpselect(file_path, mask=custom_mask, suffix=suffix, overwrite=True)

def run_pcubes(file_list):
    pol_pcube_list = pipeline.xpbin(*file_list, algorithm='PCUBE', emin=2.,
                                    emax=8., ebins=1, overwrite=True, acceptcorr=False,
                                    irfname='ixpe:obssim:v12', grayfilter=True)
    #pipeline.xpbinview(*pol_pcube_list)
    pcube = xBinnedPolarizationCube.from_file_list(pol_pcube_list)
    qn, un, qun_err = pcube.QN[0], pcube.UN[0], pcube.QN_ERR[0]
    return qn, un, qun_err, pol_pcube_list

ENERGY_BINNING = [2., 8.]
grayfilter_bool = True
acceptance_correction = False
emin = np.min(ENERGY_BINNING)
emax = np.max(ENERGY_BINNING)



# output file where the polarization results can be saved
outfile = f'/Users/stefanotugliani/Desktop/dati_ixpe/swift/event_files/02250901/polarization_ALLDU_new_{seg_size}s_{method}_v.txt'
file = open(outfile, 'w')

QN, QN_ERR, UN, UN_ERR, PD, PD_ERR, PA, PA_ERR, MDP = [], [], [], [], [], [], [], [], []
THRESHOLDS = []
EVT = []

mask_det1_bool = np.load(PATH+'mask_1_allDU.npy')
mask_det2_bool = np.load(PATH+'mask_2_allDU.npy')
mask_det3_bool = np.load(PATH+'mask_3_allDU.npy')

mask_det1_bool_path = PATH+'mask_1_allDU.npy'
mask_det2_bool_path = PATH+'mask_2_allDU.npy'
mask_det3_bool_path = PATH+'mask_3_allDU.npy'

run_xpselect(DU[0], custom_mask=mask_det1_bool_path, suffix='after_merging')
run_xpselect(DU[1], custom_mask=mask_det2_bool_path, suffix='after_merging')
run_xpselect(DU[2], custom_mask=mask_det3_bool_path, suffix='after_merging')

DU_new = [f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging.fits',
          f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging.fits',
          f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging.fits'
          ]

events_1, GTI_1 = readsimfitsfile(DU_new[0])
event_times_1 = events_1['TIME']
events_2, GTI_2 = readsimfitsfile(DU_new[1])
event_times_2 = events_2['TIME']
events_3, GTI_3 = readsimfitsfile(DU_new[2])
event_times_3 = events_3['TIME']

"""
    For each threshold used to select the events,
    I run xpselect with the correspondant boolean mask
    and so I can finally calculate the polarization 
    of the selected and unselected data set
"""
for thr in THR:

    mask_sel_path_1 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_{method}_{seg_size}_{thr}_DU1_new.npy'
    mask_not_path_1 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_not_{method}_{seg_size}_{thr}_DU1_new.npy'

    mask_sel_1 = np.load(mask_sel_path_1)
    mask_not_1 = np.load(mask_not_path_1)

    selected_events_1 = len(np.where(mask_sel_1==True)[0])
    not_selected_events_1 = len(np.where(mask_sel_1==False)[0])

    mask_sel_path_2 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_{method}_{seg_size}_{thr}_DU2_new.npy'
    mask_not_path_2 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_not_{method}_{seg_size}_{thr}_DU2_new.npy'

    mask_sel_2 = np.load(mask_sel_path_2)
    mask_not_2 = np.load(mask_not_path_2)

    selected_events_2 = len(np.where(mask_sel_2==True)[0])
    not_selected_events_2 = len(np.where(mask_sel_2==False)[0])

    mask_sel_path_3 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_{method}_{seg_size}_{thr}_DU3_new.npy'
    mask_not_path_3 = f'{PATH}ALLDU_v/seg{seg_size}s/mask_not_{method}_{seg_size}_{thr}_DU3_new.npy'

    mask_sel_3 = np.load(mask_sel_path_3)
    mask_not_3 = np.load(mask_not_path_3)

    selected_events_3 = len(np.where(mask_sel_3==True)[0])
    not_selected_events_3 = len(np.where(mask_sel_3==False)[0])

    if len(np.where(mask_sel_1==True)[0]) != len(np.where(mask_not_1==False)[0]):
        print('\n!!!!!!!!!!!!!!!!!!! ERROR: number of True in selected and False in not selected is different !!!!!!!!!!!!!')
        exit()

    SUFFIX = f'{seg_size}_{thr}'

    selected_events = selected_events_1+selected_events_2+selected_events_3
    not_selected_events = not_selected_events_1+not_selected_events_2+not_selected_events_3    
    run_xpselect(DU_new[0], custom_mask=mask_sel_path_1, suffix=SUFFIX)
    run_xpselect(DU_new[0], custom_mask=mask_not_path_1, suffix=f'{SUFFIX}_not')
    run_xpselect(DU_new[1], custom_mask=mask_sel_path_2, suffix=SUFFIX)
    run_xpselect(DU_new[1], custom_mask=mask_not_path_2, suffix=f'{SUFFIX}_not')
    run_xpselect(DU_new[2], custom_mask=mask_sel_path_3, suffix=SUFFIX)
    run_xpselect(DU_new[2], custom_mask=mask_not_path_3, suffix=f'{SUFFIX}_not')
    
    """
        the following paths are pointed to the position
        of the just created selected fits file
    """
    file_sel = [f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging_{SUFFIX}.fits',
        f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging_{SUFFIX}.fits',
        f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging_{SUFFIX}.fits'
        ]

    file_not = [f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging_{SUFFIX}_not.fits',
        f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging_{SUFFIX}_not.fits',
        f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging_{SUFFIX}_not.fits'
        ]

    """
        I run xpbin to calculate the pcubes
        of the selected events
    """
    pol_pcube_list = pipeline.xpbin(*file_sel, algorithm='PCUBE', ebinning=ENERGY_BINNING, ebinalg='LIST', 
                                    overwrite=True, acceptcorr=False, irfname='ixpe:obssim:v12', grayfilter=True)
    
    pcube = xBinnedPolarizationCube.from_file_list(pol_pcube_list)
    qn, un, qn_err, un_err, pd, pd_err, pa, pa_err, mdp = pcube.QN, pcube.UN, pcube.QN_ERR, pcube.UN_ERR, pcube.PD, pcube.PD_ERR, pcube.PA, pcube.PA_ERR, pcube.MDP_99
    """
        I run xpbin to calculate the pcubes
        of the not selected events
    """
    pol_pcube_list_n = pipeline.xpbin(*file_not, algorithm='PCUBE', ebinning=ENERGY_BINNING, ebinalg='LIST', 
                                    overwrite=True, acceptcorr=False, irfname='ixpe:obssim:v12', grayfilter=True)
    
    pcube_n = xBinnedPolarizationCube.from_file_list(pol_pcube_list_n)
    qn_n, un_n, qn_err_n, un_err_n = pcube_n.QN, pcube_n.UN, pcube_n.QN_ERR, pcube_n.UN_ERR
    pd_n, pd_err_n, pa_n, pa_err_n, mdp_n = pcube_n.PD, pcube_n.PD_ERR, pcube_n.PA, pcube_n.PA_ERR, pcube_n.MDP_99

    """
        I can plot the polar plot with
        the contour plots of the polarization angle
        and degree
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 7), tight_layout=True)
    polarization_contour(pol_pcube_list,ax=ax,text=True,global_color=['C0'],aspect='half top',levels=[0.68,0.95])
    polarization_contour(pol_pcube_list_n,ax=ax,text=True,global_color=['C1'],aspect='half top',levels=[0.68,0.95])
    # ax.set_title(f'{thr} {seg_size}s {method}')
    ax.set_thetamin(-30)
    ax.set_thetamax(30)
    ax.set_rmax(7)
    ax.text(np.radians(0), 8.25, f'{thr} {seg_size}s {method}',ha='center',va='center')
    ax.text(np.radians(0), 7.6, 'Polarization angle [°]',ha='center',va='center')
    ax.text(np.radians(60), 3, 'Selected events',color='C0',ha='center',va='center')
    ax.text(np.radians(60), 2.5, 'Not selected events',color='C1',ha='center',va='center')
    ax.text(np.radians(-45), 2, 'Polarization degree [%]',rotation=60)
    ax.text(np.radians(0), 7.75, 'N')
    ax.text(np.radians(-25), 7.75, 'W')
    ax.text(np.radians(25), 7.75, 'E')


    plt.savefig(f'{PATH}ALLDU_v/seg{seg_size}s/images/polar_{method}_{SUFFIX}.png')
    """
        Now I create these 'titles' that will be useful
        to write the output file and the output on the 
        terminal
    """
    selected_title = create_title(ENERGY_BINNING,'selected')
    not_selected_title = create_title(ENERGY_BINNING,'not selected')
    """
        I can plot the pcube plot with
        the contour plots of the Stokes
        parameters QN and UN
    """
    # polarization_plot(QN, QN_ERR, UN, UN_ERR, PD, PD_ERR, PA, PA_ERR, MDP,title)
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    pcube_contour_plot(QN=qn, QN_ERR=qn_err, UN=un, UN_ERR=un_err, PD=pd, PD_ERR=pd_err, PA=pa, PA_ERR=pa_err, MDP_99=mdp, 
                       title=selected_title, ENERGY_BINNING=ENERGY_BINNING, ax=ax1, grid=False, global_color=['C0'])
    pcube_contour_plot(QN=qn_n, QN_ERR=qn_err_n, UN=un_n, UN_ERR=un_err_n, PD=pd_n, PD_ERR=pd_err_n, PA=pa_n, PA_ERR=pa_err_n, MDP_99=mdp_n, 
                       title=not_selected_title, ENERGY_BINNING=ENERGY_BINNING, ax=ax1,grid=True,global_color=['C1'])
    ax1.set_title(f'{thr} {seg_size}s {method}')
    
    ax1.set_xlim([-0.1,0.1])
    ax1.set_ylim([-0.1,0.1])
    ax1.set_title(f'{thr} {seg_size}s {method} pcube')

    plt.savefig(f'{PATH}ALLDU_v/seg{seg_size}s/images/pcube_{method}_{SUFFIX}.png')

    """
        I create the QN, QN_ERR, UN, UN_ERR,
        PD, PD_ERR, PA, PA_ERR, MDP, thresholds
        and number of events array in a very 
        naive way, but necessary for me to 
        create the pandas data frame table that 
        will be saved
    """
    QN.append(qn[0])
    QN.append(qn_n[0])
    QN_ERR.append(qn_err[0])
    QN_ERR.append(qn_err_n[0])
    UN.append(un[0])
    UN.append(un_n[0])
    UN_ERR.append(un_err[0])
    UN_ERR.append(un_err_n[0])
    PD.append(pd[0])
    PD.append(pd_n[0])
    PD_ERR.append(pd_err[0])
    PD_ERR.append(pd_err_n[0])
    PA.append(pa[0])
    PA.append(pa_n[0])
    PA_ERR.append(pa_err[0])
    PA_ERR.append(pa_err_n[0])
    MDP.append(mdp[0])
    MDP.append(mdp_n[0])
    THRESHOLDS.append(thr)
    THRESHOLDS.append(thr)
    EVT.append(selected_events)
    EVT.append(not_selected_events)
    
    """
        with maybe too many code lines 
        I write the results on the output 
        file and on the terminal
    """
    print(f'')
    print(f'segment_size = {seg_size}s\nthreshold = {thr}')
    print(f'selected events: {selected_events}\nnon-selected events: {not_selected_events}\n')
    print(f'method: {method}\n')
    if save_out_file==True:
        file.write(f'segment_size = {seg_size}s\nthreshold = {thr}\n')
        file.write(f'selected events: {selected_events}\nnon-selected events: {not_selected_events}\n')
        file.write(f'method: {method}\n')
    title=['selected', 'not selected']
    for i in range(len(selected_title)):
        print(f'{selected_title[i]} events\n')
        print(f'QN = {round_to_significant_figures(qn[i]*100.,1)} +- {round_to_significant_figures(qn_err[i]*100.,0)} %')
        print(f'UN = {round_to_significant_figures(un[i]*100.,1)} +- {round_to_significant_figures(un_err[i]*100.,0)} %')
        print(f'PD = {round_to_significant_figures(pd[i]*100.,1)} +- {round_to_significant_figures(pd_err[i]*100.,0)} %')
        print(f'PA = {round_to_significant_figures(pa[i]*1.,2)} +- {np.round(pa_err[i]*1.,3)} °')
        print(f'MDP = {np.round(mdp[i]*100.,3)} %')
        print(f'{not_selected_title[i]} events\n')
        print(f'QN = {round_to_significant_figures(qn_n[i]*100.,1)} +- {round_to_significant_figures(qn_err_n[i]*100.,0)} %')
        print(f'UN = {round_to_significant_figures(un_n[i]*100.,1)} +- {round_to_significant_figures(un_err_n[i]*100.,0)} %')
        print(f'PD = {round_to_significant_figures(pd_n[i]*100.,1)} +- {round_to_significant_figures(pd_err_n[i]*100.,0)} %')
        print(f'PA = {round_to_significant_figures(pa_n[i]*1.,2)} +- {np.round(pa_err_n[i]*1.,3)} °')
        print(f'MDP = {np.round(mdp_n[i]*100.,3)} %')
        print(f'')
        if save_out_file==True:
            file.write(f'{selected_title[i]} events\n')
            file.write(f'QN = {round_to_significant_figures(qn[i]*100.,1)} +- {round_to_significant_figures(qn_err[i]*100.,0)} %\n')
            file.write(f'UN = {round_to_significant_figures(un[i]*100.,1)} +- {round_to_significant_figures(un_err[i]*100.,0)} %\n')
            file.write(f'PD = {round_to_significant_figures(pd[i]*100.,1)} +- {round_to_significant_figures(pd_err[i]*100.,0)} %\n')
            file.write(f'PA = {round_to_significant_figures(pa[i]*1.,2)} +- {np.round(pa_err[i]*1.,3)} °\n')
            file.write(f'MDP = {np.round(mdp[i]*100.,3)} %\n')
            file.write(f'{not_selected_title[i]} events\n')
            file.write(f'QN = {round_to_significant_figures(qn_n[i]*100.,1)} +- {round_to_significant_figures(qn_err_n[i]*100.,0)} %\n')
            file.write(f'UN = {round_to_significant_figures(un_n[i]*100.,1)} +- {round_to_significant_figures(un_err_n[i]*100.,0)} %\n')
            file.write(f'PD = {round_to_significant_figures(pd_n[i]*100.,1)} +- {round_to_significant_figures(pd_err_n[i]*100.,0)} %\n')
            file.write(f'PA = {round_to_significant_figures(pa_n[i]*1.,2)} +- {np.round(pa_err_n[i]*1.,3)} °\n')
            file.write(f'MDP = {np.round(mdp_n[i]*100.,3)} %\n')
    if save_out_file==True:
        file.write(f'\n\n')

    # remove .fits files which have been created
    for f in file_sel + file_not:
        if os.path.exists(f):
            os.remove(f)

    for f in pol_pcube_list + pol_pcube_list_n:
        if os.path.exists(f):
            os.remove(f)
      
if save_out_file==True:
    file.close()

"""
    I save the pandas data frame table
    with the useful and necessary 
    polarization results
"""
if save_table:
    d = {'thr': THRESHOLDS, 'N_events': EVT, 'QN': QN, 'QN_ERR': QN_ERR, 'UN': UN, 'UN_ERR': UN_ERR, 'PD': PD, 'PD_ERR': PD_ERR, 'PA': PA, 'PA_ERR': PA_ERR, 'MDP': MDP}
    df = pandas.DataFrame(data=d)
    df.to_csv(f'/Users/stefanotugliani/Desktop/dati_ixpe/swift/event_files/02250901/ALL_DU_merged_{method}_{seg_size}s_v.csv', index=False)

# plt.show()
