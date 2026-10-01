
import numpy as np
import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pf
import os
import math
import pandas

# sys.path.insert(0,'/Users/stefanotugliani/Desktop/analisi/analysis_functions')
sys.path.insert(0, '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/analisi')
from polarization_plots import *

from ixpeobssim.binning.polarization import xBinnedPolarizationCube, xBinnedCountSpectrum
from ixpeobssim.binning.misc import xBinnedLightCurve
import ixpeobssim.core.pipeline as pipeline
from ixpeobssim.evt.event import xEventFile

import argparse


formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

parser.add_argument(
    '-seg',
    '--seg_size',
    type=str,
    help='segment size',
    required=True
)

parser.add_argument(
    '-m',
    '--method',
    type=str,
    help='method for selection',
    required=True
)

parser.add_argument(
    '-s',
    '--save',
    action='store_true',
    help='Do you want to save the figures ?',
    required=False,
    default=False
)

parser.add_argument(
    '-sf',
    '--save_out_file',
    action='store_true',
    help='Do you want to save the output file ?',
    required=False,
    default=False
)

parser.add_argument(
    '-st',
    '--save_table',
    action='store_true',
    help='Do you want to save the output csv table ?',
    required=False,
    default=False
)

parser.add_argument(
    '-ebin',
    '--energy-bin',
    type=str,
    help='energy bin label: E2-4keV or E4-8keV',
    choices=['E2-4keV', 'E4-8keV'],
    required=True
)


args = parser.parse_args()


seg_size = args.seg_size
method = args.method
save = args.save
save_out_file = args.save_out_file
save_table = args.save_table


PATH = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'


DU = [
    f'{PATH}ixpe02250901_det1_evt2_v01_src.fits',
    f'{PATH}ixpe02250901_det2_evt2_v01_src.fits',
    f'{PATH}ixpe02250901_det3_evt2_v01_src.fits'
]


# ============================================================
# ENERGY BINNING
# ============================================================

ENERGY_BIN_LABEL = args.energy_bin

if ENERGY_BIN_LABEL == 'E2-4keV':

    ENERGY_BINNING = [2.0, 4.0]

    # Valori passati a xpselect
    emin = 2.0
    emax = 4.0

elif ENERGY_BIN_LABEL == 'E4-8keV':

    ENERGY_BINNING = [4.0, 8.0]

    # Valori passati a xpselect
    emin = 4.0
    emax = 8.0


# Controllo per evitare di passare None a xpselect
if emin is None or emax is None:
    raise ValueError(
        f'Errore: emin={emin}, emax={emax}. '
        f'Controllare ENERGY_BIN_LABEL={ENERGY_BIN_LABEL}'
    )


print('\n========================================')
print('ENERGY BINNING')
print('========================================')
print(f'Energy bin: {ENERGY_BIN_LABEL}')
print(f'emin = {emin}')
print(f'emax = {emax}')
print('========================================\n')


seg_folder = f'{PATH}ALLDU_v/seg{seg_size}s_{ENERGY_BIN_LABEL}'
images_folder = f'{seg_folder}/images'

os.makedirs(images_folder, exist_ok=True)


path_thr = f'{seg_folder}/threshold_ALLdu_new.npy'
THR = np.load(path_thr)


grayfilter_bool = True
acceptance_correction = False


# ============================================================
# FUNCTIONS
# ============================================================

def readsimfitsfile(file_path):
    """Function reads fits file and returns events and GTI."""

    data_f = pf.open(file_path)

    data_f.info()

    events = data_f['EVENTS'].data
    GTI = data_f['GTI'].data

    data_f.close()

    return events, GTI


def run_xpselect(file_path, custom_mask, suffix, emin, emax):

    # Controllo aggiuntivo
    if emin is None or emax is None:
        raise ValueError(
            f'xpselect riceverebbe emin={emin}, emax={emax}'
        )

    print(
        f'Running xpselect: '
        f'emin={emin}, emax={emax}, suffix={suffix}'
    )

    pipeline.xpselect(
        file_path,
        mask=custom_mask,
        suffix=suffix,
        overwrite=True,
        emin=float(emin),
        emax=float(emax)
    )


# ============================================================
# OUTPUT FILE
# ============================================================

outfile = (
    f'{PATH}polarization_ALLDU_new_'
    f'{seg_size}s_{method}_{ENERGY_BIN_LABEL}_v.txt'
)

if save_out_file:
    file = open(outfile, 'w')


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

QN = []
QN_ERR = []

UN = []
UN_ERR = []

PD = []
PD_ERR = []

PA = []
PA_ERR = []

MDP = []

THRESHOLDS = []
EVT = []


# ============================================================
# MASK PATHS
# ============================================================

mask_det1_bool_path = PATH + 'mask_1_allDU.npy'
mask_det2_bool_path = PATH + 'mask_2_allDU.npy'
mask_det3_bool_path = PATH + 'mask_3_allDU.npy'


# ============================================================
# FIRST SELECTION: ENERGY RANGE
# ============================================================

run_xpselect(
    DU[0],
    custom_mask=mask_det1_bool_path,
    suffix='after_merging',
    emin=emin,
    emax=emax
)

run_xpselect(
    DU[1],
    custom_mask=mask_det2_bool_path,
    suffix='after_merging',
    emin=emin,
    emax=emax
)

run_xpselect(
    DU[2],
    custom_mask=mask_det3_bool_path,
    suffix='after_merging',
    emin=emin,
    emax=emax
)


DU_new = [
    f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging.fits',
    f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging.fits',
    f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging.fits'
]


# ============================================================
# READ FITS FILES
# ============================================================

events_1, GTI_1 = readsimfitsfile(DU_new[0])
events_2, GTI_2 = readsimfitsfile(DU_new[1])
events_3, GTI_3 = readsimfitsfile(DU_new[2])


# ============================================================
# LOOP OVER THRESHOLDS
# ============================================================

for thr in THR:

    # --------------------------------------------------------
    # MASK DU1
    # --------------------------------------------------------

    mask_sel_path_1 = (
        f'{seg_folder}/'
        f'mask_{method}_{seg_size}_{thr}_DU1_new.npy'
    )

    mask_not_path_1 = (
        f'{seg_folder}/'
        f'mask_not_{method}_{seg_size}_{thr}_DU1_new.npy'
    )

    mask_sel_1 = np.load(mask_sel_path_1)

    selected_events_1 = len(
        np.where(mask_sel_1 == True)[0]
    )

    not_selected_events_1 = len(
        np.where(mask_sel_1 == False)[0]
    )


    # --------------------------------------------------------
    # MASK DU2
    # --------------------------------------------------------

    mask_sel_path_2 = (
        f'{seg_folder}/'
        f'mask_{method}_{seg_size}_{thr}_DU2_new.npy'
    )

    mask_not_path_2 = (
        f'{seg_folder}/'
        f'mask_not_{method}_{seg_size}_{thr}_DU2_new.npy'
    )

    mask_sel_2 = np.load(mask_sel_path_2)

    selected_events_2 = len(
        np.where(mask_sel_2 == True)[0]
    )

    not_selected_events_2 = len(
        np.where(mask_sel_2 == False)[0]
    )


    # --------------------------------------------------------
    # MASK DU3
    # --------------------------------------------------------

    mask_sel_path_3 = (
        f'{seg_folder}/'
        f'mask_{method}_{seg_size}_{thr}_DU3_new.npy'
    )

    mask_not_path_3 = (
        f'{seg_folder}/'
        f'mask_not_{method}_{seg_size}_{thr}_DU3_new.npy'
    )

    mask_sel_3 = np.load(mask_sel_path_3)

    selected_events_3 = len(
        np.where(mask_sel_3 == True)[0]
    )

    not_selected_events_3 = len(
        np.where(mask_sel_3 == False)[0]
    )


    # --------------------------------------------------------
    # TOTAL EVENTS
    # --------------------------------------------------------

    SUFFIX = f'{seg_size}_{thr}'

    selected_events = (
        selected_events_1
        + selected_events_2
        + selected_events_3
    )

    not_selected_events = (
        not_selected_events_1
        + not_selected_events_2
        + not_selected_events_3
    )


    # ========================================================
    # XPSELECT SELECTED / NOT SELECTED
    # ========================================================

    run_xpselect(
        DU_new[0],
        custom_mask=mask_sel_path_1,
        suffix=SUFFIX,
        emin=None,
        emax=None
    )

    run_xpselect(
        DU_new[0],
        custom_mask=mask_not_path_1,
        suffix=f'{SUFFIX}_not',
        emin=None,
        emax=None
    )

    run_xpselect(
        DU_new[1],
        custom_mask=mask_sel_path_2,
        suffix=SUFFIX,
        emin=None,
        emax=None
    )

    run_xpselect(
        DU_new[1],
        custom_mask=mask_not_path_2,
        suffix=f'{SUFFIX}_not',
        emin=None,
        emax=None
    )

    run_xpselect(
        DU_new[2],
        custom_mask=mask_sel_path_3,
        suffix=SUFFIX,
        emin=None,
        emax=None
    )

    run_xpselect(
        DU_new[2],
        custom_mask=mask_not_path_3,
        suffix=f'{SUFFIX}_not',
        emin=None,
        emax=None
    )


    # ========================================================
    # FILE LISTS
    # ========================================================

    file_sel = [
        f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging_{SUFFIX}.fits',
        f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging_{SUFFIX}.fits',
        f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging_{SUFFIX}.fits'
    ]


    file_not = [
        f'{PATH}ixpe02250901_det1_evt2_v01_src_after_merging_{SUFFIX}_not.fits',
        f'{PATH}ixpe02250901_det2_evt2_v01_src_after_merging_{SUFFIX}_not.fits',
        f'{PATH}ixpe02250901_det3_evt2_v01_src_after_merging_{SUFFIX}_not.fits'
    ]


    # ========================================================
    # XPBIN SELECTED
    # ========================================================

    pol_pcube_list = pipeline.xpbin(
        *file_sel,
        algorithm='PCUBE',
        emin=emin,
        emax=emax,
        ebins=1,
        overwrite=True,
        acceptcorr=acceptance_correction,
        irfname='ixpe:obssim:v12',
        grayfilter=grayfilter_bool
    )


    pcube = xBinnedPolarizationCube.from_file_list(
        pol_pcube_list
    )


    qn = pcube.QN
    un = pcube.UN

    qn_err = pcube.QN_ERR
    un_err = pcube.UN_ERR

    pd = pcube.PD
    pd_err = pcube.PD_ERR

    pa = pcube.PA
    pa_err = pcube.PA_ERR

    mdp = pcube.MDP_99


    # ========================================================
    # XPBIN NOT SELECTED
    # ========================================================

    pol_pcube_list_n = pipeline.xpbin(
        *file_not,
        algorithm='PCUBE',
        emin=emin,
        emax=emax,
        ebins=1,
        overwrite=True,
        acceptcorr=acceptance_correction,
        irfname='ixpe:obssim:v12',
        grayfilter=grayfilter_bool
    )


    pcube_n = xBinnedPolarizationCube.from_file_list(
        pol_pcube_list_n
    )


    qn_n = pcube_n.QN
    un_n = pcube_n.UN

    qn_err_n = pcube_n.QN_ERR
    un_err_n = pcube_n.UN_ERR

    pd_n = pcube_n.PD
    pd_err_n = pcube_n.PD_ERR

    pa_n = pcube_n.PA
    pa_err_n = pcube_n.PA_ERR

    mdp_n = pcube_n.MDP_99


    # ========================================================
    # PLOTS
    # ========================================================

    if save:

        fig, ax = plt.subplots(
            subplot_kw={'projection': 'polar'},
            figsize=(10, 7),
            tight_layout=True
        )


        polarization_contour(
            pol_pcube_list,
            ax=ax,
            text=True,
            global_color=['C0'],
            aspect='half top',
            levels=[0.68, 0.95]
        )


        polarization_contour(
            pol_pcube_list_n,
            ax=ax,
            text=True,
            global_color=['C1'],
            aspect='half top',
            levels=[0.68, 0.95]
        )


        ax.set_thetamin(-30)
        ax.set_thetamax(30)
        ax.set_rmax(7)


        ax.text(
            np.radians(0),
            8.25,
            f'{thr} {seg_size}s {method}',
            ha='center',
            va='center'
        )


        ax.text(
            np.radians(0),
            7.6,
            'Polarization angle [°]',
            ha='center',
            va='center'
        )


        ax.text(
            np.radians(60),
            3,
            'Selected events',
            color='C0',
            ha='center',
            va='center'
        )


        ax.text(
            np.radians(60),
            2.5,
            'Not selected events',
            color='C1',
            ha='center',
            va='center'
        )


        ax.text(
            np.radians(-45),
            2,
            'Polarization degree [%]',
            rotation=60
        )


        ax.text(
            np.radians(0),
            7.75,
            'N'
        )

        ax.text(
            np.radians(-25),
            7.75,
            'W'
        )

        ax.text(
            np.radians(25),
            7.75,
            'E'
        )


        plt.savefig(
            f'{images_folder}/polar_{method}_{SUFFIX}.png'
        )

        plt.close(fig)


        selected_title = create_title(
            ENERGY_BINNING,
            'selected'
        )

        not_selected_title = create_title(
            ENERGY_BINNING,
            'not selected'
        )


        fig1, ax1 = plt.subplots(
            figsize=(7, 7)
        )


        pcube_contour_plot(
            QN=qn,
            QN_ERR=qn_err,
            UN=un,
            UN_ERR=un_err,
            PD=pd,
            PD_ERR=pd_err,
            PA=pa,
            PA_ERR=pa_err,
            MDP_99=mdp,
            title=selected_title,
            ENERGY_BINNING=ENERGY_BINNING,
            ax=ax1,
            grid=False,
            global_color=['C0']
        )


        pcube_contour_plot(
            QN=qn_n,
            QN_ERR=qn_err_n,
            UN=un_n,
            UN_ERR=un_err_n,
            PD=pd_n,
            PD_ERR=pd_err_n,
            PA=pa_n,
            PA_ERR=pa_err_n,
            MDP_99=mdp_n,
            title=not_selected_title,
            ENERGY_BINNING=ENERGY_BINNING,
            ax=ax1,
            grid=True,
            global_color=['C1']
        )


        ax1.set_xlim([-0.1, 0.1])
        ax1.set_ylim([-0.1, 0.1])

        ax1.set_title(
            f'{thr} {seg_size}s {method} pcube'
        )


        plt.savefig(
            f'{images_folder}/pcube_{method}_{SUFFIX}.png'
        )

        plt.close(fig1)


    # ========================================================
    # STORE RESULTS
    # ========================================================

    QN.extend([
        qn[0],
        qn_n[0]
    ])

    QN_ERR.extend([
        qn_err[0],
        qn_err_n[0]
    ])

    UN.extend([
        un[0],
        un_n[0]
    ])

    UN_ERR.extend([
        un_err[0],
        un_err_n[0]
    ])

    PD.extend([
        pd[0],
        pd_n[0]
    ])

    PD_ERR.extend([
        pd_err[0],
        pd_err_n[0]
    ])

    PA.extend([
        pa[0],
        pa_n[0]
    ])

    PA_ERR.extend([
        pa_err[0],
        pa_err_n[0]
    ])

    MDP.extend([
        mdp[0],
        mdp_n[0]
    ])

    THRESHOLDS.extend([
        thr,
        thr
    ])

    EVT.extend([
        selected_events,
        not_selected_events
    ])


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f'\nsegment_size = {seg_size}s\n'
        f'threshold = {thr}'
    )

    print(
        f'selected events: {selected_events}\n'
        f'non-selected events: {not_selected_events}\n'
    )

    print(
        f'method: {method}\n'
    )


    # ========================================================
    # SAVE TXT
    # ========================================================

    if save_out_file:

        file.write(
            f'segment_size = {seg_size}s\n'
            f'threshold = {thr}\n'
        )

        file.write(
            f'selected events: {selected_events}\n'
            f'non-selected events: {not_selected_events}\n'
        )

        file.write(
            f'method: {method}\n'
        )


        file.write(
            'Selected events:\n'
        )


        file.write(
            f'QN = {np.round(qn[0] * 100., 2)} '
            f'+- {np.round(qn_err[0] * 100., 2)} %\n'
        )


        file.write(
            f'UN = {np.round(un[0] * 100., 2)} '
            f'+- {np.round(un_err[0] * 100., 2)} %\n'
        )


        file.write(
            f'PD = {np.round(pd[0] * 100., 2)} '
            f'+- {np.round(pd_err[0] * 100., 2)} %\n'
        )


        file.write(
            f'PA = {np.round(pa[0], 2)} '
            f'+- {np.round(pa_err[0], 3)} °\n'
        )


        file.write(
            f'MDP = {np.round(mdp[0] * 100., 3)} %\n'
        )


        file.write(
            'Not selected events:\n'
        )


        file.write(
            f'QN = {np.round(qn_n[0] * 100., 2)} '
            f'+- {np.round(qn_err_n[0] * 100., 2)} %\n'
        )


        file.write(
            f'UN = {np.round(un_n[0] * 100., 2)} '
            f'+- {np.round(un_err_n[0] * 100., 2)} %\n'
        )


        file.write(
            f'PD = {np.round(pd_n[0] * 100., 2)} '
            f'+- {np.round(pd_err_n[0] * 100., 2)} %\n'
        )


        file.write(
            f'PA = {np.round(pa_n[0], 2)} '
            f'+- {np.round(pa_err_n[0], 3)} °\n'
        )


        file.write(
            f'MDP = {np.round(mdp_n[0] * 100., 3)} %\n\n'
        )


    # ========================================================
    # CLEAN TEMPORARY FILES
    # ========================================================

    for f in (
        file_sel
        + file_not
        + pol_pcube_list
        + pol_pcube_list_n
    ):

        if os.path.exists(f):
            os.remove(f)


# ============================================================
# CLOSE OUTPUT FILE
# ============================================================

if save_out_file:
    file.close()


# ============================================================
# SAVE CSV TABLE
# ============================================================

if save_table:

    d = {
        'thr': THRESHOLDS,
        'N_events': EVT,

        'QN': QN,
        'QN_ERR': QN_ERR,

        'UN': UN,
        'UN_ERR': UN_ERR,

        'PD': PD,
        'PD_ERR': PD_ERR,

        'PA': PA,
        'PA_ERR': PA_ERR,

        'MDP': MDP
    }


    df = pandas.DataFrame(
        data=d
    )


    df.to_csv(
        f'{PATH}ALL_DU_merged_'
        f'{method}_{seg_size}s_'
        f'{ENERGY_BIN_LABEL}_v.csv',
        index=False
    )