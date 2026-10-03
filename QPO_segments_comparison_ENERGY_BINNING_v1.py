import argparse
import numpy as np
import matplotlib.pyplot as plt

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-du', '--DU', type=int, help='DU number you want to analyze, type 0 if all DUs', required=False, default=0)
parser.add_argument('-ebin', '--energy-bin', type=str, help='energy bin label: E2-4keV or E4-8keV', choices=['E2-4keV', 'E4-8keV'], required=True)

args = parser.parse_args()

du = args.DU
ENERGY_BIN_LABEL = args.energy_bin

if du == 0:
    alldus = True
else:
    alldus = False

# Path con sintassi / come nel file originale
path_ = 'C:/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'

if alldus:
    path = path_ + f'ALLDU_v/'
else:
    path = path_ + f'DU{du}/'

segs = np.array([70, 75,80, 85,90])

thr_array, ttest_pvalue_q, ttest_pvalue_u = [], [], []

for i in range(len(segs)):
    # 1. Caricamento p-value
    ttest_pvalue_ = np.load(path + f'ttest_data_{segs[i]}s_{ENERGY_BIN_LABEL}.npy')
    ttest_pvalue_q.append(ttest_pvalue_[:, 0])
    ttest_pvalue_u.append(ttest_pvalue_[:, 1])

    # 2. FIX PER IL RANGE DELLE SOGLIE (devono essere 56 elementi come i p-value)
    try:
        # Tenta di caricare il file dedicato alle soglie se esiste
        thr_ = np.load(path + f'seg{segs[i]}s_{ENERGY_BIN_LABEL}/threshold_array.npy')
    except FileNotFoundError:
        try:
            thr_ = np.load(path + f'seg{segs[i]}s/threshold_array.npy')
        except FileNotFoundError:
            # Se il file .npy delle soglie non esiste, ricrea l'asse X con 56 punti 
            # corrispondenti alla dimensione del p-value (es. da 0 a 55 o il range usato)
            num_thresholds = len(ttest_pvalue_[:, 0])
            thr_ = np.arange(num_thresholds)  # Oppure np.linspace(min_thr, max_thr, num_thresholds)

    thr_array.append(thr_)

marker_list = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'X', 'D', 'P', '8']

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for i in range(len(segs)):
    ax[0].plot(thr_array[i], np.array(ttest_pvalue_q[i]) * 100., label=f'{segs[i]}s', color=f'C{i}', marker='', linestyle='-')
    ax[1].plot(thr_array[i], np.array(ttest_pvalue_u[i]) * 100., label=f'{segs[i]}s', color=f'C{i}', marker='', linestyle='-')

for a in ax:
    a.axhspan(0, 5, facecolor='lightgray', alpha=0.5)
    a.axhline(y=5, color='black', linestyle='--')

ax[1].set_xlabel('thresholds')
ax[0].set_ylabel('pvalue q')
ax[1].set_ylabel('pvalue u')

ax[0].axhline(y=5, color='black')
ax[1].axhline(y=5, color='black')

ax[0].grid(True)
ax[1].grid(True)
ax[0].legend()
ax[1].legend()
ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.subplots_adjust(hspace=0.05)

fig.savefig(path_ + f'pvalue_thresholds_Q_U_seg_comp_{ENERGY_BIN_LABEL}.png', dpi=300, bbox_inches='tight')

plt.show()