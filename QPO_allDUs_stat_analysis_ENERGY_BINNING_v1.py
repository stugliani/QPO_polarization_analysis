import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from QPO_classes import T_TEST_STAT, Polarization
import argparse
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('-f','--file', type=str,  help='PATH to the csv file', required=True)
parser.add_argument('-du','--DU', type=int,  help='DU number ypu want to analyze, type 0 if all DUs', required=False)
parser.add_argument('-seg','--seg_size', type=str,  help='segment size', required=True)
parser.add_argument('-s','--save', action='store_true',  help='Do you want to save the figures ?', required=False, default=False)
parser.add_argument('-stt','--save-ttest', action='store_true',  help='Do you want to save the t test arrays ?', required=False, default=False)
parser.add_argument('-ebin','--energy-bin', type=str, help='energy bin label: E2-4keV or E4-8keV', choices=['E2-4keV','E4-8keV'], required=True)

args = parser.parse_args()

file_csv = args.file
du = args.DU
if du==0:
    alldus=True
else:
    alldus=False

seg = args.seg_size
save = args.save
save_t = args.save_ttest
ENERGY_BIN_LABEL = args.energy_bin

file_name = (os.path.basename(file_csv)).split('.')[0]

path = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'

# Leggi il file CSV e crea un DataFrame
df = pandas.read_csv(file_csv)
print(df)
THR = df['thr'].values
EVT = df['N_events'].values
QN = df['QN'].values
QN_ERR = df['QN_ERR'].values
UN = df['UN'].values
UN_ERR = df['UN_ERR'].values
PD = df['PD'].values
PD_ERR = df['PD_ERR'].values
PA = df['PA'].values
PA_ERR = df['PA_ERR'].values
MDP = df['MDP'].values

qn = QN[::2]*100.
qn_n = QN[1::2]*100.
qn_err = QN_ERR[::2]*100.
qn_err_n = QN_ERR[1::2]*100.
un = UN[::2]*100.
un_n = UN[1::2]*100.
un_err = UN_ERR[::2]*100.
un_err_n = UN_ERR[1::2]*100.
pd = PD[::2]*100.
pd_n = PD[1::2]*100.
pd_err = PD_ERR[::2]*100.
pd_err_n = PD_ERR[1::2]*100.
pa = PA[::2]
pa_n = PA[1::2]
pa_err = PA_ERR[::2]
pa_err_n = PA_ERR[1::2]
mdp = MDP[::2]*100.
mdp_n = MDP[1::2]*100.
thr = THR[::2]
selected_events = EVT[::2]
not_selected_events = EVT[1::2]


t_stat_q, t_stat_u = [], []
t_stat_pd, t_stat_pa = [], []
pvalues_q, pvalues_u = [], []
pvalues_pd, pvalues_pa = [], []

for i in range(len(thr)):
    TQ = T_TEST_STAT(X_1 = qn[i], X_ERR_1 = qn_err[i], X_2 = qn_n[i], X_ERR_2 = qn_err_n[i], N_1 = selected_events[i], N_2 = not_selected_events[i], CL = 0.95)
    TU = T_TEST_STAT(X_1 = un[i], X_ERR_1 = un_err[i], X_2 = un_n[i], X_ERR_2 = un_err_n[i], N_1 = selected_events[i], N_2 = not_selected_events[i], CL = 0.95)

    POL = Polarization(qn=qn[i],un=un[i],qn_err=qn_err[i],un_err=un_err[i])
    POL_N = Polarization(qn=qn_n[i],un=un_n[i],qn_err=qn_err_n[i],un_err=un_err_n[i])

    TPD = T_TEST_STAT(X_1 = POL.PD(), X_ERR_1 = POL.PD_ERR(), X_2 = POL_N.PD(), X_ERR_2 = POL_N.PD_ERR(), N_1 = selected_events[i], N_2 = not_selected_events[i], CL = 0.95)
    TPA = T_TEST_STAT(X_1 = POL.PA(), X_ERR_1 = POL.PA_ERR(), X_2 = POL_N.PA(), X_ERR_2 = POL_N.PA_ERR(), N_1 = selected_events[i], N_2 = not_selected_events[i], CL = 0.95)

    t_stat_q.append(TQ.t())
    pvalues_q.append(TQ.p_value())
    t_stat_u.append(TU.t())
    pvalues_u.append(TU.p_value())

    t_stat_pd.append(TPD.t())
    pvalues_pd.append(TPD.p_value())
    t_stat_pa.append(TPA.t())
    pvalues_pa.append(TPA.p_value())

    print(f'Threshold = {thr[i]}')
    print(f'    Q: t stat = {TQ.t()}, p value = {TQ.p_value()*100.} %')
    print(f'    U: t stat = {TU.t()}, p value = {TU.p_value()*100.} %')
    print(f'    PD: t stat = {TPD.t()}, p value = {TPD.p_value()*100.} %')
    print(f'    PA: t stat = {TPA.t()}, p value = {TPA.p_value()*100.} %')
    print()


t_stat_q = np.array(t_stat_q)
t_stat_u = np.array(t_stat_u)
pvalues_q = np.array(pvalues_q)
pvalues_u = np.array(pvalues_u)

t_stat_pd = np.array(t_stat_pd)
pvalues_pd = np.array(pvalues_pd)
t_stat_pa = np.array(t_stat_pa)
pvalues_pa = np.array(pvalues_pa)

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(12,8), sharex=True)
plt.suptitle(f'{file_name} {ENERGY_BIN_LABEL}', fontsize=16)
plt.subplots_adjust(top=0.9)
ax1.errorbar(x=thr,y=qn,yerr=qn_err,capsize=7,linestyle='',marker='o',label='QN selected')
ax1.errorbar(x=thr,y=qn_n,yerr=qn_err_n,capsize=7,linestyle='',marker='o',label='QN not selected')
ax2.errorbar(x=thr,y=un,yerr=un_err,capsize=7,linestyle='',marker='o',label=' UN selected')
ax2.errorbar(x=thr,y=un_n,yerr=un_err_n,capsize=7,linestyle='',marker='o',label=' UN not selected')
ax5.errorbar(x=thr,y=t_stat_q,linestyle='',marker='+',label='t stat Q')
ax6.errorbar(x=thr,y=t_stat_u,linestyle='',marker='+',label='t stat U')
ax1.grid('both')
ax2.grid('both')
ax5.grid('both')
ax6.grid('both')
ax1.legend(loc='upper left')
ax2.legend()
ax1.set_ylabel('QN [%]')
ax2.set_ylabel('UN [%]')
ax5.set_ylabel('QN t stat')
ax6.set_ylabel('UN t stat')
ax1.set_title('QN')
ax2.set_title('UN')
plt.subplots_adjust(hspace=0.05)
# fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,7), sharex=True)
ax3.errorbar(x=thr,y=pvalues_q*100.,linestyle='dotted',marker='.',markersize=8,label='QN pvalues')
ax4.errorbar(x=thr,y=pvalues_u*100.,linestyle='dotted',marker='.',markersize=8,label='UN pvalues')
ax3.grid('both')
ax4.grid('both')
ax3.legend()#loc='upper left')
ax4.legend()
# ax1.set_ylim([-0.5, 5])
ax3.set_ylim([-5,100])
ax4.set_ylim([-5,100])
ax3.axhline(y=5,color='black',linestyle='dashdot')
ax4.axhline(y=5,color='black',linestyle='dashdot')
ax3.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax4.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax3.set_ylabel('pvalues QN [%]')
ax4.set_ylabel('pvalues UN [%]')
ax5.set_xlabel('thresholds')
ax6.set_xlabel('thresholds')
ax4.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax3.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])

############### PD and PA

fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(12,8), sharex=True)
plt.suptitle(f'{file_name} {ENERGY_BIN_LABEL}', fontsize=16)
plt.subplots_adjust(top=0.9)
ax1.errorbar(x=thr,y=pd,yerr=pd_err,capsize=7,linestyle='',marker='o',label='PD selected')
ax1.errorbar(x=thr,y=pd_n,yerr=pd_err_n,capsize=7,linestyle='',marker='o',label='PD not selected')
ax2.errorbar(x=thr,y=pa,yerr=pa_err,capsize=7,linestyle='',marker='o',label=' PA selected')
ax2.errorbar(x=thr,y=pa_n,yerr=pa_err_n,capsize=7,linestyle='',marker='o',label=' PA not selected')
ax5.errorbar(x=thr,y=t_stat_pd,linestyle='',marker='+',label='t stat PD')
ax6.errorbar(x=thr,y=t_stat_pa,linestyle='',marker='+',label='t stat PA')
ax1.grid('both')
ax2.grid('both')
ax5.grid('both')
ax6.grid('both')
ax1.legend(loc='upper left')
ax2.legend()
ax1.set_ylabel('PD [%]')
ax2.set_ylabel('PA [°]')
ax5.set_ylabel('PD t stat')
ax6.set_ylabel('PA t stat')
ax1.set_title('PD')
ax2.set_title('PA')
plt.subplots_adjust(hspace=0.05)
# fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,7), sharex=True)
ax3.errorbar(x=thr,y=pvalues_q*100.,linestyle='dotted',marker='.',markersize=8,label='PD pvalues')
ax4.errorbar(x=thr,y=pvalues_u*100.,linestyle='dotted',marker='.',markersize=8,label='PA pvalues')
ax3.grid('both')
ax4.grid('both')
ax3.legend()#loc='upper left')
ax4.legend()
# ax1.set_ylim([-0.5, 5])
ax3.set_ylim([-5,100])
ax4.set_ylim([-5,100])
ax3.axhline(y=5,color='black',linestyle='dashdot')
ax4.axhline(y=5,color='black',linestyle='dashdot')
ax3.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax4.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax3.set_ylabel('pvalues PD [%]')
ax4.set_ylabel('pvalues PA [%]')
ax5.set_xlabel('thresholds')
ax6.set_xlabel('thresholds')
ax4.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])
ax3.set_xlim([np.min(thr)-0.1,np.max(thr)+0.1])

if save==True:
    if alldus==False:
        plt.savefig(path+f'DU{du}/seg{seg}s_{ENERGY_BIN_LABEL}/images/t_test.png', dpi=300)
    else:
        plt.savefig(path+f'ALLDU_v/seg{seg}s_{ENERGY_BIN_LABEL}/images/t_test.png', dpi=300)

if save_t==True:
    ttest_data = np.column_stack((pvalues_q, pvalues_u))
    if alldus==False:
        np.save(path+f'DU{du}/ttest_data_{seg}s_{ENERGY_BIN_LABEL}.npy', ttest_data)
    if alldus==True:
        np.save(path+f'ALLDU_v/ttest_data_{seg}s_{ENERGY_BIN_LABEL}.npy', ttest_data)

plt.show()

# # Visualizza il DataFrame
# print(df)