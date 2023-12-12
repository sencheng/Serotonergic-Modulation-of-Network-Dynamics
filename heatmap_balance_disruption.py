import pickle
from typing import List

from matplotlib import colors
from matplotlib import pyplot as plt

from searchParams import *
from defaultParams import *

PATH = '/work/mohammad/5HT2A/simdata/fr-baseline-disrupt-balance-with-pert/'
FILENAME = 'diff_data'
FORMATS = ['svg', 'png']
os.makedirs('figs', exist_ok=True)
plt.set_cmap('bwr')

def label_figures(axes: List[plt.Axes]):
    for ax in axes:
        ax.set_xlabel('Perturbation of inihibitory population (%)')
        ax.set_ylabel('Perturbation of excitatory population (%)')

with open(os.path.join(PATH, FILENAME), 'rb') as fl:
    data = pickle.load(fl)

pop_ind_map = {'exc': 0, 'inh': 1}
pert0 = nn_stim_rng[0]
pert = nn_stim_rng[1]
fig_exc, ax_exc = plt.subplots()
fig_inh, ax_inh = plt.subplots()
frchg_map_exc = np.zeros((EEconn_chg_factor.size, EIconn_chg_factor.size))
frchg_map_inh = np.zeros_like(frchg_map_exc)
fig_exc, ax_exc = plt.subplots()
fig_inh, ax_inh = plt.subplots()
frchg_map_exc = np.zeros((EEconn_chg_factor.size, EIconn_chg_factor.size))
frchg_map_inh = np.zeros_like(frchg_map_exc)
for id_e, pert_e in enumerate(EEconn_chg_factor):
    for id_i, pert_i in enumerate(EIconn_chg_factor):
        frchg_map_exc[id_e, id_i] = data[pert_e][pert_i]['exc'][pert]
        frchg_map_inh[id_e, id_i] = data[pert_e][pert_i]['inh'][pert]
norm_exc = colors.TwoSlopeNorm(vmax=frchg_map_exc.max(), vmin=frchg_map_exc.min(), vcenter=1)
norm_inh = colors.TwoSlopeNorm(vmax=frchg_map_inh.max(), vmin=frchg_map_inh.min(), vcenter=1)
im_exc = ax_exc.pcolor(EEconn_chg_factor, EIconn_chg_factor, frchg_map_exc, norm=norm_exc)
im_inh = ax_inh.pcolor(EEconn_chg_factor, EIconn_chg_factor, frchg_map_inh, norm=norm_inh)
cbar_exc = fig_exc.colorbar(im_exc)
cbar_inh = fig_inh.colorbar(im_inh)
cbar_exc.set_ticks(np.concatenate((np.arange(0.2, 1.1, 0.2), np.arange(20, 81, 20))))
cbar_inh.set_ticks(np.concatenate((np.arange(0.2, 1.1, 0.2), np.arange(20, 81, 20))))
label_figures([ax_exc, ax_inh])
cbar_exc.set_label('Firing rate (norm.)')
cbar_inh.set_label('Firing rate (norm.)')
for format in FORMATS:
    fig_exc.savefig(f'figs/diff-map-exc-{pert}.{format}')
    fig_inh.savefig(f'figs/diff-map-inh-{pert}.{format}')