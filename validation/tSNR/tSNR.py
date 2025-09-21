import nibabel as nib
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.io as sio
import time, os, pickle, json
import matplotlib.pyplot as plt
import matplotlib as mpl
# from nod_utils import save_ciftifile
from os.path import join as pjoin
from matplotlib import font_manager
from scipy.io import loadmat







font_manager.fontManager.addfont('E:/Retinotopy/NOD-fmri-main/validation/supportfiles/arial.ttf')
# define plot utils
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams.update({'font.size': 12, 'font.family': 'Arial', 'mathtext.fontset': 'stix'})

# define calc tSNR function
def tsnr(data):
    mean = data.mean(axis=-1)
    std = data.std(axis=-1)
    return np.nan_to_num(mean/std)

data_dir = 'F:/CPPdata_nii/GLM_surface20230928/sub05'
hemis = ['lh', 'rh']
# runix = range(1, 5)
# runix = range(5, 9)
# runix = range(9, 41)
runix = range(41, 69)
stim_resp_map={}
data = []
data_list = []
current_data = []
for p in runix:
    for hm in range(2):
        data_file = f"{data_dir}/surface_run0{p}/{hemis[hm]}.surfacedata.mat"
        vol = loadmat(data_file)["vals"]
        if len(current_data) == 0:
            current_data = vol
        else:
            current_data = np.vstack((current_data, vol)).transpose((1,0))
    stim_resp_map[p] = np.array(current_data)
    vol = np.array([])
    current_data= np.array([])
task_data = np.dstack(tuple([stim_resp_map[_] for _ in list(stim_resp_map.keys())]))
task_data = task_data.transpose((2,1,0)).astype(np.float32)





# load data
dataset_path = '  '
ciftify_dir = f'{dataset_path}/'
result_dir = 'results/'
beta_path = '  '
task_name = 'face_video'
sub_names = ['sub-{:02d}'.format(i) for i in range(1,2)]
n_run = None
all_sub_tsnr_mean = np.zeros((len(sub_names),), dtype=object)
tsnr_sub_path1=' '
tsnr_sub_path2=' '

for i, sub_name in enumerate(sub_names):
    # handle special subject
    tsnr_sub_path = pjoin(beta_path, sub_name, f'{sub_name}-tsnr.npy')
    if not os.path.exists(tsnr_sub_path):
        dtseries_sum = task_data
        tsnr_sum = np.zeros_like(dtseries_sum[:,:,0])
        n_run = dtseries_sum.shape[0]
        for run in range(n_run):
            run_data = dtseries_sum[run,:,:]
            tsnr_sum[run, :] = tsnr(run_data)
        tsnr_sub1 = tsnr_sum.mean(axis=0)
        tsnr_sub = tsnr_sum.mean(axis=1)
        np.save(pjoin(tsnr_sub_path1, f'sub-05_tsnr_mean.npy'),tsnr_sub1)
        np.save(pjoin(tsnr_sub_path2, f'sub-05_tsnr_mean.npy'), tsnr_sub)
    else:
        tsnr_sub = np.load(tsnr_sub_path)
    # concatenate data
    all_sub_tsnr_mean[i] = tsnr_sub

np.save(pjoin(beta_path, f'sub-{task_name}_tsnr_mean.npy'), all_sub_tsnr_mean)

# For localizer data
# load data
# dataset_path = 'E:/Retinotopy/NOD-fmri-main/validation'
# ciftify_dir = f'{dataset_path}/derivatives/ciftify'
# result_dir = 'E:/Retinotopy/NOD-fmri-main/validation/ciftify/results/'
# beta_path = 'E:/Retinotopy/NOD-fmri-main/validation/supportfiles/'
#
# task_name = 'floc'
#
# sub_names = sorted(['sub-{:02d}'.format(i) for i in range(1,2)])
#
# all_sub_tsnr_mean = np.zeros((len(sub_names),), dtype=object)
#
# for i, sub_name in enumerate(sub_names):
#     # handle special subject
#     tsnr_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_localizer-tsnr.npy')
#     if not os.path.exists(tsnr_sub_path):
#         dtseries_sum = get_task_time_series(sub_name, ciftify_dir, result_dir, task_name)
#         tsnr_sum = np.zeros_like(dtseries_sum[:,:,0])
#         n_run = dtseries_sum.shape[0]
#         for run in range(n_run):
#             run_data = dtseries_sum[run,:,:]
#             tsnr_sum[run, :] = tsnr(run_data)
#         tsnr_sub = tsnr_sum.mean(axis=1)
#         np.save(tsnr_sub_path, tsnr_sub)
#     else:
#         tsnr_sub = np.load(tsnr_sub_path)
#     # collect data
#     all_sub_tsnr_mean[i] = tsnr_sub
# np.save(pjoin(beta_path, f'sub-{task_name}_tsnr_mean.npy'), all_sub_tsnr_mean)

