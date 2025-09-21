import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join as pjoin
import matplotlib.font_manager as font_manager
import h5py

# define path and params

dataset_path = ''
fmriprep_path = pjoin(dataset_path, ' ', ' ')
support_path = ' '
# change to path of current file
os.chdir(os.path.abspath(''))

sub_dirs = [ _ for _ in os.listdir(fmriprep_path) if 'sub' in _ and '.' not in _ ]
sub_dirs.sort()

tsnr_sum = {}





# loop to get frame displacement value
for sub_dir in sub_dirs:
    FD_tmp = []
    files = [ _ for _ in os.listdir(pjoin(fmriprep_path, sub_dir)) if 'tnsr.mat' in _ \
                and int(re.findall(r'run-\d+', _)[0].split('-')[-1])<=1]
    files.sort()
    for file in files:
        # df = pd.read_excel(os.path.join(fmriprep_path, sub_dir, file))
        df_1 = h5py.File(os.path.join(fmriprep_path, sub_dir, file))
        df = df_1['tsnr'][:,:]

        tsnr = np.mean(np.abs(df), axis=1)


        tsnr_tmp.extend(tsnr.tolist())
    tsnr_sum[sub_dir] = tsnr_tmp

    print(f'Finish {sub_dir} with {len(files)} runs')
    print(np.mean(tsnr_tmp))

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value



fig, ax = plt.subplots(figsize=(18, 6))#fig, ax = plt.subplots(figsize=(20, 5))
width = 0.5

median = []
for i, sub_name in enumerate(tsnr_sum.keys()):
    data = tsnr_sum[sub_name]
    parts = ax.violinplot(data, positions=[i], widths=width, showextrema=False, showmedians=False, vert=True)
    for pc in parts['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)


    q1, medians, q3 = np.percentile(data, [25, 50, 75], axis=0)#
    median.append(medians)
    whiskers = np.array([adjacent_values(np.sort(data), q1, q3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    ax.scatter(i, medians, marker='o', color='#B73E3E', s=20, zorder=3)

    data_max = np.max(data)
    data_min = np.min(data)

    ax.vlines(i, ymin= data_min, ymax=data_max, colors='k', linewidth=1.5)


    ax.hlines(y=data_min, xmin=i - 0.06, xmax=i + 0.06, colors='k', linewidth=1.5)
    ax.hlines(y=data_max, xmin=i - 0.06, xmax=i + 0.06, colors='k', linewidth=1.5)

    ax.vlines(i, q1, q3, color='red', edgecolors='#E97777', linestyle='-', lw=3.6)

# define plot details
font = font_manager.FontProperties(fname=pjoin(support_path, 'arial.ttf'), size=20, weight='bold')
ax.set_ylim([-20, 250])
ax.set_yticks(np.arange(0,300, 50))

ax.set_xlim([-1,5])




ax.set_xticks(np.arange(5))

ax.set_xticklabels(['%02d' % (i + 1) for i in np.arange(5)])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2.0)
ax.spines['bottom'].set_linewidth(2.0)
ax.tick_params(labelsize=18, width=2, length=6, direction="in", pad=7)

plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.ylabel('tSNR', fontproperties=font, size=20, labelpad=15)
plt.xlabel('subID', fontproperties=font, size=25, labelpad=15)
plt.show()
fig.savefig(f'E:/Retinotopy/FD/Temporal_SNR.jpg',dpi=300,bbox_inches='tight')