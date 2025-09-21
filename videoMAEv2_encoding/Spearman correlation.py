
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as pjoin
# from had_utils import save2cifti, roi_mask
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.stats import spearmanr, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

# 加载数据
rank_normalized_RDM_sub02 = np.load('D:/dataZL/Multi-Modalities-Pretraining/feature_txt_action_sub01/rank_normalized_RDM_sub04.npy')
rank_normalized_RDM_fmri = np.load('D:/dataZL/Multi-Modalities-Pretraining/feature_txt_action_sub01/rank_normalized_RDM_fmri.npy')

# 计算Spearman相关系数
correlation_Spearman,p_values= spearmanr(rank_normalized_RDM_sub02.squeeze(), rank_normalized_RDM_fmri.squeeze())
Kendall_max = np.mean(correlation_Spearman, axis=0)

# 执行t检验
t_statistic, t_p_values = ttest_1samp(p_values, popmean=0)

# 进行FDR校正
rejected, corrected_p_values = fdrcorrection(t_p_values, alpha=0.05)

# 将每个体素除以上限噪声层
subject_noise_ceiling = np.max(rank_normalized_RDM_fmri, axis=0)
normalized_values = rank_normalized_RDM_fmri / subject_noise_ceiling

# 计算平均值
mean_normalized_values = np.mean(normalized_values, axis=0)

# 仅保留显著的体素
# significant_voxels = mean_normalized_values[rejected]
#
# # 显示结果
# print("显著的体素: ", significant_voxels)

# plot fc
plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.imshow(significant_voxels, cmap = 'rainbow')
labels = region_sum
ax.set_xticks(range(4))
ax.set_xticklabels(labels, rotation=60, size=20)
ax.set_yticks(range(4))
ax.set_yticklabels(labels, size=20)
ax.set_title('Spearman correlation', size=30, pad=20)
# show text
for i in range(4):
    for j in range(4):
        text = ax.text(i, j, f'{significant_voxels[j, i]:.3f}', ha='center', va='center', color='black', size=15)
plt.colorbar(ax=ax)
plt.show()