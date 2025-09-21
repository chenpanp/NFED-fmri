import os
import torch
import random

import matplotlib.pylab as plt
from src.runner.metric import vectorized_correlation
import scipy.io as io
import numpy as np
from sklearn import linear_model
def get_roi_score(fmri_pred, fmri_true, mapping, index):
    fmri_pred = np.array(fmri_pred)
    fmri_true = np.array(fmri_true)
    roi_score = {}

    for roi in mapping:
        roi_mapping = mapping[roi]
        roi_true = fmri_true[:, roi_mapping[0] : roi_mapping[1]]
        roi_pred = fmri_pred[:, roi_mapping[0] : roi_mapping[1]]
        score = vectorized_correlation(roi_pred, roi_true, reduction=None)
        # io.savemat('E:/LLM/Algonauts_2023/compress_features/videoMAEV2/' + roi + '.mat', {'COR': score})
        index_score = np.sort(score)[::-1][:index]

        roi_score[roi] = {
            "score":score,
            "max": np.max(index_score), # 选取前100个最大的值
            "min": np.min(index_score), # 选取前100个最小的值
            "median": np.median(index_score), # 选取前100个中位数
            "mean": np.mean(index_score), # 选取前100个平均值
            "std": np.std(index_score), # 选取前100个标准差
            "var": np.var(index_score), # 选取前100个方差
        }

    return roi_score


def print_score(fmri_pred, fmri_true, mapping, start_string=""):
    score = vectorized_correlation(fmri_pred, fmri_true)
    roi_score = get_roi_score(fmri_pred, fmri_true, mapping, index=100)
    
    print(f"{start_string}", end="")
    print(f"Score: {score} - ", end="")
    for roi in roi_score:
        print(f"{roi}: ", end="")
        for key in roi_score[roi].keys():
            if key in "score":
                continue
            print(f"{key}: {roi_score[roi][key]:.3f}", end=", ")
        # print(f"{roi}: {roi_score[roi]:.3f}", end=", ")
        print("\n")
    print("")
    return score, roi_score


def plot_output(targets, outputs, history):
    fig, axes = plt.subplots(3, 3, figsize=(24, 8))
    axes = axes.ravel()
    for i in range(6):
        axes[i].plot(targets[-i], label="target")
        axes[i].plot(outputs[-i], label="output")

        RMSE = np.sqrt(np.mean((targets[-i] - outputs[-i]) ** 2))
        axes[i].title.set_text(f"RMSE: {RMSE}")
        axes[i].legend()

    axes[6].plot(history["loss"])
    axes[6].title.set_text("Loss")
    axes[7].plot(history["score"])
    axes[7].title.set_text("Score")

    corr = vectorized_correlation(targets, outputs, reduction=None)
    history["corr"].append(sorted(corr))
    axes[8].plot(np.transpose(history["corr"]))
    axes[8].title.set_text("Voxel correlation")
    axes[8].legend([str(i) for i in range(len(history["score"]))])

    plt.show()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
