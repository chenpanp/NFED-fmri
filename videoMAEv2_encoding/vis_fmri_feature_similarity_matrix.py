# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
from nilearn import plotting
from sklearn.preprocessing import StandardScaler

from utils.helper import load_dict, saveasnii
from numpy import mean
import src.data.utils as data_utils
from src.data.dataset import VidFMRIDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from autosklearn.regression import AutoSklearnRegressor
from src import settings
import src.runner.utils as runner_utils
from src.runner.metric import evaluation_metrics,RDM_correlation
from utils.ols import OLS_pytorch
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import seaborn as sns

def predict_fmri_fast(train_activations, test_activations, train_fmri, use_gpu=False):
    """This function fits a linear regressor using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    Parameters
    ----------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos.
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    use_gpu : bool
        Description of parameter `use_gpu`.

    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x  #voxels
        containing predicted fMRI responses to test videos .

    """

    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test
def main():
    # init model type
    model_type = 'Marlin/marlin_vit_base_ytf_384'  # 如果是Marlin的模型，需要加上Marlin/，如Marlin/marlin_vit_base_ytf_768
    # init 降维之后的特征数据路径
    root_dir = "E:/LLM/Algonauts_2023/compress_features/Marlin/marlin_vit_small_ytf_384"
    compression_ratio = [50, 60, 70, 80, 90, 100]
    compression_method = ['pca'] # ['pca', 'umap', 'autoencoder']

    save_dir = os.path.join(settings.OUTPUT_FOLDER, "OLS_pytorch", model_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for method in compression_method:
        for num_comps  in compression_ratio:
            # video clip index
            feature_clip_idx = ["sub01", "sub02", "sub03", "sub04"]
            # network layer index
            # Loop over all layers to consider and run BO search for each one!
            for i, subject_id in enumerate(feature_clip_idx):
                save_subject_id_path = os.path.join(save_dir, subject_id, "rdm")
                normalized_similarity_matrix_path = os.path.join(save_subject_id_path,
                             "{}_{}_{}_fmri_feature_similarity_matrix.npy".format(subject_id, method, num_comps))
                normalized_similarity_matrix = np.load(normalized_similarity_matrix_path)
                print(normalized_similarity_matrix)


if __name__ == "__main__":
    main()
