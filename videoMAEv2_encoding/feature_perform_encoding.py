# -*- coding: utf-8 -*-
import os
import time

import numpy as np
from sklearn.metrics import r2_score

import src.data.utils as data_utils
import src.runner.utils as runner_utils
from src import settings
from src.runner.metric import evaluation_metrics, RDM_correlation
import sys
sys.path.append('E:/LLM/Algonauts_2023/fMRI-video-encoding-2023-lhy/')

from utils.ols import OLS_pytorch

import scipy.io as io
def predict_fmri_fast(train_activations, test_activations, train_fmri, use_gpu=False):
    """This function fits a linear regressor using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    Parameters
    ----------


    """

    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test
def main():
    # init model type
    model_type = 'InternVideo-MM-B-16-768'  # 如果是Marlin的模型，需要加上Marlin/，如Marlin/marlin_vit_base_ytf_768  InterVideo、Marlin、UniFormerV2、videoMAEV2
    # init 降维之后的特征数据路径
    root_dir = "E:/LLM/Algonauts_2023/compress_features/InterVideo/InternVideo-MM-B-16-768"#root_dir = "E:/LLM/Algonauts_2023/compress_features/Marlin/marlin_vit_small_ytf_384"
    compression_ratio = [60] # [50, 60, 70, 80, 90, 100]
    compression_method = ['pca'] # ['pca', 'umap', 'autoencoder']

    save_dir = os.path.join(settings.OUTPUT_FOLDER, "OLS_pytorch", model_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    uuid_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    if "Marlin" in model_type:
        # tmp = model_type.split("/")[1]
        log_record_file = save_dir + '/{}_%s.txt'.format(model_type.split("/")[1]) % uuid_str#
    else:
        log_record_file = save_dir + '/{}_%s.txt'.format(model_type) % uuid_str
    file = open(log_record_file, 'a')

    for method in compression_method:
        for num_comps  in compression_ratio:
            if settings.COMPRESSION:
                activations_dir =os.path.join(root_dir, "mean", "{}_{}".format(method, num_comps))
            else:
                activations_dir = root_dir

            # video clip index
            feature_clip_idx = ["sub01", "sub02", "sub03"]
            # feature_clip_idx = ["sub01"]
            # network layer index
            layer_idx = [str(i).zfill(2) for i in range(0, 4)]

            # Loop over all layers to consider and run BO search for each one!
            for i, subject_id in enumerate(feature_clip_idx):
                print(f"predictions {subject_id}...")
                save_pred_path = os.path.join(save_dir, subject_id)
                if not os.path.exists(save_pred_path):
                    os.makedirs(save_pred_path)

                # load fmri data
                fmri_data_train, fmri_data_test = data_utils.get_fmris_data(subject_id)

                # subject specific setup
                fmri_mapping_train = fmri_data_train["mapping"]
                fmri_data_train = fmri_data_train["data"]
                fmri_data_test = fmri_data_test["data"]
                #
                fmri_merged_array = np.concatenate((fmri_data_train, fmri_data_test), axis=0)
                fmri_merged_array = np.array(fmri_merged_array)

                fmri_merged_rdm = []
                for roi in fmri_mapping_train:
                    roi_mapping = fmri_mapping_train[roi]
                    roi_fmri = fmri_merged_array[:, roi_mapping[0]: roi_mapping[1]]

                    save_rsa_path = os.path.join(save_dir, subject_id, f"{subject_id}_{method}_{num_comps}_{roi}_fmri_rsa.png")
                    roi_fmri_rdm = RDM_correlation(roi_fmri, save_rsa_path)

                    fmri_merged_rdm.append(roi_fmri_rdm)

                    save_fmri_rdm_path = os.path.join(save_pred_path, "rdm", "fmri")
                    if not os.path.exists(save_fmri_rdm_path):
                        os.makedirs(save_fmri_rdm_path)

                    np.save(os.path.join(save_fmri_rdm_path, f"{roi}_{method}_{num_comps}_fmri_rdm.npy"),
                            roi_fmri_rdm)
                np.save(os.path.join(save_pred_path, "rdm", f"{method}_{num_comps}_fmri_merge_rdm.npy"),
                        fmri_merged_rdm)

                feature_merged_rdm = []
                for j, layer_id in enumerate(layer_idx):
                    print("===============================")
                    print(f"Start  {method}_{num_comps} subject id {subject_id} layer id {layer_id}...")
                    file.write(f"======================================================\n")
                    file.write(f"Start {method}_{num_comps} subject id {subject_id} layer id {layer_id}... \n")
                    file.write(f"======================================================\n")
                    # load data
                    feature_files_train, feature_files_test = data_utils.get_activations_data(activations_dir, subject_id, layer_id)

                    dataname = f'{model_type}_{subject_id}_{layer_id}_{method}_{num_comps}'
                     # train model
                    y_test_pred = predict_fmri_fast(feature_files_train, feature_files_test, fmri_data_train, use_gpu=False)
                    # evaluate model
                    corr, mse, mae = evaluation_metrics(fmri_data_test, y_test_pred)
                    r2_stats = r2_score(fmri_data_test, y_test_pred)

                    # After done with BO for layer features - save current best predictions
                    #
                    # np.save(os.path.join(save_pred_path, "{}_{}_y_test_pred.npy".format(method, num_comps)),
                    #         y_test_pred)
                    save_feature_rdm_path = os.path.join(save_pred_path, "rdm", "feature")
                    if not os.path.exists(save_feature_rdm_path):
                        os.makedirs(save_feature_rdm_path)
                    #############################################
                    # 计算RSA相似性表征分析
                    #############################################
                    # 合并训练集和测试集的特征向量
                    feature_merged_array = np.concatenate((feature_files_train, feature_files_test), axis=0)
                    save_path = os.path.join(save_pred_path,  f"{subject_id}_{method}_{num_comps}_{layer_id}_feature_rsa.png")
                    # 计算特征向量的RDM
                    feature_rdm = RDM_correlation(feature_merged_array,save_path)
                    feature_merged_rdm.append(feature_rdm)
                    # 保存特征向量的RDM
                    np.save(os.path.join(save_feature_rdm_path, f"{layer_id}_{method}_{num_comps}_feature_rdm.npy"),
                            feature_rdm)

                    #############################################

                    # rdm = RDM_correlation(fmri_data_test, y_test_pred)
                    stats_tick = {"subject_id": subject_id,
                                  "layer_id": layer_id,
                                  "mse_mean": mse,
                                  "mae_mean": mae,
                                  "corr_mean": corr,
                                  "corr_max": corr.max(),
                                  "r2_score": r2_stats,
                                  # "rdm_score": rdm
                                  }

                    print(f"Stats for {dataname}: {stats_tick}")
                    file.write(f"MEAN.....\n")
                    file.write(f"mse_mean：{mse} , mae_mean：{mae} \n")

                    score, roi_score = runner_utils.print_score(fmri_data_test, y_test_pred, fmri_mapping_train, "\n\t")
                    print("===============================")
                    file.write(f" Score: {score} - ")
                    save_feature_roi_cor_path = os.path.join(save_pred_path, "roi_cor")
                    if not os.path.exists(save_feature_roi_cor_path):
                        os.makedirs(save_feature_roi_cor_path)
                    for roi in roi_score:
                        file.write(f"{roi}： ")
                        for key in roi_score[roi].keys():
                            if key in "score":
                                io.savemat(os.path.join(save_feature_roi_cor_path,"{}_{}_{}_{}.mat".format(layer_id,method,num_comps,roi)), {'COR': roi_score[roi][key]})
                                continue
                            file.write(f"{key}: {roi_score[roi][key]:.3f}")
                        # print(f"{roi}: {roi_score[roi]:.3f}", end=", ")
                        file.write("\n")

                    file.write(f"\n")
                    file.write(f"======================================================\n")
                    file.write(f"\n")
                    file.write(f"\n")

                np.save(os.path.join(save_pred_path, "rdm", f"{method}_{num_comps}_feature_merge_rdm.npy"),
                        feature_merged_rdm)
    file.close()

if __name__ == "__main__":
    main()
