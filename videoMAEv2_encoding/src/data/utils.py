from src import settings
import src.data.handler as data_handler

import os
import numpy as np

import albumentations as A
import albumentations.pytorch as APT
import cv2
from sklearn.preprocessing import StandardScaler
import glob
# TRANSFORMATIONS
def get_train_transform():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.IAAPerspective(p=0.25),
            A.CoarseDropout(p=0.5),
            A.RandomBrightness(p=0.25),
            A.ToFloat(),
        ]
    )


def get_test_transform():
    return A.Compose([A.Resize(settings.IMG_SIZE, settings.IMG_SIZE), A.ToFloat(),])


# GET CHALLANGE FMRI DATA
def get_fmri_data_for_sub(data_type="tr", ROIs=settings.ROIs, len=settings.train_data_len,subject_id="sub01"):
    fmri_ROI_mapping = {}
    fmri_voxel_total = 0
    print(int(subject_id[-1]))
    idx = int(subject_id[-1]) -1
    # get selected voxel count for each ROI and create data mapping
    for ROI in ROIs:
        ROI_data, _ = data_handler.get_fmri(settings.FMRI_FOLDER , data_type+"_"+ROI)
        ROI_data = np.array(ROI_data)[idx::3, :] #
        ROI_voxel_count = ROI_data.shape[1]
        fmri_ROI_mapping[ROI] = [
            fmri_voxel_total,
            fmri_voxel_total + ROI_voxel_count,
            ROI_voxel_count,
        ]
        fmri_voxel_total += ROI_voxel_count

    # load fmri data
    sub_fmri_data = np.empty((len, fmri_voxel_total))
    for ROI in ROIs:
        ROI_mapping = fmri_ROI_mapping[ROI]
        ROI_data, _ = data_handler.get_fmri(settings.FMRI_FOLDER, data_type+"_"+ ROI)
        ROI_data = np.array(ROI_data)[idx::3, :]
        sub_fmri_data[:, ROI_mapping[0] : ROI_mapping[1]] = ROI_data

    return sub_fmri_data, fmri_ROI_mapping


def get_fmri_data(data_type="tr",ROIs=settings.ROIs, len=settings.train_data_len,subject_id="sub01"):
    sub_fmri_data, fmri_ROI_mapping = get_fmri_data_for_sub(data_type, ROIs, len,subject_id)
    fmri_data = {"mapping": fmri_ROI_mapping, "data": sub_fmri_data}

    return fmri_data


def get_activations(compression, activations_dir, video_clip_num, layer_num):
    """ Loads NN features into a np array according to layer. """
    # Use transformed features
    if compression:
        train_file = os.path.join(activations_dir, "train_" + video_clip_num + "_{}.npy".format(layer_num))
        test_file = os.path.join(activations_dir, "test_" + video_clip_num + "_{}.npy".format(layer_num))
        train_activations = np.load(train_file)
        test_activations = np.load(test_file)
    # Use raw activations (e.g. together with PLS)
    else:
        activations_file_list = glob.glob(activations_dir + "/" + video_clip_num + '/*'+"_{}.npy".format(layer_num))
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0]).shape[0]
        x = np.zeros((len(activations_file_list), feature_dim))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        train_activations = x[:1200]
        test_activations = x[1200:]
        scaler = StandardScaler()
        train_activations = scaler.fit_transform(train_activations)
        test_activations = scaler.fit_transform(test_activations)
    return train_activations, test_activations

def get_data(subject_id='sub01', ROIs=settings.ROIs):
    vid_files = sorted(os.listdir(settings.VID_FOLDER))
    train_vid_files = vid_files[: settings.train_data_len]
    test_vid_files = vid_files[settings.train_data_len :]
    fmri_data_train = get_fmri_data("tr", ROIs, settings.train_data_len,subject_id)
    fmri_data_test = get_fmri_data("ts", ROIs, settings.test_data_len,subject_id)
    return fmri_data_train, fmri_data_test, train_vid_files, test_vid_files


def get_encoding_data(activations_dir='E:/LLM/Algonauts_2023/compress_features/UniFormerV2/marlin_vit_large_ytf_1024/pca_100',
                      subject_id='sub01', layer_num="00", ROIs=settings.ROIs):

    # Loop over layers and stack features together

    train_activations, test_activations = get_activations(settings.COMPRESSION, activations_dir,subject_id,layer_num)

    fmri_data_train = get_fmri_data("tr", ROIs, settings.train_data_len,subject_id)
    fmri_data_test = get_fmri_data("ts", ROIs, settings.test_data_len,subject_id)
    return train_activations, test_activations, fmri_data_train, fmri_data_test


def get_activations_data(activations_dir='E:/LLM/Algonauts_2023/compress_features/Marlin/marlin_vit_large_ytf_1024/mean/pca_100',
                      subject_id='sub01', layer_num="00"):

    # Loop over layers and stack features together
    train_activations, test_activations = get_activations(settings.COMPRESSION, activations_dir,subject_id,layer_num)
    return train_activations, test_activations


def get_fmris_data(subject_id='sub01',  ROIs=settings.ROIs):

    fmri_data_train = get_fmri_data("tr", ROIs, settings.train_data_len, subject_id)
    fmri_data_test = get_fmri_data("ts", ROIs, settings.test_data_len, subject_id)
    return fmri_data_train, fmri_data_test
