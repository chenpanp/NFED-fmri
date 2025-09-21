import os.path

from src import settings
import src.data.handler as data_handler

import torch
import itertools
import numpy as np


def sigmoid(z):
    return (1 / (np.exp(-z))) - 1


def calc_weigths(fmri_data, min_weight=0.0, max_weight=2.0):
    weight = sigmoid(np.abs(fmri_data))  # calc 0 based sigmoid
    weight = np.clip(weight, min_weight, max_weight)  # clip values to desired range
    return weight


class VidFMRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vid_files,
        fmri_data,
        fmri_transform=None,
    ):

        # self.vid_files = vid_files
        # self.fmri_data = fmri_data
        # prepare data
        self.prepare_data(vid_files, fmri_data, fmri_transform)
        # weights based on voxel activity
        # self.weight = calc_weigths(self.fmri_data)
        # input and target transformations
        self.fmri_transform = fmri_transform
    def prepare_data(self, vid_files, fmri_data, fmri_transform):
        self.vid_files = vid_files
        self.fmri_data = fmri_data

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        vid_fn = self.vid_files[idx]
        vide_feature= np.load(os.path.join(settings.VID_FOLDER, vid_fn))



        return {
            "vid_data": vide_feature.astype(np.float32),
            "fmri": fmri.astype(np.float32),
        }

