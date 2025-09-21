import os
import glob
from tqdm import tqdm
import numpy as np
from .fit_pca import fit_trafo_pca
from .fit_umap import fit_trafo_umap
from .fit_autoencoder import fit_trafo_autoencoder

# from mle_toolbox.utils import save_pkl_object
from sklearn.preprocessing import StandardScaler


def do_dim_reduction_and_save(activations_dir, save_dir,
                              trafo_type, dim_red_params):
    video_clip = ['sub0' + str(l) for l in range(1,4)]  # 视频分成三个部分
    # video_clip = ['sub0' + str(l) for l in range(1, 12)]#video daluan
    # video_clip = ['sub0' + str(l) for l in range(1, 5)]  # video daluan
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # explained_variance = []
    # layers_nums = [str(i).zfill(2) for i in range(0, 1)]#12 12ceng
    layers_nums = [str(i).zfill(2) for i in range(0, 4)]  # 12 12ceng
    all_layers_info = {}
    for num_clip in tqdm(video_clip):
        for layer_num in layers_nums:
            # Load activations
            # tmp = activations_dir + "/" + num_clip + '/*'+'_{}.npy'.format(layer_num)
            activations_file_list = glob.glob(activations_dir + "/" + num_clip + '/*'+'_{}.npy'.format(layer_num))

            activations_file_list.sort()
            feature_dim = np.load(activations_file_list[0]).shape[1]
            x = np.zeros((len(activations_file_list),feature_dim))
            for i, activation_file in enumerate(activations_file_list):
                temp = np.load(activation_file)
                x[i,:] = temp
            x_train = x[:1230,:]
            x_test = x[1230:]
            #print(x.shape, x_train.shape, x_test.shape)S
            x_test = StandardScaler().fit_transform(x_test)
            x_train = StandardScaler().fit_transform(x_train)

            # Plug&Play - Fitting of dimensionality reduction technique
            if trafo_type == "pca":
                x_train_trafo, x_test_trafo, info_l = fit_trafo_pca(x_train, x_test,
                                                                    dim_red_params)
            elif trafo_type == "umap":
                x_train_trafo, x_test_trafo, info_l = fit_trafo_umap(x_train, x_test,
                                                                     dim_red_params)
            elif trafo_type == "autoencoder":
                x_train_trafo, x_test_trafo, info_l = fit_trafo_autoencoder(
                                                                   x_train, x_test,
                                                                   dim_red_params)
            #print(x_train.shape, x_train_trafo.shape)
            train_save_path = os.path.join(save_dir, "train_" + num_clip + "_" + layer_num)
            test_save_path = os.path.join(save_dir, "test_" + num_clip + "_" + layer_num)
            np.save(train_save_path, x_train_trafo)
            np.save(test_save_path, x_test_trafo)

            all_layers_info[num_clip] = info_l
    # info_path = os.path.join(save_dir, info_title)
    # save_pkl_object(all_layers_info, info_path)
    print(all_layers_info)


def run_compression(feature_dir, save_dir, model_type, trafo_type, num_components):
    activations_dir = feature_dir
    # preprocessing using PCA and save
    dim_red_dir = os.path.join(save_dir, f'{trafo_type}_{num_components}')
    print(dim_red_dir)
    print(f"------performing  {trafo_type}: {num_components}---------")
    dim_red_params = {"n_components": num_components}
    do_dim_reduction_and_save(activations_dir,
                              dim_red_dir,
                              trafo_type,
                              dim_red_params)

