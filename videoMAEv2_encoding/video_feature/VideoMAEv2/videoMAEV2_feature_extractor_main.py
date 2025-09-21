from feature_extractor import FeatureExtractor
# from feature_extractor import FeatureExtractor
from models import vit_giant_patch14_224
import torch
import torch.nn as nn
import os
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
ckpt_path = "E:/LLM/video_feature/VideoMAEv2/checkpoints/vit_g_hybrid_pt_1200e_k710_ft.pth"

# get model & load ckpt
model = vit_giant_patch14_224(
    img_size=224,
    pretrained=False,
    num_classes=710,
    all_frames=16,
    tubelet_size=2,
    drop_path_rate=0.3,
    use_mean_pooling=True)
feature_extractor = FeatureExtractor(model, ckpt_path, device, 16)

video_paths_dir = "E:/LLM/Algonauts_2023/video"
video_paths = os.listdir(video_paths_dir)
output_features_dir = "E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048"

if not os.path.exists(output_features_dir):
    os.makedirs(output_features_dir)

# Extract features from a list of videos
for video_file_name in video_paths:
    # if video_file_name not in ["735.mp4","1315.mp4"]:
    #     continue

    video_path = os.path.join(video_paths_dir, video_file_name)
    features = feature_extractor.extract_video(video_path,  sample_rate=2)

    for i, sub_feats in enumerate(features):
        features_dir = os.path.join(output_features_dir, "sub0"+str(i+1))
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        # layer feature
        sub_feats = list(sub_feats)
        for j, feature in enumerate(sub_feats):
            save_video_file_name = video_file_name.split(".")[0]
            if len(save_video_file_name) < 4:
                save_video_file_name = save_video_file_name.zfill(4)

            save_video_file_name = save_video_file_name + "_" + str(j).zfill(2) + ".npy"
            output_features_path = os.path.join(features_dir, save_video_file_name)
            print(output_features_path)
            # print(features[i].shape)
            # tmp = features[i].cpu().numpy()
            np.save(output_features_path, feature.cpu().numpy())

        # tmp_video_file_name = video_file_name.split(".")[0] + "_" + str(i) + ".npy"
        # filename = os.path.splitext(tmp_video_file_name)[0].split("_")[0]
        # index = os.path.splitext(tmp_video_file_name)[0].split("_")[1]
        # filetype = os.path.splitext(tmp_video_file_name)[1]
        # if len(filename) < 4:
        #     # Newdir = os.path.join(output_path, filename.zfill(4) + "_" + index + filetype)
        #     tmp_video_file_name = filename.zfill(4) + "_" + index + filetype

