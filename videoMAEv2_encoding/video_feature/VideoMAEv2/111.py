import os

import numpy as np

a=np.load("E:/情绪分类论文/zuixin_yuyi/TYPED_FITHRF_GLMDENOISE_RR.npy")
# 指定要创建文件夹的目录
base_dir = 'F:/LLM/piceng/'  # 你可以将这里的路径改为你需要的目录

# 生成文件夹名称列表
folder_names = [f"sub{i:02d}" for i in range(1, 66)]

# 创建文件夹
for folder_name in folder_names:
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建文件夹: {folder_path}")
    else:
        print(f"文件夹已存在: {folder_path}")