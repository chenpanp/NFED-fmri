# import pandas as pd
#
# # 基础文件路径和文件名
# base_path = r'F:/NFED/FD/sub-05/'
# input_pattern = 'sub-05_task-video_run-{}_FD.xlsx'
# output_pattern = 'sub-05_task-video_run-{:02d}_FD.tsv'
#
# # 处理从 run-1 到 run-68 的文件
# for run_number in range(1, 69):
#     # 构造输入和输出文件路径
#     xlsx_file = base_path + input_pattern.format(run_number)
#     tsv_file = base_path + output_pattern.format(run_number)
#
#     # 读取 .xlsx 文件
#     try:
#         df = pd.read_excel(xlsx_file, engine='openpyxl')
#
#         # 保存为 .tsv 文件
#         df.to_csv(tsv_file, sep='\t', index=False)
#         print(f'文件已成功转换为 {tsv_file}')
#     except FileNotFoundError:
#         print(f'文件 {xlsx_file} 未找到，跳过。')
#     except Exception as e:
#         print(f'处理文件 {xlsx_file} 时出错: {e}')

# import numpy as np
# from sklearn.decomposition import PCA
#
# # 假设你的特征数据已经保存在一个.npy文件中
# features_path = "E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048/sub02/0001_00.npy"
# features = np.load(features_path)
#
#
# pca = PCA(n_components=60)
#
# pca.fit(features)
#
#
# explained_variance_ratio = pca.explained_variance_ratio_
#
# print("Explained variance ratio for each of the 60 components:")
# print(explained_variance_ratio)
#
#
# total_explained_variance = explained_variance_ratio.sum()
# print("Total explained variance ratio (for the 60 components):", total_explained_variance)


# import numpy as np
# from sklearn.decomposition import PCA
#
# # 加载特征数据
# features_path = "E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048/sub02/0001_00.npy"
# features = np.load(features_path)
#
# # 打印特征数据的形状
# print("Features shape:", features.shape)
#
# # 确保 PCA 的 n_components 在合适的范围内
# n_components = min(features.shape[0], features.shape[1], 100)  # 选择合适的主成分数
# print(f"Using {n_components} components for PCA")
#
# # 执行 PCA
# pca = PCA(n_components=n_components)
# pca.fit(features)
#
# # 计算解释方差比率
# explained_variance_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio for each component:")
# print(explained_variance_ratio)
#
# # 计算总解释方差比率
# total_explained_variance = explained_variance_ratio.sum()
# print("Total explained variance ratio:", total_explained_variance)
# import numpy as np
# from sklearn.decomposition import PCA
#
# X_trn=np.load("E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048/sub02/0001_00.npy")
# X_tes=np.load("E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048/1200_00.npy")
# X_con = np.concatenate((X_trn, X_tes), axis=0)
# pca = PCA(n_components=60)
# pca.fit(X_con)   # 进行降维，降到1000维
# newX = pca.fit_transform(X_con)
#
#
#
# import numpy as np
# from sklearn.decomposition import PCA
#
# # 加载特征数据
# features_path = "E:/LLM/Algonauts_2023/features/videoMAEV2/vit_g_hybrid_pt_1200e_k710_ft_1048/sub02/0001_00.npy"
# features = np.load(features_path)
#
# # 打印特征数据的形状
# print("Features shape:", features.shape)
#
# # 检查样本数和特征数
# n_samples, n_features = features.shape
#
# # 如果样本数少于 2，则 PCA 无法执行
# if n_samples < 2:
#     raise ValueError("PCA requires at least 2 samples. Your data has only one sample.")
#
# # 使用尽可能多的主成分
# n_components = min(n_samples, n_features)
# print(f"Using {n_components} components for PCA")
#
# # 执行 PCA
# pca = PCA(n_components=n_components)
# pca.fit(features)
#
# # 计算解释方差比率
# explained_variance_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio for each component:")
# print(explained_variance_ratio)
#
# # 计算总解释方差比率
# total_explained_variance = explained_variance_ratio.sum()
# print("Total explained variance ratio:", total_explained_variance)


# import numpy as np
# from sklearn.decomposition import PCA
#
# # 假设 features 是一个字典，包含各层的特征图数据
# a=np.load ("E:/subdata_5/TANZHAODENG/data/action_feature.npy")# 例如：100 个样本，60 个特征
#
# sssd=a

from PIL import Image


def resize_image(input_image_path, output_image_path, new_width, new_height):
    # 打开原始图像
    with Image.open(input_image_path) as img:
        # 调整图像尺寸
        resized_img = img.resize((new_width, new_height))

        # 保存调整后的图像
        resized_img.save(output_image_path)


# 定义输入和输出路径
input_image_path = 'E:\\videodecode\\i2vgen-xl\\data\\Algonauts_2023\images\\0006.jpg'  # 替换为您的输入图片路径
output_image_path = 'E:\\videodecode\\i2vgen-xl\\data\\Algonauts_2023\\output_image_320x320_6.jpg'  # 替换为您想要的输出图片路径

# 定义新的尺寸
new_width = 320
new_height = 320

# 调用函数调整图片尺寸
resize_image(input_image_path, output_image_path, new_width, new_height)

print(f"图像尺寸已调整为 {new_width}x{new_height} 并保存到 {output_image_path}")