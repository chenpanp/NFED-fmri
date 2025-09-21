import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import seaborn as sns

def vectorized_correlation(x, y, reduction="mean"):
    dim = 0
    x = np.array(x)
    y = np.array(y)

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    # if reduction is None:
    #     return corr.ravel()
    # elif reduction == "mean":
    #     return corr.ravel().mean()
    # else:
    #     raise Exception("Unknown reduction")
    return corr
def RDM_correlation(merged_array, save_path):
    # 计算特征向量之间的相似性矩阵（RDM）
    similarity_matrix = squareform(pdist(merged_array, metric='correlation'))
    # 归一化相似性矩阵到[0, 1]
    normalized_similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
            similarity_matrix.max() - similarity_matrix.min())
    # 绘制热图
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(similarity_matrix, cmap='coolwarm', square=True)
    plt.imshow(normalized_similarity_matrix, cmap='RdBu', interpolation='nearest')
    plt.colorbar()

    # 显示图像
    # plt.show()
    plt.savefig(save_path)
    plt.close()
    return normalized_similarity_matrix


def evaluation_metrics(y, y_pred):
    corr = vectorized_correlation(y, y_pred)
    mse = np.mean((y-y_pred)**2)
    mae = np.mean(np.abs(y-y_pred))
    return corr, mse, mae