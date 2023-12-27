import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score


def loader(file_path):
    with open(file_path, 'r') as f:
        data = np.loadtxt(f)  # 读取数据文件
        # 提取特征和标签
        features = data[:, :-1]  # 特征
        labels: object = data[:, -1].astype(int)  # 标签，假设标签是整数类型
        return features, labels


def mapped(labels, cluster_membership, n_clusters):
    mapped_labels = np.zeros_like(cluster_membership)
    for i in range(n_clusters):
        mask = (cluster_membership == i)
        mapped_labels[mask] = mode(labels[mask])[0]
    return mapped_labels


def reduce_features(features, labels, mapped_labels):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(features)
    df_reduced = pd.DataFrame(
        X_reduced, columns=['Principal Component 1', 'Principal Component 2'])
    df_reduced['Original Label'] = labels
    df_reduced['Predicted Label'] = mapped_labels
    return df_reduced


def visualize(df_reduced, save_path):
    # 创建一个散点图，使用不同颜色表示原始标签和预测标签
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_reduced,
                    x='Principal Component 1',
                    y='Principal Component 2',
                    hue='Original Label',
                    palette='coolwarm',
                    marker='o',
                    label='Original Label',
                    s=100)
    sns.scatterplot(data=df_reduced,
                    x='Principal Component 1',
                    y='Principal Component 2',
                    hue='Predicted Label',
                    palette='dark',
                    marker='*',
                    label='Predicted Label',
                    s=100)
    inconsistent_points = df_reduced[
        df_reduced['Original Label'] != df_reduced['Predicted Label']]
    sns.scatterplot(data=inconsistent_points,
                    x='Principal Component 1',
                    y='Principal Component 2',
                    hue='Predicted Label',
                    palette='plasma',
                    marker='s',
                    s=100,
                    label='Inconsistent')
    # 添加图例
    plt.legend(title='Label Type')

    # 设置坐标轴标签和标题
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('C-means Clustering Results (PCA)')

    # 显示图
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def scores_spectral(true_labels, cluster_labels):
    # Assume `true_labels` contains the true labels and `cluster_labels` contains the labels from spectral clustering
    # Compute the metrics
    conf_matrix = confusion_matrix(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    v_measure = v_measure_score(true_labels, cluster_labels)

    # Print the results
    print("Confusion Matrix:\n", conf_matrix)
    print("Adjusted Rand Index:", ari)
    print("Adjusted Mutual Information:", ami)
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
