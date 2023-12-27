import skfuzzy as fuzz
from util import *


def CmeansCluster(features, labels, n_clusters, maxiter):
    # 尝试不同的模糊因子 (fuzziness parameter, m)
    # 通常取值范围在 [1.5, 3]
    best_accuracy = 0
    best_cntr = None
    V = np.zeros((n_clusters, features.shape[1]))
    best_fuzziness = 0
    iteration = 0
    cluster_membership = np.zeros_like(labels)
    error_rate = 0
    mapped_labels = np.zeros_like(cluster_membership)

    for m in np.arange(1.5, 3, 0.005):
        # 模糊 C-均值算法
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features.T,
                                                         n_clusters,
                                                         m,
                                                         error=1e-10,
                                                         maxiter=maxiter,
                                                         seed=42)

        # 分配每个数据点到最高隶属值的聚类
        cluster_membership = np.argmax(u, axis=0)

        # # 创建一个标签映射
        # mapped_labels = np.zeros_like(cluster_membership)
        # for i in range(n_clusters):
        #     mask = (cluster_membership == i)
        #     mapped_labels[mask] = mode(labels[mask])[0]
        mapped_labels = mapped(labels, cluster_membership, n_clusters)

        # 计算准确率
        accuracy = np.sum(mapped_labels == labels) / len(labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            error_rate = 1 - best_accuracy
            best_cntr = cntr
            best_fuzziness = m
            iteration = p
            V = np.dot(features.T, u0.T ** m) / np.sum(
                u0.T ** m, axis=0, keepdims=True)
            mapped_labels = mapped_labels
    # 输出最佳准确率和对应的聚类中心
    print(f"最佳模糊因子：{best_fuzziness}")
    print(f"迭代次数：{iteration}")
    print(f"误差率：{(error_rate * 100):.2f}%")

    for i in range(n_clusters):
        initial_center = V.T[i]
        final_center = best_cntr[i]
        class_count = np.sum(mapped_labels == i)
        sample_count = np.sum(cluster_membership == i)

        print(f"类别 {i + 1}：")
        print(f"初始聚类中心：{initial_center}")
        print(f"最终聚类中心：{final_center}")
        print(f"分类个数：{class_count}")
        print(f"样本数：{sample_count}")
    return mapped_labels
