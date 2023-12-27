import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from util import *


def Spectral_Clustering(features, labels, n_clusters):
    # Building the clustering model
    spectral_model_rbf = SpectralClustering(n_clusters=n_clusters, affinity='rbf')

    # Training the model and Storing the predicted cluster labels
    labels_rbf = spectral_model_rbf.fit_predict(features)
    mapped_labels_rbf = mapped(labels, labels_rbf, n_clusters)

    scores_spectral(labels, mapped_labels_rbf)
    spectral_model_nn = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

    labels_nn = spectral_model_nn.fit_predict(features)
    mapped_labels_nn = mapped(labels, labels_nn, n_clusters)
    scores_spectral(labels, mapped_labels_nn)
    affinity = ['rbf', 'nearest-neighbours']
    # List of Silhouette Scores
    s_scores = [silhouette_score(features, labels_rbf), silhouette_score(features, labels_nn)]
    # Evaluating the performance
    print(s_scores)
    plt.bar(affinity, s_scores)
    plt.xlabel('Affinity')
    plt.ylabel('Silhouette Score')
    plt.title('Comparison of different Clustering Models')
    plt.savefig("./fig/SilhouetteScore.png")
    plt.show()
    return mapped_labels_rbf, mapped_labels_nn
