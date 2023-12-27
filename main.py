from Cmeans import *
from Spectral import *

if __name__ == "__main__":
    features, labels = loader("iris.dat")
    mapped_labels_fcm = CmeansCluster(features, labels, 3, 100)
    df = reduce_features(features, labels, mapped_labels_fcm)
    visualize(df, "./fig/fcm")
    mapped_labels_rbf, mapped_labels_nn = Spectral_Clustering(features, labels, 3)
    df2 = reduce_features(features, labels, mapped_labels_rbf)
    visualize(df2, "./fig/rbf")
    df3 = reduce_features(features, labels, mapped_labels_nn)
    visualize(df3, "./fig/nn")
