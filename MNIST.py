from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import struct
import itertools

wcss = []
homogeneity_score = []
completeness_score = []
v_measure_score = []
adjusted_rand_score = []
adjusted_mutual_info_score = []
silhouette_score = []

def initStats():
  wcss = []
  homogeneity_score = []
  completeness_score = []
  v_measure_score = []
  adjusted_rand_score = []
  adjusted_mutual_info_score = []
  silhouette_score = []

def plotGraph(min_range, max_range, data, title, xlab, ylab, filename):
  #Plotting the results onto a line graph, allowing us to observe 'The elbow'
  plt.plot(range(min_range, max_range), data)

  plt.title(title)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.savefig(filename)
  print(filename + ' plot saved!')
  plt.close()

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(n=1000):

    raw_train = read_idx("train-images-idx3-ubyte")
    train_data = np.reshape(raw_train, (60000, 28*28))
    train_label = read_idx("train-labels-idx1-ubyte")

    raw_test = read_idx("t10k-images-idx3-ubyte")
    test_data = np.reshape(raw_test, (10000, 28*28))
    test_label = read_idx("t10k-labels-idx1-ubyte")

    X = train_data
    Y = train_label

    random_state = check_random_state(0)
    permutation = random_state.permutation(test_data.shape[0])[0:n]
    X = X[permutation,:]
    Y = Y[permutation]
    return X, Y, test_data, test_label

#train, labels, test, labels_test = load_mnist()
data, labels, data_test, labels_test = load_mnist()

np.random.seed(42)
#(0) Prepare
#digits = load_digits()
#data = scale(digits.data)
#data = train.iloc[:,0:64]
#data = data.values
n_samples, n_features = data.shape

#labels = train['salary']# digits.target
#data_test = test.iloc[:,0:64]
#data_test = data_test.values
#labels_test = test['salary']
#n_digits = len(np.unique(digits.target))
n_digits = len(np.unique(labels))
'''
data = train.iloc[:,0:38]
data = data.values
n_samples, n_features = data.shape
labels = train['Consumption_additive']
data_test = test.iloc[:,0:38]
data_test = data_test.values
labels_test = test['Consumption_additive']
n_digits = len(np.unique(labels))
'''
sample_size = 50

#(1) Clustering
print("Part 1: Clustering")
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette\tAccuracy')

kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
float(sum(kmeans.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,kmeans.labels_)
metrics.completeness_score(labels, kmeans.labels_)

EMax = GaussianMixture(n_components=20,random_state=0).fit(data)
# EMax = GMM(n_components=2,random_state=0).fit(data)
EMax.labels_ = EMax.predict(data)
float(sum(EMax.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,EMax.labels_)
metrics.completeness_score(labels, EMax.labels_)

#def bench_k_means(estimator, name, data):
#    t0 = time()
#    estimator.fit(data)
#    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
#          % (name, (time() - t0), estimator.inertia_,
#             metrics.homogeneity_score(labels, estimator.labels_),
#             metrics.completeness_score(labels, estimator.labels_),
#             metrics.v_measure_score(labels, estimator.labels_),
#             metrics.adjusted_rand_score(labels, estimator.labels_),
#             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#             metrics.silhouette_score(data, estimator.labels_,
#                                      metric='euclidean',
#                                      sample_size=sample_size)))

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, estimator.predict(data)),
             metrics.completeness_score(labels, estimator.predict(data)),
             metrics.v_measure_score(labels, estimator.predict(data)),
             metrics.adjusted_rand_score(labels, estimator.predict(data)),
             metrics.adjusted_mutual_info_score(labels,  estimator.predict(data)),
             metrics.silhouette_score(data, estimator.predict(data),metric='euclidean',sample_size=sample_size),
             float(sum(estimator.predict(data) == labels))/float(len(labels))))
n_digits_i = [2,5,10,20,50]
print("Part 1: K-Means on MNIST")
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=data)
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
print("Part 1: EM on MNIST")
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=data)


# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
#pca = PCA(n_components=n_digits).fit(data)
#bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#              name="PCA-based",
#              data=data)
#print(79 * '_')

print("Part 2: Dimensionality Reduction PCA")
#(2) apply dimension reduction algorithms
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
PCA_data = PCA(n_components = 5,whiten=False)
temp = PCA_data.fit(data)
#temp1= temp.components_
PCA_data_trans = PCA_data.transform(data)
PCA_data_trans_test = PCA_data.transform(data_test)
#PCA_comp = PCA_data.components_

print("Part 2: Dimensionality Reduction ICA")
ICA_data = FastICA(n_components = 5)
ICA_data.fit(data)
ICA_data_trans = ICA_data.transform(data)
ICA_data_trans_test = ICA_data.transform(data_test)

print("Part 2: Dimensionality Reduction RP")
from sklearn.random_projection import GaussianRandomProjection
transformer = GaussianRandomProjection(n_components=5,eps=0.1)
RP_data_trans = transformer.fit_transform(data)
RP_data_trans_test = transformer.fit_transform(data_test)

print("Part 2: Dimensionality Reduction LDA")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
transformer = LinearDiscriminantAnalysis(solver="svd",n_components = 5)
LDA_data_trans = transformer.fit_transform(data, labels)
LDA_data_trans_test = transformer.fit_transform(data_test, labels_test)

#(3)
print("Part 3: PCA")
#PCA----------------------------------
kmeans_PCA = KMeans(n_clusters=2, random_state=0).fit(PCA_data_trans)
float(sum(kmeans_PCA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,kmeans_PCA.labels_)
EMax_PCA = GaussianMixture(n_components=2,random_state=0).fit(PCA_data_trans)
EMax_PCA.labels_ = EMax_PCA.predict(PCA_data_trans)
float(sum(EMax_PCA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,EMax_PCA.labels_)

n_digits_i = [2,5,10,20,50]
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=PCA_data_trans)
print("Part 3: KMeans finished")
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=PCA_data_trans)
print("Part 3: GaussianMixture finished")

#ICA----------------------------------
kmeans_ICA = KMeans(n_clusters=2, random_state=0).fit(ICA_data_trans)
float(sum(kmeans_ICA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,kmeans_ICA.labels_)
EMax_ICA = GaussianMixture(n_components=2,random_state=0).fit(ICA_data_trans)
EMax_ICA.labels_ = EMax_ICA.predict(ICA_data_trans)
float(sum(EMax_ICA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,EMax_ICA.labels_)
metrics.completeness_score(labels,EMax_ICA.labels_)
metrics.v_measure_score(labels,EMax_ICA.labels_)
metrics.adjusted_rand_score(labels,EMax_ICA.labels_)
metrics.adjusted_mutual_info_score(labels,EMax_ICA.labels_)
metrics.silhouette_score(ICA_data_trans, EMax_ICA.labels_,metric='euclidean',sample_size=sample_size)
print("Part 3: ICA loaded")

n_digits_i = [2,5,10,20,50]
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=ICA_data_trans)
print("Part 3: ICA Kmeans Finished")
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=ICA_data_trans)
print("Part 3: ICA GaussianMixture Finished")

#RP----------------------------------
kmeans_RP = KMeans(n_clusters=2, random_state=0).fit(RP_data_trans)
float(sum(kmeans_RP.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,kmeans_RP.labels_)
EMax_RP = GaussianMixture(n_components=2,random_state=0).fit(RP_data_trans)
EMax_RP.labels_ = EMax_RP.predict(RP_data_trans)
float(sum(EMax_RP.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,EMax_RP.labels_)
print("Part 3: RP Loaded")


n_digits_i = [2,5,10,20,50]
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=RP_data_trans)
print("Part 3: RP KMeans Finished")
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=RP_data_trans)
print("Part 3: RP GaussianMixture Finished")

#LDA----------------------------------
kmeans_LDA = KMeans(n_clusters=2, random_state=0).fit(LDA_data_trans)
float(sum(kmeans_LDA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,kmeans_LDA.labels_)
EMax_LDA = GaussianMixture(n_components=2,random_state=0).fit(LDA_data_trans)
EMax_LDA.labels_ = EMax_LDA.predict(LDA_data_trans)
float(sum(EMax_LDA.labels_ == labels))/float(len(labels))
metrics.homogeneity_score(labels,EMax_LDA.labels_)
metrics.completeness_score(labels,EMax_LDA.labels_)
metrics.v_measure_score(labels,EMax_LDA.labels_)
metrics.adjusted_rand_score(labels,EMax_LDA.labels_)
metrics.adjusted_mutual_info_score(labels,EMax_LDA.labels_)
metrics.silhouette_score(LDA_data_trans, EMax_LDA.labels_,metric='euclidean',sample_size=sample_size)
print("Part 3: LDA loaded")

#n_digits_i = [2,5,10,20,50]
for i in range(len(n_digits_i)):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=LDA_data_trans)
print("Part 3: LDA Kmeans Finished")
# bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),name="random", data=data)
for i in range(len(n_digits_i)):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=LDA_data_trans)
print("Part 3: LDA GaussianMixture Finished")


#(4)
t0 = time()
from sklearn.neural_network import MLPClassifier
data_nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
data_nn.fit(data, labels)  
data_train_pred = data_nn.predict(data)
float(sum(data_train_pred == labels))/float(len(labels))
data_test_pred = data_nn.predict(data_test) # data_test = x_test
float(sum(data_test_pred == labels_test))/float(len(labels_test))
regular_time = (time() - t0)
print("Part 4(1) complete")

t0 = time()
data_nn.fit(PCA_data_trans, labels)  
data_train_pred_PCA = data_nn.predict(PCA_data_trans)
float(sum(data_train_pred_PCA == labels))/float(len(labels))
data_test_pred_PCA = data_nn.predict(PCA_data_trans_test)
float(sum(data_test_pred_PCA == labels_test))/float(len(labels_test))
pca_time = (time() - t0)
print("Part 4(2) complete")

t0 = time()
data_nn.fit(ICA_data_trans, labels)  
data_train_pred_ICA = data_nn.predict(ICA_data_trans)
float(sum(data_train_pred_ICA == labels))/float(len(labels))
data_test_pred_ICA = data_nn.predict(ICA_data_trans_test)
float(sum(data_test_pred_ICA == labels_test))/float(len(labels_test))
ica_time = (time() - t0)
print("Part 4(3) complete")

t0 = time()
data_nn.fit(RP_data_trans, labels)  
data_train_pred_RP = data_nn.predict(RP_data_trans)
float(sum(data_train_pred_RP == labels))/float(len(labels))
data_test_pred_RP = data_nn.predict(RP_data_trans_test)
float(sum(data_test_pred_RP == labels_test))/float(len(labels_test))
rp_time = (time() - t0)
print("Part 4(4) complete")

print("Algorithm\tTraining Accuracy\tTesting Accuracy\tTime Taken")
print("Original\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred, labels), metrics.accuracy_score(data_test_pred, labels_test), regular_time))
print("PCA\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_PCA, labels), metrics.accuracy_score(data_test_pred_PCA, labels_test), pca_time))
print("ICA\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_ICA, labels), metrics.accuracy_score(data_test_pred_ICA, labels_test), ica_time))
print("RP\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_RP, labels), metrics.accuracy_score(data_test_pred_RP, labels_test), rp_time))
print("******************************************")
print("Run\tWithout Transformation\tPCA\tICA\tRP")
print("Training\t{}\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred, labels), metrics.accuracy_score(data_train_pred_PCA, labels), metrics.accuracy_score(data_train_pred_ICA, labels), metrics.accuracy_score(data_train_pred_RP, labels)))
print("Testing\t{}\t{}\t{}\t{}".format(metrics.accuracy_score(data_test_pred, labels_test), metrics.accuracy_score(data_test_pred_PCA, labels_test), metrics.accuracy_score(data_test_pred_ICA, labels_test), metrics.accuracy_score(data_test_pred_RP, labels_test)))

#(5)
t0 = time()
data_new = np.hstack((data,PCA_data_trans))
data_test_new = np.hstack((data_test,PCA_data_trans_test))
data_nn.fit(data_new, labels)  
data_train_pred_PCA_new = data_nn.predict(data_new)
float(sum(data_train_pred_PCA_new == labels))/float(len(labels))
data_test_pred_PCA_new = data_nn.predict(data_test_new)
float(sum(data_test_pred_PCA_new == labels_test))/float(len(labels_test))
pca_time = (time() - t0)
print("Part 5(1) complete")

t0 = time()
data_ICA_new = np.hstack((data,ICA_data_trans))
data_ICA_test_new = np.hstack((data_test,ICA_data_trans_test))
data_nn.fit(data_ICA_new, labels)  
data_train_pred_ICA_new = data_nn.predict(data_ICA_new)
float(sum(data_train_pred_ICA_new == labels))/float(len(labels))
data_test_pred_ICA_new = data_nn.predict(data_ICA_test_new)
float(sum(data_test_pred_ICA_new == labels_test))/float(len(labels_test))
ica_time = (time() - t0)
print("Part 5(2) complete")

t0 = time()
data_RP_new = np.hstack((data,RP_data_trans))
data_RP_test_new = np.hstack((data_test,RP_data_trans_test))
data_nn.fit(data_RP_new, labels)  
data_train_pred_RP_new = data_nn.predict(data_RP_new)
float(sum(data_train_pred_RP_new == labels))/float(len(labels))
data_test_pred_RP_new = data_nn.predict(data_RP_test_new)
float(sum(data_test_pred_RP_new == labels_test))/float(len(labels_test))
rp_time = (time() - t0)
print("Part 5(3) complete")

print("Algorithm\tTraining Accuracy\tTesting Accuracy\tTime Taken")
print("PCA\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_PCA_new, labels), metrics.accuracy_score(data_test_pred_PCA_new, labels_test), pca_time))
print("ICA\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_ICA_new, labels), metrics.accuracy_score(data_test_pred_ICA_new, labels_test), ica_time))
print("RP\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred_RP_new, labels), metrics.accuracy_score(data_test_pred_RP_new, labels_test), rp_time))
print("******************************************")
print("Run\tWithout Transformation\tPCA\tICA\tRP")
print("Training\t{}\t{}\t{}\t{}".format(metrics.accuracy_score(data_train_pred, labels), metrics.accuracy_score(data_train_pred_PCA_new, labels), metrics.accuracy_score(data_train_pred_ICA_new, labels), metrics.accuracy_score(data_train_pred_RP_new, labels)))
print("Testing\t{}\t{}\t{}\t{}".format(metrics.accuracy_score(data_test_pred, labels_test), metrics.accuracy_score(data_test_pred_PCA_new, labels_test), metrics.accuracy_score(data_test_pred_ICA_new, labels_test), metrics.accuracy_score(data_test_pred_RP_new, labels_test)))
