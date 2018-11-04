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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GaussianMixture as GMM
import itertools
from sklearn import mixture


wcss = []
homogeneity_score = []
completeness_score = []
v_measure_score = []
adjusted_rand_score = []
adjusted_mutual_info_score = []
silhouette_score = []

components_min = 1
components_max = 30

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

def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov':
        return 'govt'
    elif x['workclass'] == ' Private':
        return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc':
        return 'self_employed'
    else:
        return 'without_pay'

def loadAndProcessData(filename):
    #importing the adult dataset with pandas

    df = pd.read_csv(filename,1,",")
    data = [df]
    # Convert salary to integer
    salary_map = {' <=50K':1,' >50K':0}
    df['salary']=df['salary'].map(salary_map).astype(int)
    # convert sex into integer
    df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)

    # Drop empty value marked as '?'
    df['country'] = df['country'].replace(' ?',np.nan)
    df['workclass'] = df['workclass'].replace(' ?',np.nan)
    df['occupation'] = df['occupation'].replace(' ?',np.nan)

    df.dropna(how='any',inplace=True)

    for dataset in data:
        dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
        dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US' 

    # Convert country in integer
    df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)

    # Categorise marital-status into single and couple
    df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')

    df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

    df['marital-status'] = df['marital-status'].map({'Couple':0,'Single':1})

    rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}

    df['relationship'] = df['relationship'].map(rel_map)

    race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}

    df['race']= df['race'].map(race_map)
       
    df['employment_type']=df.apply(f, axis=1)

    df.loc[(df['capital-gain'] > 0),'capital-gain'] = 1
    df.loc[(df['capital-gain'] == 0 ,'capital-gain')]= 0

    df.loc[(df['capital-loss'] > 0),'capital-loss'] = 1
    df.loc[(df['capital-loss'] == 0 ,'capital-loss')]= 0

    employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

    df['employment_type'] = df['employment_type'].map(employment_map)
    df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)

    return df

def selectEM(X):

  np.random.seed(0)

  lowest_bic = np.infty
  bic = []
  n_components_range = range(components_min, components_max)
  cv_types = ['spherical', 'tied', 'diag', 'full']
  for cv_type in cv_types:
    for n_components in n_components_range:
      # Fit a Gaussian mixture with EM
      use_init = False
      if use_init == True:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, n_init=5)
      else:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
      gmm.fit(X)
      bic.append(gmm.bic(X))
      if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

  bic = np.array(bic)
  color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
  clf = best_gmm
  bars = []

  # Plot the BIC scores
  plt.figure(figsize=(8, 6))
  spl = plt.subplot(1,1,1)
  for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
      (i + 1) * len(n_components_range)],width=.2, color=color))
  plt.xticks(n_components_range)
  plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
  plt.title('BIC score per model')
  xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
  plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
  spl.set_xlabel('Number of components')
  spl.legend([b[0] for b in bars], cv_types)

  plt.savefig('EM_Adult_BIC_ModelSelection.png')
  plt.close()

  print(clf.n_components)
  print(clf.covariance_type)

  return clf

train = loadAndProcessData('adult_training.csv')
test = loadAndProcessData('adult_test.csv')

np.random.seed(42)
#(0) Prepare
#digits = load_digits()
#data = scale(digits.data)
data = train.iloc[:,0:64]
data = data.values
n_samples, n_features = data.shape

labels = train['salary']# digits.target
data_test = test.iloc[:,0:64]
data_test = data_test.values
labels_test = test['salary']
#n_digits = len(np.unique(digits.target))
n_digits = len(np.unique(labels))

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

def plot_estimator(estimator_type, data, part):
  for i in range(components_min, components_max):
      estimator = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
      estimator.fit(data)

      wcss.append(estimator.inertia_)
      homogeneity_score.append(metrics.homogeneity_score(labels, estimator.predict(data)))
      completeness_score.append(metrics.completeness_score(labels, estimator.predict(data)))
      v_measure_score.append(metrics.v_measure_score(labels, estimator.predict(data)))
      adjusted_rand_score.append(metrics.adjusted_rand_score(labels, estimator.predict(data)))
      adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(labels, estimator.predict(data)))
      #silhouette_score.append(metrics.silhouette_score(y, kmeans.labels_, metric='euclidean', sample_size=n_Samples))
      print("Fitted " + estimator_type + " for: ", i)

  plotGraph(components_min, components_max, wcss, 'Adult ' + estimator_type + ' - Elbow method', 'Number of clusters', 'Within cluster sum of squares', part + '-Adult-' + estimator_type + '-elbow.png')
  plotGraph(components_min, components_max, homogeneity_score, 'Adult ' + estimator_type + ' - Homogeneity Score', 'Number of clusters', 'Homogeneity Score', part + '-Adult-' + estimator_type + '-homogeneity.png')
  plotGraph(components_min, components_max, completeness_score, 'Adult ' + estimator_type + ' - Completeness Score', 'Number of clusters', 'Completeness Score', part + '-Adult-' + estimator_type + '-completeness.png')
  plotGraph(components_min, components_max, v_measure_score, 'Adult ' + estimator_type + ' - V_Measure Score', 'Number of clusters', 'V_Measure Score', part + '-Adult-' + estimator_type + '-v_measure.png')
  plotGraph(components_min, components_max, adjusted_rand_score, 'Adult ' + estimator_type + ' - Adjusted Random Score', 'Number of clusters', 'Adjusted Random Score', part + '-Adult-' + estimator_type + '-adjusted_random.png')  
  plotGraph(components_min, components_max, adjusted_mutual_info_score, 'Adult ' + estimator_type + ' - Adjusted Mutual Info Score', 'Number of clusters', 'Adjusted Mutual Info Score', part + '-Adult-' + estimator_type + '-adjusted_mutual_info.png')

def plot_GM(estimator_type, data, part):
  for i in range(1, 30):
      estimator = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
      estimator.fit(data)

      homogeneity_score.append(metrics.homogeneity_score(labels, estimator.predict(data)))
      completeness_score.append(metrics.completeness_score(labels, estimator.predict(data)))
      v_measure_score.append(metrics.v_measure_score(labels, estimator.predict(data)))
      adjusted_rand_score.append(metrics.adjusted_rand_score(labels, estimator.predict(data)))
      adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(labels, estimator.predict(data)))
      #silhouette_score.append(metrics.silhouette_score(y, kmeans.labels_, metric='euclidean', sample_size=n_Samples))
      print("Fitted " + estimator_type + " for: ", i)

  plotGraph(components_min, components_max, wcss, 'Adult ' + estimator_type + ' - Elbow method', 'Number of clusters', 'Within cluster sum of squares', part + '-Adult-' + estimator_type + '-elbow.png')
  plotGraph(components_min, components_max, homogeneity_score, 'Adult ' + estimator_type + ' - Homogeneity Score', 'Number of clusters', 'Homogeneity Score', part + '-Adult-' + estimator_type + '-homogeneity.png')
  plotGraph(components_min, components_max, completeness_score, 'Adult ' + estimator_type + ' - Completeness Score', 'Number of clusters', 'Completeness Score', part + '-Adult-' + estimator_type + '-completeness.png')
  plotGraph(components_min, components_max, v_measure_score, 'Adult ' + estimator_type + ' - V_Measure Score', 'Number of clusters', 'V_Measure Score', part + '-Adult-' + estimator_type + '-v_measure.png')
  plotGraph(components_min, components_max, adjusted_rand_score, 'Adult ' + estimator_type + ' - Adjusted Random Score', 'Number of clusters', 'Adjusted Random Score', part + '-Adult-' + estimator_type + '-adjusted_random.png')  
  plotGraph(components_min, components_max, adjusted_mutual_info_score, 'Adult ' + estimator_type + ' - Adjusted Mutual Info Score', 'Number of clusters', 'Adjusted Mutual Info Score', part + '-Adult-' + estimator_type + '-adjusted_mutual_info.png')


n_digits_i = [2,5,10,20,50]
print("Part 1: K-Means on Adult")
for i in range(5):
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits_i[i], n_init=10),name="k-means++", data=data)

plot_estimator("Kmeans", data, "Part1")

print("Part 1: EM on Adult")
for i in range(5):
    bench_k_means(GaussianMixture(n_components=n_digits_i[i],random_state=0),name="GaussianMixture", data=data)
selectEM(data)

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

'''

    expected = y_test
    predicted = clf.predict(X_test)
    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
'''
from sklearn.neural_network import MLPClassifier
t0 = time()
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
