import os
import sys
import re
import time
from pyspark import SparkContext
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row
# from pyspark.sql.functions import *
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyspark.sql.functions as func
import matplotlib.patches as mpatches
from operator import add
from pyspark.mllib.clustering import KMeans, KMeansModel
from operator import add
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
import itertools

input_path_of_file = "/datasets/kddcup.data"
data_raw = sc.textFile(input_path_of_file, 12)


def parseVector(line):
    columns = line.split(',')
    thelabel = columns[-1]
    featurevector = columns[:-1]
    featurevector = [element for i, element in enumerate(featurevector) if i not in [1, 2, 3]]
    featurevector = np.array(featurevector, dtype=np.float)
    return (thelabel, featurevector)

labelsAndData = raw_data.map(parseVector).cache()
thedata = labelsAndData.map(lambda row: row[1]).cache()
n = thedata.count()


# we do a initial K Mean with 2 clusters
time1 = time.time()
k_clusters = KMeans.train(thedata, 2, maxIterations=10, runs=10, initializationMode="random")
print(time.time() - time1)



def getFeatVecs(data):
    n = thedata.count()
    means = thedata.reduce(add) / n
    vecs_ = thedata.map(lambda x: (x - means)**2).reduce(add) / n
    return vecs_

vecs_ = getFeatVecs(data)


mean = thedata.map(lambda x: x[1]).reduce(add) / n
print(thedata.filter(lambda x: x[1] > 10*mean).count())
4499




#We want to identify the features that vary the most and be able to plot them

indices_of_variance = [t[0] for t in sorted(enumerate(vars_), key=lambda x: x[1])[-3:]] 
dataprojected = thedata.randomSplit([10, 90])[0]
# separate into two rdds
rdd0 = thedata.filter(lambda point: k_clusters.predict(point)==0)
rdd1 = thedata.filter(lambda point: k_clusters.predict(point)==1)

center_0 = k_clusters.centers[0]
center_1 = k_clusters.centers[1]
cluster_0 = rdd0.take(5)
cluster_1 = rdd1.take(5)

cluster_0_projected = np.array([[point[i] for i in indices_of_variance] for point in cluster_0])
cluster_1_projected = np.array([[point[i] for i in indices_of_variance] for point in cluster_1])


M = max(max(cluster1_projected.flatten()), max(cluster_0_projected.flatten()))
m = min(min(cluster1_projected.flatten()), min(cluster_0_projected.flatten()))


fig2plot = plt.figure(figsize=(8, 8))
pltx = fig2plot.add_subplot(111, projection='3d')
pltx.scatter(cluster0_projected[:, 0], cluster0_projected[:, 1], cluster0_projected[:, 2], c="b")
pltx.scatter(cluster1_projected[:, 0], cluster1_projected[:, 1], cluster1_projected[:, 2], c="r")
pltx.set_xlim(m, M)
pltx.set_ylim(m, M)
pltx.set_zlim(m, M)
pltx.legend(["cluster 0", "cluster 1"])


def euclidean_distance_points(x1, x2):
    x3 = x1 - x2
    return np.sqrt(x3.T.dot(x3))


from operator import add
time1 = time.time()

def ss_error(k_clusters, point):
    nearest_center = k_clusters.centers[k_clusters.predict(point)]
    return euclidean_distance_points(nearest_center, point)**2

WSSSE = data.map(lambda point: ss_error(k_clusters, point)).reduce(add)
print("Within Set Sum of Squared Error = " + str(WSSSE))
print(time.time() - time1)

clusterLabel = labelsAndData.map(lambda row: ((k_clusters.predict(row[1]), row[0]), 1)).reduceByKey(add)

for items in clusterLabe.collect():
    print(items)

k_values = range(5, 126, 20)

def clustering_error_Score(thedata, k):
    k_clusters = KMeans.train(thedata, k, maxIterations=10, runs=10, initializationMode="random")
#   WSSSE = thedata.map(lambda point: error(k_clusters, point)).reduce(add)
    WSSSE = k_clusters.computeCost(thedata)
    return WSSSE

k_scores = [clustering_error_Score(thedata, k) for k in k_values]
for score in k_scores:
    print(score)
    
plt.scatter(k_values, k_scores)
plt.xlabel('k')
plt.ylabel('k_clustering score')

def normalize(thedata):
    
    n = thedata.count()
    avg = thedata.reduce(add) / n
   
    var = thedata.map(lambda x: (x - avg)**2).reduce(add) / n
    std = np.sqrt(var)
    
    std[std==0] = 1

    def normalize(val):
        return (val - avg) / std
    return thedata.map(normalize)

normalized = normalize(data).cache()
print(normalized.take(2))
print(thedata.take(2))

k_range = range(60, 111, 10)

k_scores = [clustering_error_Score(normalized, k) for k in k_range]
for kscore in k_scores:
    print(kscore)

plt.plot(k_range, kscores)

# testing normalization and non normalized clusters
#before norm
K_norm = 90

var = getVariance(thedata)
indices_of_variance = [t[0] for t in sorted(enumerate(var), key=lambda x: x[1])[-3:]]

dataprojected = thedata.randomSplit([1, 999])[0].cache()

kclusters = KMeans.train(thedata, K_norm, maxIterations=10, runs=10, initializationMode="random")

listdataprojected = dataprojected.collect()
projected_data = np.array([[point[i] for i in indices_of_variance] for point in listdataprojected])
klabels = [kclusters.predict(point) for point in listdataprojected]


Maxi = max(projected_data.flatten())
mini = min(projected_data.flatten())

figs = plt.figure(figsize=(8, 8))
pltx = figs.add_subplot(111, projection='3d')
pltx.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c=klabels)
pltx.set_xlim(mini, Maxi)
pltx.set_ylim(mini, Maxi)
pltx.set_zlim(mini, Maxi)
pltx.set_title("Before normalization")

#After normalization:

kclusters = KMeans.train(normalized, K_norm, maxIterations=10, runs=10, initializationMode="random")

dataprojected_normed = normalize(thedata, dataprojected).cache()
dataprojected_normed = dataprojected_normed.collect()
projected_data = np.array([[point[i] for i in indices_of_variance] for point in dataprojected_normed])
klabels = [kclusters.predict(point) for point in dataprojected_normed]

# Take the extrema values
Maxi = max(projected_data.flatten())
mini = min(projected_data.flatten())

# Do the plot
figs = plt.figure(figsize=(8, 8))
pltx = fig.add_subplot(111, projection='3d')
pltx.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c=klabels)
pltx.set_xlim(mini, Maxi)
pltx.set_ylim(mini, Maxi)
pltx.set_zlim(mini, Maxi)
pltx.set_title("After normalization")

#one hot encoding for categorical
col1 = raw_data.map(lambda line: line.split(",")[1]).distinct().collect()
col2 = raw_data.map(lambda line: line.split(",")[2]).distinct().collect()
col2 = raw_data.map(lambda line: line.split(",")[3]).distinct().collect()

def parseWithOneHotEncoding(line):
    column = line.split(',')
    thelabel = column[-1]
    thevector = column[0:-1]
    
    col1 = [0]*len(featureCol1)
    col1[col1.index(vector[1])] = 1
    col2 = [0]*len(col2)
    col2[featureCol1.index(vector[2])] = 1
    col2 = [0]*len(featureCol3)
    col2[featureCol1.index(vector[3])] = 1
    
    thevector = ([thevector[0]] + col1 + col2 + col3 + thevector[4:])
    
    thevector = np.array(thevector, dtype=np.float)
    
    return (thelabel, thevector)

labelsAndData = raw_data.map(parseLineWithHotEncoding)

thedata = labelsAndData.values().cache()

normalized = normalize(thedata).cache()

#final k means
kclusters = KMeans.train(data, 100, maxIterations=10, runs=10, initializationMode="random")

anomaly = normalized.map(lambda point: (point, error(clusters, point))).takeOrdered(100, lambda key: key[1])
plt.plot([ano[1] for ano in anomaly])

