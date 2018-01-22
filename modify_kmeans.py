import sys, os

import numpy as np
import scipy.stats as stats

from Bio import Cluster

#############################
##configuration parameters
kmeans_n = 100

num_process = 10
#############################
##public API
def k_means(dataset,k = kmeans_n):
    m, n = np.shape(dataset)
    cluster_assment = np.zeros((m, 2))
    avg = []
    avg_r = 0

    print "begin clustering"

    centroids = rand_cent(dataset, k)

    for t in range(10):
        cluster_changed = False
        for i in range(m):
            max_r = -1
            max_index = -1
            for j in range(k):
                r_ji = dist_r(centroids[j, :], dataset[i, :])
                # print centroids[j,:],dataset[i,:],r_ji
                if r_ji > max_r:
                    max_r = r_ji
                    max_index = j
            if cluster_assment[i, 0] != max_index:
                cluster_changed = True
                cluster_assment[i, :] = max_index, max_r

        if not cluster_changed:
            break

        for cent in range(k):
            pts_in_clust = dataset[
                np.nonzero(np.logical_and(cluster_assment[:, 0] == cent, cluster_assment[:, 1] > 0.8))]
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)

        avg_r = np.mean(cluster_assment[:, 1])
        avg += [avg_r]
        if avg_r > 0.95:
            break

        print "cycle count:", t, "\t average r2:", avg_r

    print "exit"
    return centroids, cluster_assment, avg_r


#############################
##private API

def dist_r(vecA, vecB):
    return abs(stats.pearsonr(vecA, vecB)[0])

def rand_cent(dataset, k):
    n=np.shape(dataset)[1]

    centroids=np.zeros((k,n))
    for j in range(n):
        min_j = min(dataset[:,j])
        range_j = float(max(dataset[:,j])-min_j)
        centroids[:,j]=min_j+range_j*np.random.rand(k)

    return centroids


##for debug:
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dataMat = np.array([[0,1,2],[0,0.1,0.2],[1,2,3],[1,4,7],[0,3,6],[0.1,0.4,0.7]])
    print k_means(dataset=dataMat, k=2)



