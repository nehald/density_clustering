import numpy as np
from scipy import spatial
from math import sqrt, radians, cos, sin, asin, atan2
import pdb
import data


def haversine(A, B):
    """ This function Calculates the distance between points A and B.
    A is a tuple.. A= (longitude,latitude) """
    lon1, lat1 = A
    lon2, lat2 = B
    R = 6372.8  # Earth radius in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def dist_mat(points, metric):
    distance_mat = spatial.distance.pdist(points, lambda u, v: haversine(u, v))
    return spatial.distance.squareform(distance_mat)


def update(cd_mat, sf_filtered_mat, point, processed):
    neighbors = np.where(sf_filtered_mat[point] > 0)
    neighbors

    num_neighbors = len(neighbors[0])
    cd_vector = np.ones(num_neighbors) * cd_mat[point]
    reach_vector = sf_filtered_mat[point][neighbors]
    temp = np.column_stack((cd_vector, reach_vector))
    maxtemp = np.column_stack((neighbors[0], np.max(temp, axis=1)))
    return maxtemp


def optics(points, max_radius, min_cluster_size):
    # setup variables
    """ The optics algorithm.  The optics algorithm tries overcome the
        limitation of the DBSCAN algorithms.  The main limitation of the DBSCAB
        alogrithm is that it requires having a idea of the density of the clusters.
        """


    m, n = points.shape
    rd = np.zeros(m) * 100000000
    cd = {i: -1 for i in range(m)}

    ordered = []
    sf = dist_mat(np.array(X), "foo")
    tmp = np.zeros((m, m)) - 1
    sf_filtered = np.where(sf < max_radius, sf, tmp)
    print sf_filtered
    # calculate core distance
    for j in xrange(m):
        try:
            nbr_list = np.unique(sorted([i for i in sf_filtered[j] if i > 0]))
            if len(nbr_list) > min_cluster_size - 1:
                cd[j] = nbr_list[min_cluster_size - 1]
        except:
            cd[j] = -1
            pass

    unprocessed = [i for i in range(0, m)]
    processed = [False for i in range(0, m)]
    seeds = np.zeros((m))
    rd = np.zeros((m))
    # update
    # reachability distance
    rd_new = True
    while unprocessed:
        point = unprocessed.pop(0)
        # get neighbors
        processed[point] = True
        mat_ret = update(cd, sf_filtered, point, processed)
        # sort the result by shortest distance
        mat_ret = mat_ret[np.argsort(mat_ret[:, 1])]
        if rd_new:
            seeds = mat_ret[:, 0].astype(int)
            rd[seeds] = mat_ret[:, 1]
            rd_new = False
        else:
            seeds = np.append(seeds, mat_ret[:, 0].astype(int))
            rd[seeds] = np.minimum(rd[seeds], mat_ret[:, 1])

        while(list(seeds)):
            seed_index = list(seeds).pop(0)
            processed[seed_index] = True
            mat_ret = update(cd, sf_filtered, seed_index, processed)
            seeds = np.append(seeds, mat_ret[:, 0].astype(int))
            rd[seeds] = mat_ret[:, 1]
    print 'foo'

if __name__ == '__main__':
	X = data.sf_data
	cd = optics(np.array(X), 4, 6)
