import numpy as np
from scipy import spatial
from math import sqrt, radians, cos, sin, asin, atan2
import pdb
import get_data
import AutomaticClustering as AutoC

def haversine(A, B):
    """ This function Calculates the distance between points A and B.
    A and B are tuples.. A= (longitude,latitude) """
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


def update(core_dist_pt, distance_mat, point, seeds):
    ## find all the neighbors
    cd_vector = np.ones(len(seeds)) * core_dist_pt
    reach_vector = distance_mat[point][seeds]
    temp = np.column_stack((cd_vector, reach_vector))
    #pdb.set_trace()
    maxtemp =  np.max(temp, axis=1)
    return maxtemp



def dist_mat(points, metric):
	if metric == 'haversine':
    		distance_mat = spatial.distance.pdist(points, lambda u, v: haversine(u, v))
   	else:
    		distance_mat = spatial.distance.pdist(points, metric)
	return spatial.distance.squareform(distance_mat)


def optics(points, min_cluster_size,metric):
    # setup variables
    """ The optics algorithm.  The optics algorithm tries overcome the
        limitation of the DBSCAN algorithms.  The main limitation of the DBSCAB
        alogrithm is that it requires having a idea of the density of the clusters.
       """
    #  Constant values
    m, n = points.shape
    rd = np.ones(m) * 100000000
#    cd = {i: -1 for i in range(m)}
    cd = np.ones(m)*-1;    
    ordered = []
    distance_mat = dist_mat(np.array(X), metric)
    tmp = np.zeros((m, m)) - 1
    # calculate core distance.  The core distance 
    # is the distance from a point to its nth-neighbor 
    #  
    for point in xrange(m):
        try:
	    ## get neighbor list in sorted order (closest to farthest)
            nbr_list = sorted([i for i in distance_mat[point] if i > 0])
            if len(nbr_list) > min_cluster_size - 1:
                cd[point] = nbr_list[min_cluster_size - 1]
	except:
            	cd[point] = -1
	
    ## calculate the reachability
    processed=[]
    index = 0
    seeds =  np.array([i for i in range(0,m)]) 
    while len(seeds) != 1: 
	seed_trial = seeds[index] 
	processed.append(seed_trial)
	#print processed	
	seed_indexes = np.where(seeds != seed_trial)
	seeds = seeds[seed_indexes]
	## compare the core distance and the reachability		
	rd_temp=update(cd[seed_trial], distance_mat,seed_trial, seeds)
	## compare the current reachability matrix with an updated rd
	## if the updata rd is less then the rd	
	rd_index = np.where(rd[seeds]>rd_temp)[0]	
	#pdb.set_trace()	
	rd[seeds[rd_index]] = rd_temp[rd_index]
	index = np.argmin(rd[seeds]) 
    processed.append(seeds[0])
    rd[0] =0
    return rd,cd,processed 
		
if __name__ == '__main__':
	#X = np.load("zhang.dat.npy")	
	X = get_data.get_data()[0:900]	
	#rd,cd,processed = optics(np.array(X),30,"euclidean")
	rd,cd,processed = optics(np.array(X),100,"haversine")
	# zhang_results= np.array(processed) 
	# np.save("zhang_results.dat",zhang_results)
	RPlot = []
	RPoints = []
	for item in processed:
    		RPlot.append(rd[item]) #Reachability Plot
    		RPoints.append([X[item][0],X[item][1]]) #points in their order determined by OPTICS
	rootNode = AutoC.automaticCluster(RPlot, RPoints)
	AutoC.graphTree(rootNode, RPlot)
