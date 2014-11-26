import numpy as np
from scipy import spatial
from math import sqrt,radians,cos,sin,asin,atan2
import pdb
import data

def haversine(A,B):
	lon1,lat1 = A
	lon2,lat2 = B  
	print A,B	
	R = 6372.8 # Earth radius in kilometers
  	dLat = radians(lat2 - lat1)
  	dLon = radians(lon2 - lon1)
  	lat1 = radians(lat1)
  	lat2 = radians(lat2)
  	a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  	c = 2*asin(sqrt(a))
	return R * c

def dist_mat(points,metric):
	distance_mat = spatial.distance.pdist(points,lambda u,v: haversine(u,v))
	return spatial.distance.squareform(distance_mat)


def update(cd_mat,sf_filtered_mat,point) : 
	neighbors = np.where(sf_filtered_mat[point]>0)
	num_neighbors = len(neighbors[0])
	cd_vector = np.ones(num_neighbors)*cd_mat[point]
	reach_vector = sf_filtered_mat[point][neighbors]
	temp = np.column_stack((cd_vector, reach_vector))
	maxtemp = np.column_stack((neighbors[0],np.max(temp,axis=1)))
	return maxtemp


def optics(points,max_radius,min_cluster_size):
	###	
	m,n = points.shape
	rd = np.zeros(m)*100000000
	cd = {i: -1 for i in range(m)} 
	
	ordered=[]
	sf =dist_mat(np.array(X),"foo")
	tmp = np.zeros((m,m))-1 
	sf_filtered = np.where(sf<max_radius,sf,tmp)
	print sf_filtered
	## calculate core distance 
	for j in xrange(m):
		try:	
			nbr_list=np.unique(sorted([i for i in sf_filtered[j] if i>0]))
			if len(nbr_list) > min_cluster_size -1:
				cd[j] = nbr_list[min_cluster_size-1]	
		except:
			cd[j] = -1 
			pass;

	unprocessed= [i for i in range(0,m)]	
	processed=[]
	seeds = np.zeros((m,2)) 
	rd = np.arange(m,dtype=np.int)
	## update 
	## reachability distance
	while unprocessed:	
		point = unprocessed.pop()	
		## get neighbors
		processed.append(point)
		pdb.set_trace()
		mat_ret = update(cd,sf_filtered,point)	
	print 'foo'


X = data.sf_data
cd = optics(np.array(X),4,6)
