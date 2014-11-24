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

def optics(points,max_radius,min_cluster_size):
	###	
	m,n = points.shape
	rd = np.zeros(m)*100000000
	cd = {} 
	
	ordered=[]
	sf =dist_mat(np.array(X),"foo")
	tmp = np.zeros((m,m))-1 
	sf_filtered = np.where(sf<max_radius,sf,tmp)
	print sf_filtered
	## calculate core distance 
	for j in xrange(m):
		try:	
			nbr_list=np.unique(sorted([i for i in sf_filtered[j] if i>0]))
			print nbr_list
			pdb.set_trace()	
			if len(nbr_list) > min_cluster_size -1:
				cd[j] = nbr_list[min_cluster_size-1]	
		except:
			cd[j] = -1 
			pass;
	processed_list=[]
	
	## seeds ZZ
	seeds = np.arange(m,dtype=N.it)

X = data.sf_data
cd = optics(np.array(X),4,6)
for i in cd.keys():
	print i,cd[i]
