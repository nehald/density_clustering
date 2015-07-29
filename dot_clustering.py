import time
from sklearn.cluster import DBSCAN
import cv2
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib.pyplot as plt
import collections
pos=[]
img = cv2.imread("testimage.jpg")
for line in open("class_meta"):
	line = [float(i) for i in line.strip().split(" ")]
	if line[2] > .8:
		pos.append([line[0],line[1]])

db=DBSCAN(eps=16,min_samples=6)
X = db.fit(pos)

pos_label_dict={}
pos_labels = zip(X.labels_,pos) 
for a,b in pos_labels:
	if a > 0:
		try:
			pos_label_dict[a].append(b)
		except:
			pos_label_dict[a] = []	
			pos_label_dict[a].append(b)


convex_hull_area=[]
for k in pos_label_dict.keys():
	x = pos_label_dict[k]
#	for i in x:
#		a,b = tuple(i) 
#		cv2.circle(img,(int(a),int(b)),4,(255,255,0),-1)	
#		cv2.imshow("Window",img)
#		cv2.waitKey(1)
	sp = MultiPoint(x) 
	sp_convex= sp.convex_hull
	convex_hull_area.append(sp_convex.area)
##
conv_array = np.array(convex_hull_area)
conv_array_count = collections.Counter(conv_array)
conv = dict(conv_array_count.most_common(5))
total = 0
samps=0
for k in conv.keys():
	total = k*conv[k]+total	
	samps = samps+conv[k]
	print total,samps,total/samps 
weight_avg = total/samps 
mean=np.mean(np.array(convex_hull_area)) 
std=np.std(np.array(convex_hull_area))
print mean,std
print sum(conv_array/weight_avg)
#plt.hist(conv_array)
#plt.show()
##

