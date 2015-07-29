import time
import cv2
pos=[]
img = cv2.imread("testimage.jpg")


## loop through class_meta
## select entries greater than .9 
for line in open("class_meta"):
        line = [float(i) for i in line.strip().split(" ")]
        if line[2] > .9:
            	print line 
   		pos.append([line[0],line[1]])

## loop through the position array
for i in pos:
	a,b= tuple(i)
	cv2.circle(img,(int(a),int(b)),4,(255,255,0),-1)
	cv2.imshow("Window",img)
	cv2.waitKey(1)

time.sleep(10)
