import numpy as np
import scipy as sc
def fitz_data():
	fitz =[]
	for i in open("/tmp/fitz.dat").readlines():
		fitz.append([int(j) for j in i.strip()[1:-1].split(",")])
	return np.array(fitz)
