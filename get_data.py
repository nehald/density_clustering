

def get_data():
	data=[]
	f = open("sfpd_incident_2012_copy.csv")
	f.next()
	for i in f:
		try:
			lon,lat = tuple([float(i) for i in i.strip().split(",")[9:11]]) 
			data.append((lon,lat))
		except:
			pass
	return data

#d = get_data()[0:30]
#for i in range(0,30):
#	a,b = d[i]
#	print str(list(d[i]))+"," 

