import numpy as np
import scipy.stats as ss
import random
import subprocess as sp

evt_id = np.loadtxt("6T5_data_to_simulate.txt",usecols=(0,),dtype=int)
E,glon,glat = np.loadtxt("6T5_data_to_simulate.txt",usecols=(5,1,2),unpack=True)

for i in range(10):

	ID = evt_id[i]

	data_array = np.zeros((20000,9))

	for j in range(40000,60000):
		if j % 50 == 0:
			np.save('samples/evt_%i_d' %ID,data_array) #Save to disk every 1000 iterations
			print("Iteration %i / %i" %(j,60000))
		s1,s2,s3 = random.sample(range(1,2**26-1),3)
		cmd = ['timeout','35','./call_crp.py','%f' %E[i],
			'%f' %glon[i], '%f' %glat[i], '%i' %s1, '%i' %s2, '%i' %s3]
		proc=sp.Popen(cmd,stdout=sp.PIPE,stderr=sp.PIPE,close_fds=True)
		out,err=proc.communicate()
		out=out.decode('ascii')
		try:
			out=out.replace('(','')
			out=out.replace(')','')
			out=out.replace('\n','')
			out=out.split(',')
			out=[float(m) for m in out]
			for k in range(8):
			  data_array[j-60000,k] = out[k]
		except:
			data_array[j-60000,8] = 1.
			print("Timeout")
	np.save('samples/evt_%i_d' %ID,data_array)
