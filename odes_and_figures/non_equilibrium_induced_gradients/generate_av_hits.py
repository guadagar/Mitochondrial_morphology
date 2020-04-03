
#!/usr/bin/env python
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,ScalarFormatter

'''
This script generates the average trace over 10 different initial conditions.
GCG
04.02.2020
'''

seed_number = 10 #number of seeds
nv = 6 #number of variables
npoints = int(10e4)#int(1e5) #time points available
arr = np.zeros((nv,npoints,seed_number))
av_ar = np.zeros((nv,npoints)) #average traces
ar_std = np.zeros((nv,npoints))
time = np.zeros((nv, npoints)) #time

li = ['/D.om.dat','/D.im.dat','/D.cm.dat','/T.om.dat','/T.im.dat','/T.cm.dat']


for i in range(seed_number):
    print i
    if i>8:
        for s,j in enumerate(li):
            var = np.genfromtxt("./random/react_data/param1m_000"+str(i+1)+j,dtype = float)
            print "./random/react_data/param1m_0000"+str(i+1)
            arr[s,:,i]= var[:npoints,1]
            time[s,:] = var[:npoints,0]

    else:
        for s,j in enumerate(li):
            var = np.genfromtxt("./random/react_data/param1m_0000"+str(i+1)+j, dtype = float)
            #print "./random/seed_0000"+str(i+1)+"_1e9"+j
            arr[s,:,i]= var[:npoints,1]
            time[s,:] = var[:npoints,0]

#arr[29,:,:] = arr[29,:,:]-arr[3,:,:]
#arr[0,:,:] = arr[0,:,:]-arr[1,:,:]
#arr[3,:,:] = arr[3,:,:]-arr[2,:,:]
#arr[21,:,:] = arr[21,:,:]-arr[22,:,:]
#arr[23,:,:] = arr[23,:,:]-arr[24,:,:]
#arr[27,:,:] = arr[27,:,:]-arr[28,:,:]

av_ar = np.average(arr, axis=2)
ar_std = np.std(arr, axis=2)


the_filename = 'av_10r_1e9_conc'
with open(the_filename, 'wb') as f:#
	pickle.dump(av_ar, f)
np.savetxt('time_10r_1e9_conc',time)
