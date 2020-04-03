
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
nv = 30 #number of variables
npoints = int(10e4)#int(1e5) #time points available
arr = np.zeros((nv,npoints,seed_number))
av_ar = np.zeros((nv,npoints)) #average traces
ar_std = np.zeros((nv,npoints))
time = np.zeros((nv, npoints)) #time

li = ['/D.outer_membrane_final.dat','/D.inner_with_cristae_final.dat','/T.inner_with_cristae_final.dat','/T.outer_membrane_final.dat','/L.inner_with_cristae_final.dat','/DL.World.dat','/LD.World.dat','/TL.World.dat','/LT.World.dat','/DLT.World.dat','/TLD.World.dat','/DLD.World.dat','/DLDp.World.dat','/TLT.World.dat','/TLTp.World.dat','/Eo.World.dat','/Ei.World.dat','/H3Eo.World.dat', '/H3E.World.dat','/EH3.World.dat','/H3ES.World.dat','/atp_prod.World.dat','/atp_dis.World.dat','/prod.World.dat','/counter_prod.World.dat','/unprod_d.World.dat','/unprod_dp.World.dat','/exp_t.World.dat','/imp_t.World.dat','/T.Cube.dat']


for i in range(seed_number):
    print i
    if i>8:
        for s,j in enumerate(li):
            var = np.genfromtxt("./random/react_data/param1_000"+str(i+1)+j, dtype = float)
            arr[s,:,i]= var[:npoints,1]
            time[s,:] = var[:npoints,0]

    else:
        for s,j in enumerate(li):
            var = np.genfromtxt("./random/react_data/param1_0000"+str(i+1)+j, dtype = float)
            #print "./random/seed_0000"+str(i+1)+"_1e9"+j
            arr[s,:,i]= var[:npoints,1]
            time[s,:] = var[:npoints,0]

arr[29,:,:] = arr[29,:,:]-arr[3,:,:]
arr[0,:,:] = arr[0,:,:]-arr[1,:,:]
arr[3,:,:] = arr[3,:,:]-arr[2,:,:]
arr[21,:,:] = arr[21,:,:]-arr[22,:,:]
arr[23,:,:] = arr[23,:,:]-arr[24,:,:]
arr[27,:,:] = arr[27,:,:]-arr[28,:,:]

av_ar = np.average(arr, axis=2)
ar_std = np.std(arr, axis=2)

the_filename = 'av_10rc_1e9'
with open(the_filename, 'wb') as f:#
	pickle.dump(av_ar, f)#
np.savetxt('time_10rc_1e9',time)
