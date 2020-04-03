
#!/usr/bin/env python
import PyDSTool
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,ScalarFormatter
from time import clock
import matplotlib as mpl

'''
ANT-ATPase final version, with clamp, and vdacs. The average traces generated in mcell are needed.
04.02.2020
GCG
'''

params = {'axes.labelsize': 9,
           'axes.titlesize': 6,
          'legend.fontsize': 6,
           'xtick.labelsize': 6,
           'ytick.labelsize': 6,
            'figure.figsize': (3.4,3.6)}
mpl.rcParams.update(params)

Na = 6.02214e23
#vmito = 0.033221e-15 #lt
#vcyto = 0.232549e-15 #
vmito = 0.020706e-15 #lt OM
vcyto = 0.0181e-15 # IM
vcube = 0.409194e-15

no1_ant = 20e3
k7 = 800.0 #0.74
k8 = 0.7 #0.05
k9 = 0.58 #0.37
k10 = 0.48#0.47
no_atp = 3800 #number of atphase
a12 = 24
a21 = 40.0
a23 = 4.0
a32 = 5e3
ac = 9807 #clamp*Na*vcyto = number of particles
kp  = 1.0
n_porin = 7100

DSargs = PyDSTool.args(name='ex')

DSargs.pars = { 'k1_on':'4.0',#          #KDo
                'k1_off':'100', #         #KDo
                'k2_on': '6.4',#         #KTi
		        'k2_off':'40000.0',#     #KTi
		        'k5_on':'0.4',           #KTo
	            'k5_off':'200.0',           #KTo
	         	'k6_on':'4.0',           #KDi
		        'k6_off':'40000.0',      #KDi
		        'k7':k7,           #kp
		        'k8':k8,              #kcp
		        'k9':k9,             #kt
		        'k10':k10,            #kd
		        'no_ant': no1_ant,
		        'a12':a12, #s-1
		        'a21':a21,#s-1
		        'a65':'924',           #s-1
		        'a56':'1e3',             #s-1,
		        'a16':'452457',           #s-1
		        'a61':'11006',           #s-1
		        'a25':'1.17e-12',     #s-1
		        'a52':'2.0',          #s-1
		        'a54':'1e2',          #s-1
	         	'a45':'1e2',          #s-1
		        'a43':'0.8',         #uM-1s-1
		        'a34':'1e2',
		        'a32':a32,          #s-1
		        'a23':a23,         #uM-1s-1
		        'no1_atp': no_atp, #number of atpases
		        'vcyto':vcyto,
		        'vcube':vcube,
		        'vmito':vmito,
		        'kp':kp,
		        'n_porin':n_porin,
		        'Na':Na,
		        'fa':'0.5',
		        'ac':ac}

DSargs.varspecs = { 'am':'-k6_on*(1e6*am/(Na*vmito))*bl + k6_off*bla - k6_on*(1e6*am/(Na*vmito))*l + k6_off*la - 2.0*k6_on*fa*(1e6*am/(Na*vmito))*al + k6_off*ala + k6_off*alap + a34*h3es - a43*(1e6*am/(Na*vmito))*h3eo',
                    'bm':'-k2_on*(1e6*bm/(Na*vmito))*al + k2_off*(no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp) - k2_on*(1e6*bm/(Na*vmito))*l + k2_off*lb - 2.0*k2_on*fa*(1e6*bm/(Na*vmito))*bl + k2_off*blb + k2_off*blbp -a23*h3e*(1e6*bm/(Na*vmito)) + a32*h3es',
                    'bo':'-kp*(1e6*bo/(Na*vcube))*n_porin + kp*(1e6*bc/(Na*vcyto))*n_porin',
                    'bc':'-k5_on*(1e6*bc/(Na*vcyto))*l + k5_off*bl - k5_on*(1e6*bc/(Na*vcyto))*la + k5_off*bla- 2.0*k5_on*fa*(1e6*bc/(Na*vcyto))*lb + k5_off*blb + k5_off*blbp - kp*(1e6*bc/(Na*vcyto))*n_porin + kp*(1e6*bo/(Na*vcube))*n_porin',
                    'l':'-k1_on*(1e6*ac/(Na*vcyto))*l + k1_off*al- k2_on*(1e6*bm/(Na*vmito))*l + k2_off*lb - k5_on*(1e6*bc/(Na*vcyto))*l + k5_off*bl - k6_on*(1e6*am/(Na*vmito))*l + k6_off*la',
                    'al':'k1_on*(1e6*ac/(Na*vcyto))*l - k1_off*al - k2_on*(1e6*bm/(Na*vmito))*al + k2_off*(no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp) - k6_on*fa*(1e6*am/(Na*vmito))*al + k6_off*ala - k6_on*fa*(1e6*am/(Na*vmito))*al + k6_off*alap',
                    'la':'k6_on*(1e6*am/(Na*vmito))*l - k6_off*la  - k5_on*(1e6*bc/(Na*vcyto))*la + k5_off*bla - k1_on*fa*(1e6*ac/(Na*vcyto))*la + k1_off*ala - k1_on*fa*(1e6*ac/(Na*vcyto))*la + k1_off*alap',
                    'lb':'k2_on*(1e6*bm/(Na*vmito))*l - k2_off*lb - k1_on*(1e6*ac/(Na*vcyto))*lb + k1_off*(no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp) - k5_on*fa*(1e6*bc/(Na*vcyto))*lb + k5_off*blb - k5_on*fa*(1e6*bc/(Na*vcyto))*lb + k5_off*blbp',
                    'bl':'k5_on*(1e6*bc/(Na*vcyto))*l - k5_off*bl- k6_on*(1e6*am/(Na*vmito))*bl + k6_off*bla - k2_on*fa*(1e6*bm/(Na*vmito))*bl + k2_off*blb- k2_on*fa*(1e6*bm/(Na*vmito))*bl + k2_off*blbp',
                    'bla':'k6_on*(1e6*am/(Na*vmito))*bl - k6_off*bla - k8*bla + k7*(no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp) + k5_on*(1e6*bc/(Na*vcyto))*la - k5_off*bla',
                    'ala':'k1_on*fa*(1e6*ac/(Na*vcyto))*la - k1_off*ala + k6_on*fa*(1e6*am/(Na*vmito))*al - k6_off*ala -k10*ala + k10*alap',
                    'alap':'k1_on*fa*(1e6*ac/(Na*vcyto))*la - k1_off*alap + k6_on*fa*(1e6*am/(Na*vmito))*al - k6_off*alap - k10*alap + k10*ala',
                    'blb':'k2_on*fa*(1e6*bm/(Na*vmito))*bl - k2_off*blb + k5_on*fa*(1e6*bc/(Na*vcyto))*lb - k5_off*blb + k9*blbp - k9*blb',
                    'blbp':'k2_on*fa*(1e6*bm/(Na*vmito))*bl - k2_off*blbp + k5_on*fa*(1e6*bc/(Na*vcyto))*lb - k5_off*blbp -k9*blbp +k9*blb',
                    'eo':'-a65*eo + a56*(no1_atp-eo-ei-h3eo-h3es-h3e)+ a16*ei - a61*eo',
                    'ei':'-a16*ei + a61*eo - a12*ei + a21*h3e',
                    'h3eo':'-a45*h3eo + a54*(no1_atp-eo-ei-h3eo-h3es-h3e) + a34*h3es - a43*h3eo*(1e6*am/(Na*vmito))',
                    'h3es':'a43*(1e6*am/(Na*vmito))*h3eo - a34*h3es + a23*h3e*(1e6*(bm)/(Na*vmito)) - a32*h3es',
                    'h3e':'-a23*h3e*(1e6*(bm)/(Na*vmito)) + a32*h3es - a25*h3e + a52*(no1_atp-eo-ei-h3eo-h3es-h3e) + a12*ei - a21*h3e'}

# initial conditions
DSargs.ics  = {'am':37926,
	           'bm':1264,
	           'bc':1.0,
	           'bo':0,
	           'l':4e3,
               'al':12e3,
               'lb':0,
               'bl':0,
               'bla':0,
               'la':0,
               'ala':2e3,
               'blb':0,
               'alap':2e3,
                'blbp':0,
                'eo':3800,
                'ei':0,
                'h3eo':0,
                'h3es':0,
                'h3e':0}

DSargs.tdomain = [0,0.01]

#ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)    # an instance of the 'Generator' class.
ode  = PyDSTool.Generator.Radau_ODEsystem(DSargs)
start = clock()
traj = ode.compute('polarization')
print ('  ... finished in %.3f seconds.\n' % (clock()-start))
pd   = traj.sample()

#------------------ upload average mcell traces ----

with open('./av_10c_1e9_n', 'rb') as f: av_ar = pickle.load(f)

with open('./av_10r_1e9_n', 'rb') as f: av_ra = pickle.load(f)

with open('./av_10rc_1e9_n', 'rb') as f: av_cu = pickle.load(f)
time = np.loadtxt('./time')

#----------to smooth the hit traces ------

def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return   y[window_len:-window_len+1]

#--------------

plt.figure(1)

ax = plt.subplot(4,2,2)
plt.plot(1e3*time,av_ar[0,:], 'b')
plt.plot(1e3*time,av_ra[0,:], 'red')
plt.plot(1e3*time,av_cu[0,:], 'm')
plt.plot(1e3*pd['t'], ac*np.ones(len(pd['t'])),c='k')
ax.locator_params(axis='y',nbins=3)
plt.xlim(-0.15,10)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True,useOffset=True))
plt.ylabel(r'#ADP$_{\rm outside}$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.locator_params(axis='x',nbins=8)
ax.set_yticks([0.8e4,0.9e4,1e4])
#plt.xlabel('Time (msec)')

ax3 = plt.subplot(4,2,3)
plt.plot(1e3*time,av_ar[1,:],'b')
plt.plot(1e3*time,av_ra[1,:],'red')
plt.plot(1e3*time,av_cu[1,:],'m')
plt.plot(1e3*pd['t'], pd['am'],c='k')
ax3.locator_params(axis='y',nbins=2)
plt.xlim(-0.15,10)
ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xlabel('Time (msec)')
plt.ylabel(r'#ADP$_{\rm matrix}$')
ax3.set_yticks([3.70e4,3.75e4,3.8e4])
#plt.text(1,69500,'D', ha='center', va='center', fontsize=11)
#ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax3.locator_params(axis='x',nbins=8)


ax2 = plt.subplot(4,2,4)
plt.plot(1e3*time,av_ar[2,:], 'b')
plt.plot(1e3*time,av_ra[2,:], 'red')
plt.plot(1e3*time,av_cu[2,:], 'm')
plt.plot(1e3*pd['t'], pd['bm'],c='k')
#plt.text(1,1945,'E', ha='center', va='center', fontsize=11)
plt.xlim(-0.15,10)
plt.ylim(0.8e3,1.3e3)
ax2.locator_params(axis='y',nbins=3)
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.set_yticks([0.8e3,1.05e3,1.3e3])
#plt.xlabel('Time (msec)')
plt.ylabel(r'#ATP$_{\rm matrix}$')
#ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2.locator_params(axis='x',nbins=8)


ax1 = plt.subplot(4,2,5)
plt.plot(1e3*time,av_ar[3,:], 'b')
plt.plot(1e3*time,av_ra[3,:], 'red')
plt.plot(1e3*time,av_cu[3,:], 'm')
plt.plot(1e3*pd['t'], pd['bc'],c='k')
#plt.text(1,865,'C', ha='center', va='center', fontsize=11)
plt.xlim(-0.15,10)
ax1.locator_params(axis='y',nbins=3)
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_yticks([0,2e2,4e2])#ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
plt.ylabel(r'#ATP$_{\rm outside}$')
#ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.locator_params(axis='x',nbins=8)
#plt.xlabel('Time (msec)')


ax2 = plt.subplot(4,2,6)
plt.plot(1e3*time,av_ar[29,:],'b')
plt.plot(1e3*time,av_ra[29,:],'red')
plt.plot(1e3*time,av_cu[29,:],'m')
plt.plot(1e3*pd['t'],pd['bo'],'k')
plt.xlim(-0.15,10)
#plt.text(1,850,'F', ha='center', va='center', fontsize=11)
ax2.locator_params(axis='y',nbins=3)
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.subplots_adjust(hspace = 0.25, wspace =0.25)
#plt.xlabel('Time (msec)')
plt.ylabel(r'#ATP$_{\rm cytosol}$')
#ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2.set_yticks([0,3e2,6e2])
ax2.locator_params(axis='x',nbins=8)

dxom = 0.02
dxcm = 0.15
with open('./av_10c_1e9_conc_n', 'rb') as f: av_ar = pickle.load(f)

with open('./av_10r_1e9_conc_n', 'rb') as f: av_ra = pickle.load(f)

with open('./av_10rc_1e9_conc_n', 'rb') as f: av_cu = pickle.load(f)
time = np.loadtxt('./time_conc')

vibs_im = 0.0098e-15 #mas cerca de IM
vibs = 0.00806e-15 # bien cerca del OM
vics = 0.03e-15
vcube = 0.4388047e-15
Na = 6.02214e23

wl = 1000
ini = 500
fin = 501

ax5 = plt.subplot(4,2,7)

#plt.plot(1e3*time[0,:-1],(1e3*(av_ar[3,1:]-av_ar[3,0:-1])/(Na*vibs)-1e3*(av_ar[4,1:]-av_ar[4,0:-1])/(Na*vibs_im))/dxom, 'b')
xsm1 = (1e3*(av_ar[3,1:]-av_ar[3,0:-1])/(Na*vibs)-1e3*(av_ar[4,1:]-av_ar[4,0:-1])/(Na*vibs_im))/dxom
plt.plot(1e3*time[ini:-fin],1*smooth(xsm1, window_len = wl, window='flat'),'b')


#plt.plot(1e3*time8[0,:-1],(1e3*(av_ra[3,1:]-av_ra[3,0:-1])/(Na*vibs)-1e3*(av_ra[4,1:]-av_ra[4,0:-1])/(Na*vibs_im)/dxom), 'red')
xsm2 = (1e3*(av_ra[3,1:]-av_ra[3,0:-1])/(Na*vibs)-1e3*(av_ra[4,1:]-av_ra[4,0:-1])/(Na*vibs_im)/dxom)
plt.plot(1e3*time[ini:-fin],1*smooth(xsm2, window_len = wl, window='flat'),'r')

#plt.plot(1e3*time_cu[0,:-1],(1e3*(av_cu[3,1:]-av_cu[3,0:-1])/(Na*vibs)-1e3*(av_cu[4,1:]-av_cu[4,0:-1])/(Na*vibs_im)/dxom), 'm')
xsm3 = (1e3*(av_cu[3,1:]-av_cu[3,0:-1])/(Na*vibs)-1e3*(av_cu[4,1:]-av_cu[4,0:-1])/(Na*vibs_im)/dxom)
plt.plot(1e3*time[ini:-fin],1*smooth(xsm3, window_len = wl, window='flat'),'m')

#plt.plot(1e3*time_cu[0,:],(1e3*av_cu[3,:]/(Na*vibs)-1e3*av_cu[4,:]/(Na*vibs_im)/dxom), 'm')
ax5.locator_params(axis='y',nbins=4)
ax5.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ylabel(r'$\frac{\Delta[ATP]_{\rm OM-IM}}{\Delta x}$ (mM/$\mu$m)')
#plt.text(1,4e-1,'C', ha='center', va='center', fontsize=11)
#ax5.xaxis.set_major_formatter(plt.NullFormatter())
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax5.locator_params(axis='x',nbins=8)
#plt.ylim(-1,1)
plt.xlabel('Time (msec)')
ax5.set_yticks([-4e-1,-2e-1,0,2e-1])

ax8 = plt.subplot(4,2,8)
xsm1 = (1e3*(av_ar[3,1:]-av_ar[3,0:-1])/(Na*vibs)-1e3*(av_ar[5,1:]-av_ar[5,0:-1])/(Na*vics))/dxcm
plt.plot(1e3*time[ini:-fin],1*smooth(xsm1, window_len = wl, window='flat'),'b')

#plt.plot(1e3*time_cu[1,:-1],(1e3*(av_cu[3,1:]-av_cu[3,0:-1])/(Na*vibs)-1e3*(av_cu[5,1:]-av_cu[5,0:-1])/(Na*vics))/dxcm,'m')
xsm2 = (1e3*(av_cu[3,1:]-av_cu[3,0:-1])/(Na*vibs)-1e3*(av_cu[5,1:]-av_cu[5,0:-1])/(Na*vics))/dxcm
plt.plot(1e3*time[ini:-fin],1*smooth(xsm2, window_len = wl, window='flat'),'m')

#plt.plot(1e3*time8[1,:-1],(1e3*(av_ar[3,1:]-av_ar[3,0:-1])/(Na*vibs)-1e3*(av_ra[5,1:]-av_ra[5,0:-1])/(Na*vics))/dxcm,'red')
xsm3 = (1e3*(av_ar[3,1:]-av_ar[3,0:-1])/(Na*vibs)-1e3*(av_ra[5,1:]-av_ra[5,0:-1])/(Na*vics))/dxcm
plt.plot(1e3*time[ini:-fin],1*smooth(xsm3, window_len = wl, window='flat'),'r')

ax8.locator_params(axis='y',nbins=2)
ax8.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xlabel('Time (msec)')
plt.ylabel(r'$\frac{\Delta[ATP]_{\rm OM-CM}}{\Delta x}$(mM/$\mu$m)')
#plt.text(1,5e-2,'D', ha='center', va='center', fontsize=11)
#ax1.xaxis.set_major_formatter(plt.NullFormatter())
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax8.locator_params(axis='x',nbins=8)
ax8.set_yticks([-5e-2,0,5e-2])

plt.xlabel('Time (msec)')
plt.tight_layout(pad=0.05, w_pad=0.001, h_pad=0.2) #saca los margenes gigantes
plt.subplots_adjust(wspace =0.6)
#plt.subplots_adjust(hspace = 0.6, wspace =0.5)
plt.savefig('pc_av_10_3x2Nn.pdf', transparent=True, format = 'pdf',dpi=600,bbox_inches='tight')
#plt.savefig('pc_av_10_fig1_1e9s_short_2x2_all.pdf', transparent=True, format = 'pdf')

plt.show()
