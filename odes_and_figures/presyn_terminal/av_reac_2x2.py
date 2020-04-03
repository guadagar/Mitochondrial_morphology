#!/usr/bin/env python
import PyDSTool
from time import clock
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import FormatStrFormatter,ScalarFormatter

'''This script runs the ODE system of the third in silico experiment
A comparison with the averaged traces of the MCell simulations is done afterwards.
'''

params = {'axes.labelsize': 7,
           'axes.titlesize': 6,
          'legend.fontsize': 6,
           'xtick.labelsize': 6,
           'ytick.labelsize': 6,
            'figure.figsize': (3.2,2.9)}
            #'figure.figsize': (6.4, 6)}
mpl.rcParams.update(params)


Na = 6.02214e23
#vmito = 0.033221e-15 #lt
#vcyto = 0.232549e-15 #
vmito = 0.020706e-15 #lt OM
vcyto = 0.0181e-15 # IM
vcube = 0.0923e-15

no1_ant = 20e3
k7 = 800.0 #0.74
k8 = 0.7 #0.05
k9 = 0.58 #0.37
k10 = 0.48#0.47
no_atp = 3800 #number of atphase
cm = 3e3 #nr.of particles = 12mM #adp+atp = cm
a12 = 24
a21 = 40.0
a23 = 4.0
a32 = 5e3
ac = 9806 #clamp*Na*vcyto
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
		'a12':'24', #s-1
		'a21':'40.0',#s-1
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
		'a32':'5e3',          #s-1
		'a23':'4.0',         #uM-1s-1
		'no1_atp': no_atp, #number of atphase
		'vcyto': vcyto,
		'vcube':vcube,#syn -vmito
		'vmito':vmito,
		'kp':kp,
		'n_porin': n_porin,
		'n_cha':'7e5',
		'k_cha':'1.0',
		#'k_cha_off':'0',
        'k_cha_off':'2e-2',
		'Na':Na,
		'fa':'0.5',
		'ac':ac,
		't_on1':'0.001',
		't_off1':'0.005',
		't_on2':'0.010',
		't_off2':'0.014'}

#DSargs.fnspecs  = {'alb': (['l','al','lb','bl','bla','la','ala','blb','alap','blbp'], 'no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp') }
DSargs.fnspecs = {'k_cha_t': (['t'], "(heav(t>=t_on1)*heav(t<t_off1)+heav(t>t_on2)*heav(t<t_off2))*(k_cha-k_cha_off)+k_cha_off")}
#DSargs.fnspecs = {'k_cha_t': (['t'], "(heav(t>=t_on1))*k_cha_off")}

DSargs.varspecs = { 'am':'-k6_on*(1e6*am/(Na*vmito))*bl + k6_off*bla - k6_on*(1e6*am/(Na*vmito))*l + k6_off*la - 2.0*k6_on*fa*(1e6*am/(Na*vmito))*al + k6_off*ala + k6_off*alap + a34*h3es - a43*(1e6*am/(Na*vmito))*h3eo',
                    'bm':'-k2_on*(1e6*bm/(Na*vmito))*al + k2_off*(no_ant-l-al-la-lb-bl-bla-ala-blb-alap-blbp) - k2_on*(1e6*bm/(Na*vmito))*l + k2_off*lb - 2.0*k2_on*fa*(1e6*bm/(Na*vmito))*bl + k2_off*blb + k2_off*blbp -a23*h3e*(1e6*bm/(Na*vmito)) + a32*h3es',
                    'bo':'-kp*(1e6*bo/(Na*vcube))*n_porin + kp*(1e6*bc/(Na*vcyto))*n_porin - n_cha*k_cha_t(t)*(1e6*bo/(Na*vcube))',
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
DSargs.ics  = {'am':37739,
	           'bm':500,
	           'bc':200,
	           'bo':0,
                'l':500,
                'al':14.5e3,
                'lb':0,
                'bl':300,
                'bla':100,
                'la':140,
                'ala':2.2e3,
                'blb':0,
                'alap':2.2e3,
                'blbp':0,
                'eo':1000,
                'ei':20,
                'h3eo':33,
                'h3es':80,
                'h3e':1940}
DSargs.tdomain = [0,0.014]

#ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)    # an instance of the 'Generator' class.
ode  = PyDSTool.Generator.Radau_ODEsystem(DSargs)
start = clock()
traj = ode.compute('polarization')
print ('  ... finished in %.3f seconds.\n' % (clock()-start))
pd   = traj.sample()

#------------average traces are loaded -----------------------

with open('./av_10c_1e9', 'rb') as f: av_ar = pickle.load(f)

with open('./av_10r_1e9', 'rb') as f: av_ra = pickle.load(f)

with open('./av_10rc_1e9', 'rb') as f: av_cu = pickle.load(f)
time = np.loadtxt('./time')

#---------

plt.figure(1)

ax = plt.subplot(4,2,2)
plt.plot(1e3*time-2.5,av_ar[0,:], 'b')
plt.plot(1e3*time-2.5,av_ra[0,:], 'red')
plt.plot(1e3*time-2.5,av_cu[0,:], 'm')
plt.plot(1e3*pd['t']-2.5, ac*np.ones(len(pd['t'])),c='k')
ax.locator_params(axis='y',nbins=3)
plt.ylabel(r'#ADP$_{\rm outside}$')
ax.locator_params(axis='x',nbins=8)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) #con 10 como base
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #notacion cientifica
ax.set_yticks([0.94e4,0.97e4,1e4])
#plt.text(1,29650,'A', ha='center', va='center', fontsize=11)
#plt.ylim(28000,30000)
plt.xlim(0,10)
#plt.arrow(1, 2.8e4, 0, 150,lw= 5, fc='k', ec='k')
#ax.xaxis.set_major_formatter(plt.NullFormatter())

ax2 = plt.subplot(4,2,4)
plt.plot(1e3*time-2.5,av_ar[1,:],'b')
plt.plot(1e3*time-2.5,av_ra[1,:],'red')
plt.plot(1e3*time-2.5,av_cu[1,:],'m')
plt.plot(1e3*pd['t']-2.5, pd['am'],c='k')
ax2.locator_params(axis='y',nbins=3)
ax2.locator_params(axis='x',nbins=8)
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.text(1,68300,'C', ha='center', va='center', fontsize=11)
#plt.arrow(1, 67500, 0, 72,lw= 5, fc='k', ec='k')
plt.xlim(0,10)
ax2.set_yticks([3.75e4,3.80e4,3.85e4])
plt.ylabel(r'#ADP$_{\rm matrix}$')
#ax2.xaxis.set_major_formatter(plt.NullFormatter())

ax3 = plt.subplot(4,2,5)
plt.plot(1e3*time-2.5,av_ar[2,:], 'b')
plt.plot(1e3*time-2.5,av_ra[2,:], 'red')
plt.plot(1e3*time-2.5,av_cu[2,:], 'm')
plt.plot(1e3*pd['t']-2.5,pd['bm'],c='k')
#plt.text(1,1235,'D', ha='center', va='center', fontsize=11)
ax3.locator_params(axis='y',nbins=2)
ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim(0,10)
ax3.set_yticks([4.5e2,5e2,5.5e2])
plt.ylabel(r'#ATP$_{\rm matrix}$')
#plt.xlabel('Time (msec)')
#ax3.xaxis.set_major_formatter(plt.NullFormatter())
ax3.set_xticks([0,2,4,6,8,10])

ax1 = plt.subplot(4,2,6)

plt.plot(1e3*time-2.5,av_ar[3,:], 'b')
plt.plot(1e3*time-2.5,av_ra[3,:], 'red')
plt.plot(1e3*time-2.5,av_cu[3,:], 'm')
plt.plot(1e3*pd['t']-2.5, pd['bc'],c='k')
ax1.locator_params(axis='y',nbins=3)
plt.ylabel(r'#ATP$_{\rm outside}$')
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax1.set_yticks([0e3,3e3,6e3])
#plt.text(1,650,'B', ha='center', va='center', fontsize=11)
plt.xlim(0, 10)
#ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_yticks([0e2,2e2,4e2])
ax1.set_xticks([0,2,4,6,8,10])
#plt.xlabel('Time (msec)')

ax4 = plt.subplot(4,1,4)

plt.plot(1e3*time-2.5,av_ar[29,:],'b')
plt.plot(1e3*time-2.5,av_ra[29,:],'red')
plt.plot(1e3*time-2.5,av_cu[29,:],'m')
plt.plot(1e3*pd['t']-2.5,pd['bo'],'k')
#plt.text(0.5,120,'E', ha='center', va='center', fontsize=11)#
#plt.xlabel('Time (msec)')
plt.ylabel(r'#ATP$_{\rm syn}$')
plt.subplots_adjust(wspace =0.5)
ax4.locator_params(axis='y',nbins=2)
#ax4.xaxis.set_major_formatter(plt.NullFormatter())
ax4.set_yticks([0,1.1e2,2.2e2])
ax4.set_xticks([0,2,4,6,8,10])
plt.ylim(0,2.2e2)
plt.xlim(0,10)

plt.tight_layout(pad=0.05, w_pad=0.001)
#plt.savefig('syn2_av_10_3x2n.pdf', transparent=True, format = 'pdf',dpi=600,bbox_inches='tight')

plt.show()
