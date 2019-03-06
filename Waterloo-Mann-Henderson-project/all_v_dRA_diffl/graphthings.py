# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 23:28:09 2018
            RUN COSMOLOGICAL CONSTANT GRAPHS
@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from mpmath import fp

#%% plot P or X or C for different l

dRA = np.load('dRA.npy')

base = 'cb_E001_M001_l'
l100 = np.load(base+'100.npy')
l20 = np.load(base+'20.npy')
l10 = np.load(base+'10.npy')
l8 = np.load(base+'8.npy')
l7 = np.load(base+'7.npy')
l6 = np.load(base+'6.npy')
l5 = np.load(base+'5.npy')
l1 = np.load(base+'1.npy')
#l1 = [fp.mpf(x) for x in l1]

plt.figure(figsize=(8,5))
plt.plot(dRA,l100,'r',label=r'$l=100\sigma$')
plt.plot(dRA,l20,'r--',label=r'$l=20\sigma$')
plt.plot(dRA,l10,'orange',label=r'$l=10\sigma$')
plt.plot(dRA,l8,'g',label=r'$l=8\sigma$')
plt.plot(dRA,l7,'c',label=r'$l=7\sigma$')
plt.plot(dRA,l6,'k',label=r'$l=6\sigma$')
plt.plot(dRA,l5,'y',label=r'$l=5\sigma$')
#plt.plot(dRA,l1,'b',label=r'$l=1\sigma$')
plt.legend(loc=8)
plt.xlabel(r'$dRA$')
plt.ylabel(r'$C_{BTZ}/\lambda^2$')
plt.title(r'$\Omega\sigma = 0.01$, $M = 0.01$')

#%% Compare b and g

dRA = np.load('dRA.npy')

###
base = '_E001_M01_l100.npy'
pb = np.load('pb'+base)
pg = np.load('pg'+base)
xb = np.load('xb'+base)
xg = np.load('xg'+base)
cb = np.load('cb'+base)
cg = np.load('cg'+base)

plt.figure()
plt.plot(dRA,pb,'r',label=r'$P_{BTZ}/\lambda^2$')
plt.plot(dRA,pb+pg,'r:',label=r'$P_{Geon}/\lambda^2$')
plt.plot(dRA,xb,'b',label=r'$|X|_{BTZ}/\lambda^2$')
plt.plot(dRA,xg,'b:',label=r'$|X|_{Geon}/\lambda^2$')
plt.plot(dRA,cb,'g',label=r'$C_{BTZ}/\lambda^2$')
plt.plot(dRA,cg,'g:',label=r'$C_{Geon}/\lambda^2$')
plt.legend(loc=1)
plt.xlabel(r'$dRA$')
plt.title(r'$\Omega\sigma = 0.01$, $M = 0.1$, $l/\sigma=100$')

###
base = '_E1_M01_l10.npy'
pb = np.load('pb'+base)
pg = np.load('pg'+base)
xb = np.load('xb'+base)
xg = np.load('xg'+base)
cb = np.load('cb'+base)
cg = np.load('cg'+base)

plt.figure()
plt.plot(dRA,pb,'r',label=r'$P_{BTZ}/\lambda^2$')
plt.plot(dRA,pb+pg,'r:',label=r'$P_{Geon}/\lambda^2$')
plt.plot(dRA,xb,'b',label=r'$|X|_{BTZ}/\lambda^2$')
plt.plot(dRA,xg,'b:',label=r'$|X|_{Geon}/\lambda^2$')
plt.plot(dRA,cb,'g',label=r'$C_{BTZ}/\lambda^2$')
plt.plot(dRA,cg,'g:',label=r'$C_{Geon}/\lambda^2$')
plt.legend(loc=1)
plt.xlabel(r'$dRA$')
plt.title(r'$\Omega\sigma = 1$, $M = 0.1$, $l/\sigma=10$')

###
base = '_E001_M01_l5.npy'
pb = np.load('pb'+base)
pg = np.load('pg'+base)
xb = np.load('xb'+base)
xg = np.load('xg'+base)
cb = np.load('cb'+base)
cg = np.load('cg'+base)

plt.figure()
plt.plot(dRA,pb,'r',label=r'$P_{BTZ}/\lambda^2$')
plt.plot(dRA,pb+pg,'r:',label=r'$P_{Geon}/\lambda^2$')
plt.plot(dRA,xb,'b',label=r'$|X|_{BTZ}/\lambda^2$')
plt.plot(dRA,xg,'b:',label=r'$|X|_{Geon}/\lambda^2$')
plt.plot(dRA,cb,'g',label=r'$C_{BTZ}/\lambda^2$')
plt.plot(dRA,cg,'g:',label=r'$C_{Geon}/\lambda^2$')
plt.legend(loc=1)
plt.xlabel(r'$dRA$')
plt.title(r'$\Omega\sigma = 0.01$, $M = 0.1$, $l/\sigma=5$')

###
base = '_E1_M01_l1.npy'
pb = np.load('pb'+base)
pg = np.load('pg'+base)
xb = np.load('xb'+base)
xg = np.load('xg'+base)
cb = np.load('cb'+base)
cg = np.load('cg'+base)

plt.figure()
plt.plot(dRA,pb,'r',label=r'$P_{BTZ}/\lambda^2$')
plt.plot(dRA,pb+pg,'r:',label=r'$P_{Geon}/\lambda^2$')
plt.plot(dRA,xb,'b',label=r'$|X|_{BTZ}/\lambda^2$')
plt.plot(dRA,xg,'b:',label=r'$|X|_{Geon}/\lambda^2$')
plt.plot(dRA,cb,'g',label=r'$C_{BTZ}/\lambda^2$')
plt.plot(dRA,cg,'g:',label=r'$C_{Geon}/\lambda^2$')
plt.legend(loc=1)
plt.xlabel(r'$dRA$')
plt.title(r'$\Omega\sigma = 1$, $M = 0.1$, $l/\sigma=1$')
