# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:12:13 2018

@author: Admin
"""

#%% FUNCTIONS

import numpy as np
from mpmath import mp
from mpmath import fp
import matplotlib.pyplot as plt

def T_P_int_geon(x,tau,tau0,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    K = lam**2/(np.sqrt(2)*np.pi*l*Om) * rh/np.sqrt(R**2-rh**2)
    Gm = rh**2/(R**2-rh**2) * (R**2/rh**2*fp.mpf(mp.cosh(rh/l * (deltaphi - 2*np.pi*(n+0.5)))) - 1)
    Gp = rh**2/(R**2-rh**2) * (R**2/rh**2*fp.mpf(mp.cosh(rh/l * (deltaphi - 2*np.pi*(n+0.5)))) + 1)
    
    if tau < tau0:
        return 0
    elif x < (tau+tau0):
        return K * fp.sin(Om*(2*tau0+x))\
                * (1/fp.sqrt(Gm + fp.mpf(mp.cosh(rh/l**2 * x))) - 1/fp.sqrt(Gp + fp.mpf(mp.cosh(rh/l**2 * x))))
    else:
        return K * fp.sin(Om*(2*tau-x))\
                * (1/fp.sqrt(Gm + fp.mpf(mp.cosh(rh/l**2 * x))) - 1/fp.sqrt(Gp + fp.mpf(mp.cosh(rh/l**2 * x))))

def T_PGEON_n(tau,tau0,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return fp.quad(lambda x: T_P_int_geon(x,tau,tau0,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0), [2*tau0,2*tau])

#%%
sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
Om = 5        # Omega
nmax = 2            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
lam = 1             # coupling constant
M = 1
rh = np.sqrt(M)*l
dR = 1
R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
lowlim, uplim = -5,5
tau = np.linspace(lowlim,uplim,num=200)
tau0 = -5
y = 0*tau

print("i=")
for i in range(len(tau)):
    print(i,end=', ',flush=True)
    for n in range(nmax+1):
        if n == 0:
            y[i] += T_PGEON_n(tau[i],tau0,0,R,rh,l,pm1,Om,lam,sig)
        else:
            y[i] += 2*T_PGEON_n(tau[i],tau0,0,R,rh,l,pm1,Om,lam,sig)

fig = plt.figure(figsize=(9,5))
plt.plot(tau,y)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$P(\tau)$')
