# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:04:33 2018
                 CHECK SUMMING OF X_GEON (GAussian)
@author: Admin
"""
import numpy as np
#import scipy.integrate as integ
import matplotlib.pyplot as plt
import warnings
from mpmath import mp
from mpmath import fp
import os

#%%

# Denominators of wightmann functions
def XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*(n+1/2))) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*(n+1/2))) + 1)
                        # Z-plus
    return 1/mp.sqrt(Zm + mp.cosh(y)) - pm1/mp.sqrt(Zp + mp.cosh(y))


def XGEON_integrand_nBA(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = 1
    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2
    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2)
    E = (bB-bA)/fp.sqrt(2)/fp.sqrt(bB**2+bA**2) * ( (bB+bA)*y/2/sig + fp.j*sig*Om)
    
    return K*mp.exp(-alp2*y**2)*mp.exp(-fp.j*bet2*Om*y) *fp.erfc(E) \
        * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

#def XGEON_integrand_nAB(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
#    bA = mp.sqrt(RA**2-rh**2)/l
#    bB = mp.sqrt(RB**2-rh**2)/l
#    K = 1
#    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2
#    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2)
#    
#    return K*mp.exp(-alp2*y**2)*mp.cos(bet2*Om*y) * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

def XGEON_nBA(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return -mp.quad(lambda y: XGEON_integrand_nBA(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi), [-fp.inf, fp.inf])

#%%
    
sig = 1             # width of Gaussian
M = 0.1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
Om = 1          # Omega

#sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dR = np.linspace(0.1, 3,num=5)
                    # proper distance of the closest detector

RA = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))

print('Truncation after n=2...')
Xre = 0 * dR
Xim = 0 * dR
nmax = 2
for n in range(-nmax,nmax+1):
    print('n = ', n)
    print('i =',end=' ')
    for i in range(len(dR)):
        print(i, end=', ',flush=True)
        val = XGEON_nBA(n,RA[i],RB[i],rh,l,pm1,Om,lam,sig) + XGEON_nBA(-n,RB[i],RA[i],rh,l,pm1,Om,lam,sig)
        print(val)
        Xre[i] += val.real
        Xim[i] += val.imag
    print('')

print('Truncation a la Su Yu...')
Xre2 = 0 * dR
Xim2 = 0 * dR
nmax = 2
for n in range(-nmax,nmax+1):
    print('n = ', n)
    print('i =',end=' ')
    for i in range(len(dR)):
        print(i, end=', ',flush=True)
        val = XGEON_nBA(n,RA[i],RB[i],rh,l,pm1,Om,lam,sig) + XGEON_nBA(-n-1,RB[i],RA[i],rh,l,pm1,Om,lam,sig)
        print(val)
        Xre2[i] += val.real
        Xim2[i] += val.imag
    print('')

print('!!!Real Part!!!')
print(Xre,Xre2)
print('\n!!!Imaginary Part!!!')
print(Xim,Xim2)
