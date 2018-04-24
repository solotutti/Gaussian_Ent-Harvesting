# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:53:53 2018

@author: Admin
"""
import numpy as np
#import scipy.integrate as integ
import matplotlib.pyplot as plt
import warnings
from mpmath import mp
from mpmath import fp
import os

#%% FUNCTIONS

# BTZ
def gammam_btz_n(s,n,R,rh,l,deltaphi,eps):
    Zm = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1))
    return Zm - mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * (s-fp.j*eps))

def gammap_btz_n(s,n,R,rh,l,deltaphi):
    Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1))
    return Zp - fp.mpf(mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * s))

def integrand_Pdotm_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi,eps):
    K = lam**2*rh/(2*fp.pi*fp.sqrt(2)*l*fp.sqrt(R**2-rh**2)) * fp.exp(-tau**2/2/sig**2)
    Zm = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1))
    if Zm == fp.mpf(mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * s)):
        return 0
    else:
        return K * fp.exp(-(tau-s)**2/2/sig**2) * fp.exp(-fp.j*Om*s) \
                 / fp.sqrt(gammam_btz_n(s,n,R,rh,l,deltaphi,eps))

def integrand_Pdotp_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    K = lam**2*rh/(2*fp.pi*fp.sqrt(2)*l*fp.sqrt(R**2-rh**2)) * fp.exp(-tau**2/2/sig**2)
    Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1))
    if Zp == fp.mpf(mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * s)):
        return 0
    else:
        return K * fp.exp(-(tau-s)**2/2/sig**2) * fp.exp(-fp.j*Om*s) \
                 / fp.sqrt(gammap_btz_n(s,n,R,rh,l,deltaphi))

def Pdot_n(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0,eps=1e-8):
    if n == 0:
        if deltaphi == 0:
            lim2 = fp.mpf(mp.acosh(mp.mpf(rh**2/(R**2-rh**2)\
                                          *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1))))
            #print(lim2)
            fmins = lambda s: integrand_Pdotm_n(s,tau,0,R,rh,l,pm1,Om,lam,sig,deltaphi,eps).real
            fplus = lambda s: integrand_Pdotp_n(s,tau,0,R,rh,l,pm1,Om,lam,sig,deltaphi).real
            return fp.quad(fmins,[0,tau+10*sig]) + fp.quad(fplus,[0,lim2]) + fp.quad(fplus,[lim2,tau+10*sig])
        else:
            lim1 = fp.mpf(mp.acosh(mp.mpf(rh**2/(R**2-rh**2)\
                                          *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1))))
            lim2 = fp.mpf(mp.acosh(mp.mpf(rh**2/(R**2-rh**2)\
                                          *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1))))
            fmins = lambda s: integrand_Pdotm_n(s,tau,0,R,rh,l,pm1,Om,lam,sig,deltaphi,0).real
            fplus = lambda s: integrand_Pdotp_n(s,tau,0,R,rh,l,pm1,Om,lam,sig,deltaphi).real
            return fp.quad(fmins,[0,lim1]) + fp.quad(fmins,[lim1,tau+10*sig])\
                   + fp.quad(fplus,[0,lim2]) + fp.quad(fplus,[lim2,tau+10*sig])
    else:
        lim1 = fp.mpf(mp.acosh(mp.mpf(rh**2/(R**2-rh**2)\
                                      *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1))))
        lim2 = fp.mpf(mp.acosh(mp.mpf(rh**2/(R**2-rh**2)\
                                      *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1))))
        fmins = lambda s: integrand_Pdotm_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi,0).real
        fplus = lambda s: integrand_Pdotp_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi).real
        return fp.quad(fmins,[0,lim1]) + fp.quad(fmins,[lim1,tau+10*sig])\
               + fp.quad(fplus,[0,lim2]) + fp.quad(fplus,[lim2,tau+10*sig])

# GEON
def integrand_deltaPdot_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    Zm = mp.mpf(rh**2/(R**2-rh**2)\
                *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*(n+1/2))) - 1))
    Zp = mp.mpf(rh**2/(R**2-rh**2)\
                *(R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*(n+1/2))) + 1))
    K = lam**2*rh/(2*fp.pi*fp.sqrt(2)*l*fp.sqrt(R**2-rh**2)) * fp.exp(-tau**2/2/sig**2)
    return K * fp.exp(-(tau-s)**2/2/sig**2) * fp.exp(-fp.j*Om*s)\
             * ( 1/fp.sqrt(Zm + mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * s))\
                - pm1*1/fp.sqrt(Zp + mp.cosh(rh/l/mp.sqrt(R**2-rh**2) * s)) )

def DeltaPdot_n(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    f = lambda s: integrand_deltaPdot_n(s,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi)
    return fp.quad(f, [0,tau+10*sig])
    

#%% 

Om = 1.
lam = 1.
sig = 1.
M = 1.
l = 10*sig
rh = np.sqrt(M)*l
pm1 = 1.

dR = 1. #np.linspace(0.1, 3,num=50)
R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)

eps = 1e-6

#%% check integrand
tau = 2
s = np.linspace(-1,10,num=1000)
integrandminus_re = []
integrandminus_im = []

n = 1
print('>>> s = ',end='')
counter = 0
for si in s:
    if counter%10 == 0:
        print('%.2f'%si,end=', ',flush=True)
    counter += 1
    value = integrand_Pdotm_n(si,tau,n,R,rh,l,pm1,Om,lam,sig,0,0)
    integrandminus_re.append(value.real)
    integrandminus_im.append(value.imag)

plt.plot(s,integrandminus_re,label='real')
#plt.plot(s,integrandminus_im,label='imag')
plt.legend()
plt.xlabel('s')
plt.ylabel('integrand')
plt.title('n=0')

f = lambda x : integrand_Pdotm_n(x,tau,n,R,rh,l,pm1,Om,lam,sig,0,0)
print('\nintegral is: ',fp.quad(f,[0,tau+10*sig]))
#%% check 
nmax = 2

tau = np.linspace(-1,5,num=50)
tr_rate = 0*tau
tr_rate_geon = 0*tau

for n in range(nmax+1):
    print('n = ',n)
    print('i = ',end='')
    for i in range(len(tau)):
        print(i,end=', ',flush=True)
        if n == 0:
            #print(Pdot_n(tau[i],n,R,rh,l,pm1,Om,lam,sig))
            tr_rate[i] += Pdot_n(tau[i],n,R,rh,l,pm1,Om,lam,sig)
            tr_rate_geon[i] += DeltaPdot_n(tau[i],n,R,rh,l,pm1,Om,lam,sig)
        else:
            tr_rate[i] += 2*Pdot_n(tau[i],n,R,rh,l,pm1,Om,lam,sig)
            tr_rate_geon[i] += 2*DeltaPdot_n(tau[i],n,R,rh,l,pm1,Om,lam,sig)
    print('')
tr_rate_geon += tr_rate

plt.plot(tau,tr_rate,label='BTZ')
plt.plot(tau,tr_rate_geon,label='Geon')
plt.xlabel(r'$\tau$')
plt.ylabel('Transtion Rate')
plt.legend()
