# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:52:03 2018
    Compute concurrence for geon
@author: Admin
"""
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
import warnings
from mpmath import mp
from mpmath import fp


#%%===========================================================================#
#======================== BTZ TRANSITION PROBABILITY =========================#
#=============================================================================#

### Using Laura's formula

# First integrand 
def f01(y,n,R,rh,l,pm1,Om,lam,sig):
    return lam**2*sig**2/2 * fp.exp(-sig**2*(y-Om)**2)\
    # / fp.mpf((mp.exp(y * 2*mp.pi*l*mp.sqrt(R**2-rh**2)/rh) + 1))

# Second integrand
def f02(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = mp.mpf((R**2+rh**2)/(R**2-rh**2))
    if Zp == mp.cosh(y):
        print("RIP MOM PLSSS")
    #print(Zp, y, fp.cosh(y))
    if Zp - mp.cosh(y) > 0:
        return K * fp.exp(-a*y**2) * fp.cos(b*y) / fp.mpf(mp.sqrt(Zp - mp.cosh(y)))
    elif Zp - mp.cosh(y) < 0:
        return -K * fp.exp(-a*y**2) * fp.sin(b*y) / fp.mpf(mp.sqrt(mp.cosh(y) - Zp))
    else:
        return 0

## First integrand in the sum
#def fn1(y,n,R,rh,l,pm1,Om,lam,sig):
#    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
#    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
#    b = fp.sqrt(R**2-rh**2)*Om*l/rh
#    Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) - 1)
#    if Zm - fp.cosh(y) > 0:
#        return K * fp.exp(-a*y**2) * fp.cos(b*y) / fp.sqrt(Zm - fp.cosh(y))
#    else:
#        return -K * fp.exp(-a*y**2) * fp.sin(b*y) / fp.sqrt(fp.cosh(y) - Zm)
#    
## Second integrand in the sum
#def fn2(y,n,R,rh,l,pm1,Om,lam,sig):
#    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
#    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
#    b = fp.sqrt(R**2-rh**2)*Om*l/rh
#    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) + 1)
#    if Zp - fp.cosh(y) > 0:
#        return K * fp.exp(-a*y**2) * fp.cos(b*y) / fp.sqrt(Zp - fp.cosh(y))
#    else:
#        return -K * fp.exp(-a*y**2) * fp.sin(b*y) / fp.sqrt(fp.cosh(y) - Zp)
    
def P_BTZn(n,R,rh,l,pm1,Om,lam,sig):
    b = fp.sqrt(R**2-rh**2)/l
    lim = 20*sig*rh/b/l**2
    print('limit: ',lim)
    Zm = fp.mpf(1) #rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) - 1)
    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) + 1)
    print('Zm: ',Zm, 'Zp: ',Zp, 'acosh(Zp): ', fp.mpf(mp.acosh(Zp)))
    #print(Zm,fp.acosh(Zm))
#    plt.figure()
#    xaxis = fp.linspace(0,lim/5,50)
#    yaxis = [f02(x,n,R,rh,l,pm1,Om,lam,sig) for x in xaxis]
#    plt.plot(xaxis,yaxis)
#    plt.show()
    if pm1==-1 or pm1==1 or pm1==0:
        if n==0:
            return 0\
                 - fp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[0,fp.mpf(mp.acosh(Zp))])\
                 - fp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[fp.mpf(mp.acosh(Zp)),lim])
                 #fp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-fp.inf,fp.inf])
#        else:
#            if fp.cosh(lim) < Zm or Zm < 1:
#                return fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
#            else:
#                return fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,fp.mpf(mp.acosh(Zm))])\
#                     - fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[fp.mpf(mp.acosh(Zm)),lim])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#%%
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
nmax = 0            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant
Om = 0.62020408163265306

def diff_X_PAPB(dRA):
    RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
    #RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
    #                 + RA - mp.sqrt(RA**2-rh**2))
#    Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#    Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig)
    print('dRA IS: ',dRA,end= '-------')
    PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig)
   
#    PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig)
    
#    for n in range(1,nmax):
#        Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)
#        Xim += 2*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)
#        PA += 2*P_BTZn(n,RA,rh,l,pm1,Om,lam,sig)
#        PB += 2*P_BTZn(n,RB,rh,l,pm1,Om,lam,sig)
    return PA #Xre**2+Xim**2 - PA*PB

dRA = np.linspace(0.22,0.28,num=300)
diffxpapb = []
print('')
for i in range(np.size(dRA)):
    print(i)
    diffxpapb.append(diff_X_PAPB(dRA[i]))
plt.figure()
plt.plot(dRA,diffxpapb)
#plt.xticks([0.250392886,0.250392887,0.250392888])