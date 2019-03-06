# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:52:03 2018
    Compute concurrence for geon
@author: Admin
"""
import numpy as np
#import scipy.integrate as integ
import matplotlib.pyplot as plt
import warnings
from mpmath import mp
from mpmath import fp
import os


#%%===========================================================================#
#======================== BTZ TRANSITION PROBABILITY =========================#
#=============================================================================#

### Using Laura's formula

# First integrand 
def f01(y,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    if deltaphi==0:
        return lam**2*sig**2/2 * fp.exp(-sig**2*(y-Om)**2)\
         / fp.mpf((mp.exp(y * 2*mp.pi*l*mp.sqrt(R**2-rh**2)/rh) + 1))
    else:
        K = lam**2*sig/2/fp.sqrt(2*fp.pi)
        a = (R**2-rh**2)*l**2/4/sig**2/rh**2
        b = fp.sqrt(R**2-rh**2)*Om*l/rh
        Zm = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l *deltaphi) - 1))
        if Zm == mp.cosh(y):
            return 0
        elif Zm - fp.mpf(mp.cosh(y)) > 0:
            return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zm - mp.cosh(y)))
        else:
            return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zm)) * fp.sign(y)

# Second integrand new
def f02(y,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    if deltaphi==0:
        Zp = mp.mpf((R**2+rh**2)/(R**2-rh**2))
    else:
        Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l *(deltaphi -  2*fp.pi*n)) + 1))
    if Zp - mp.cosh(y) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y)))
    elif Zp - mp.cosh(y) < 0:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp)) * fp.sign(y)
    else:
        return 0

def fn1(y,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zm = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l *(deltaphi -  2*fp.pi*n)) - 1))
    if Zm == mp.cosh(y):
        return 0
    elif Zm - fp.mpf(mp.cosh(y)) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zm - mp.cosh(y)))
    else:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zm)) * fp.sign(y)

def fn2(y,n,R,rh,l,pm1,Om,lam,sig,deltaphi):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l *(deltaphi -  2*fp.pi*n)) + 1))
    if Zp == mp.cosh(y):
        return 0
    elif Zp - fp.mpf(mp.cosh(y)) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y)))
    else:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp)) * fp.sign(y)
    
def P_BTZn(n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    b = fp.sqrt(R**2-rh**2)/l
    lim = 20*sig*rh/b/l**2
    Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(rh/l *(deltaphi -  2*fp.pi*n)) + 1))
    #print('Zm: ',Zm, 'Zp: ',Zp)
    #print(Zm,fp.acosh(Zm))
    if pm1==-1 or pm1==1 or pm1==0:
        if n==0:
            return fp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[-fp.inf,fp.inf])\
                 - fp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[0,fp.mpf(mp.acosh(Zp))])\
                 - fp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[fp.mpf(mp.acosh(Zp)),lim])
        else:
            Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) - 1)
            if fp.mpf(mp.cosh(lim)) < Zm or Zm < 1:
                return 2*fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[0,lim])
            else:
                return 2*fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[0,fp.mpf(mp.acosh(Zm))])\
                     - 2*fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig,deltaphi),[fp.mpf(mp.acosh(Zm)),lim])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#================= GEON ADDITION TO TRANSITION PROBABILITY ===================#
#=============================================================================#

def PGEON_gaussian(x,sig):
    return fp.mpf(mp.exp(-x**2/4/sig**2))

def sigma_geon(x,n,R,rh,l,deltaphi):
    return R**2/rh**2 * fp.mpf(mp.cosh( rh/l * (deltaphi - 2*mp.pi *(n+1/2))))\
     - 1 + (R**2-rh**2)/rh**2 * fp.mpf(mp.cosh(rh/l/fp.sqrt(R**2 - rh**2) * x))
    
def h_n(x,n,R,rh,l,pm1,deltaphi):
    return 1/(4*fp.sqrt(2)*fp.pi*l) * (1/fp.sqrt(sigma_geon(x,n,R,rh,l,deltaphi)) \
              - pm1 * 1/fp.sqrt(sigma_geon(x,n,R,rh,l,deltaphi) + 2))

def PGEON_n(n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    """
    Om = energy difference
    lam = coupling constant
    """
    if pm1==-1 or pm1==1 or pm1==0:
        return lam**2*sig*fp.sqrt(fp.pi) * fp.mpf(mp.exp(-sig**2 * Om**2)) *\
        fp.quad(lambda x: h_n(x,n,R,rh,l,pm1,deltaphi) * PGEON_gaussian(x,sig), [-fp.inf, fp.inf])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#=========================== BTZ MATRIX ELEMENT X ============================#
#=============================================================================#

### Using Laura's formula

# Denominators of wightmann functions 
def XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) + 1)
                        # Z-plus
    if Zm < mp.cosh(y) and Zp < mp.cosh(y):
        return 0
    elif Zm < mp.cosh(y):
        return - pm1/mp.sqrt(Zp - mp.cosh(y))
    else:
        return 1/mp.sqrt(Zm - mp.cosh(y)) - pm1/mp.sqrt(Zp - mp.cosh(y))

def XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) + 1)
                        # Z-plus
    #print(Zm-np.cosh(y),Zp-np.cosh(y))
    if Zm < mp.cosh(y) and Zp < mp.cosh(y):
        return - ( 1/mp.sqrt(mp.cosh(y) - Zm) - pm1/mp.sqrt(mp.cosh(y) - Zp) ) * (-fp.sign(y))
    elif Zm < mp.cosh(y):
        return - 1/mp.sqrt(mp.cosh(y) - Zm) * (-fp.sign(y))
    else:
        return 0

# gaussian exponential multiplied by 
def XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*mp.exp(-alp*y**2)*mp.cos(bet*y) * XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

def XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*mp.exp(-alp*y**2)*mp.cos(bet*y) * XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

# Integrate everything
def XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    print('\nn=',n,'\nRA=',RA,'\nRB=',RB,'\nrh=',rh,'\nl=',l,
#          '\npm1=',pm1,'\nOm=',Om,'\nlam=',lam,'\nsig=',sig)
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) + 1)
                        # Z-plus
    uplim = 10*sig*mp.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > mp.acosh(Zp):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zm), mp.acosh(Zp)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zp), uplim])
        return -integral
    elif uplim > mp.acosh(Zm):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zm), uplim])
        return -integral
    else:
        return -mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [-uplim, uplim])/2

def XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=0):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*n)) + 1)
                        # Z-plus
    uplim = 10*sig*mp.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > mp.acosh(Zp):
        integral = mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zm), mp.acosh(Zp)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zp), uplim])
        return -integral
    elif uplim > mp.acosh(Zm):
        integral = mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [mp.acosh(Zm), uplim])
        return -integral
    else:
        return -mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi),\
                           [-uplim, uplim])/2

#=============================================================================#
#=========================== Geon MATRIX ELEMENT X ===========================#
#=============================================================================#
    
# Denominators of wightmann functions
def XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*(n+1/2))) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(rh/l * (deltaphi - 2*mp.pi*(n+1/2))) + 1)
                        # Z-plus
    return 1/mp.sqrt(Zm + mp.cosh(y)) - pm1/mp.sqrt(Zp + mp.cosh(y))

#def XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
#    bA = mp.sqrt(RA**2-rh**2)/l
#    bB = mp.sqrt(RB**2-rh**2)/l
#    K = lam**2*rh*sig/4/l**2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
#         * mp.exp(-(bA-bB)**2/2/(bA**2+bB**2)*sig**2*Om**2)
#    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2
#    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2)
#    
#    return K*mp.exp(-alp2*y**2)*mp.cos(bet2*Om*y) * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

def XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-(bA-bB)**2/2/(bA**2+bB**2)*sig**2*Om**2)
    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2 * l**4/rh**2
    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2) * l**2/rh
    
    return K*mp.exp(-alp2*y**2)*mp.cos(bet2*Om*y) * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi)

def XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return -mp.quad(lambda y: XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi), [0, np.inf])

#=============================================================================#
#================================== D_Death ==================================#
#=============================================================================#
    
def find_zero(f,p1,d):
    """
    Return the zero 'c' of a monotonically increasing or decreasing function f.
    p1 is an estimate of where the zero might be.
    d is the accuracy of the zero.
    """
    if p1 > 10:
        p1 = 0.5
    c = p1 - f(p1)*d/(f(p1+d)-f(p1)) # simple
    
    # we can do this in case derivative diverges because our function is symmetric
    if c < 0:
        c = -c
    
    global n_of_zeros
    
    if n_of_zeros > 8:
        print('Go linear')
        return find_zero_linear(f,d,c,d)
    
    n_of_zeros += 1
    print('#',n_of_zeros,' current zero: ',c)
    
    # if p1 and c are close together
    if np.abs(c-p1)<d or f(c)==0:
        return c
    else:
        return find_zero(f,c,d)

def find_zero_linear(f,p1,p2,d):
    
    # if both points are negative, extend p2 further
    if f(p1) < 0 and f(p2) < 0:
        return find_zero_linear(f,p1,p2+(p2-p1),d)
    
    if f(p1) > 0 and f(p2) > 0:
        return find_zero_linear(f,p1-(p2-p1),p2,d)
    
    c = (p1 + p2)/2
    
    global n_of_zeros
    n_of_zeros += 1
    print('#',n_of_zeros,' current zero: ',c)
    
    if f(c) == 0:
        return c
    
    if np.abs(p1-p2) < d:
        return c
    elif f(p1)*f(c) < 0:
        return find_zero_linear(f,p1,c,d)
    else:
        return find_zero_linear(f,c,p2,d)
        
   
def find_d_death_BTZ(nmax,sep,rh,l,pm1,Om,lam,sig,guess,prec,deltaphi=0):
    
    # Function whose zero we are looking for
    def diff_X_PAPB(dRA):
        RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
        RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
                         + RA - mp.sqrt(RA**2-rh**2))
        Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        
        for n in range(1,nmax):
            Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
            Xim += 2*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
            PA += 2*P_BTZn(n,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
            PB += 2*P_BTZn(n,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        return Xre**2+Xim**2 - PA*PB
    
    global n_of_zeros
    n_of_zeros = 0
    return find_zero(diff_X_PAPB,guess,prec) #find_zero_linear(diff_X_PAPB,prec,guess,prec)

def find_d_death_GEON(nmax,sep,rh,l,pm1,Om,lam,sig,guess,prec,deltaphi=0):
    
    # Function whose zero we are looking for
    def diff_X_PAPB(dRA):
        RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
        RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
                         + RA - mp.sqrt(RA**2-rh**2))
        Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + 2*XGEON_n(0,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + 2*PGEON_n(0,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + 2*PGEON_n(0,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
        
        for n in range(1,nmax+1):
            Xre += 2 * (XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi))
            Xim += 2 * XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi)
            PA += 2 * (P_BTZn(n,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + PGEON_n(n,RA,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi))
            PB += 2 * (P_BTZn(n,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi) + PGEON_n(n,RB,rh,l,pm1,Om,lam,sig,deltaphi=deltaphi))
        return Xre**2+Xim**2 - PA*PB
    
    global n_of_zeros
    n_of_zeros = 0
    return find_zero(diff_X_PAPB,guess,prec) #find_zero_linear(diff_X_PAPB,prec,guess,prec)

#=============================================================================#
#============================== Random Functions =============================#
#=============================================================================#

def relative_diff(arr1,arr2):
    try:
        return np.divide( np.abs(arr1-arr2) , (arr1+arr2)/2 )
    except ZeroDivisionError:
        answer = []
        for i in range(len(arr1)):
            if arr1[i]==0 and arr1[i]==0:
                answer.append(float('nan'))
            else:
                answer.append( np.abs(arr1[i]-arr2[i])*2/(arr1[i]+arr2[i]) )
        return np.array(answer)

def relative_diff_mpf(arr1,arr2):
    arr1, arr2 = np.array([np.float(x) for x in arr1]),\
                    np.array([np.float(x) for x in arr2])
    return np.divide( np.abs(arr1-arr2) , (arr1+arr2)/2 )

def converge_cutoff(mass,precision):
    return int(np.ceil(-np.log(precision)/3/np.pi/np.sqrt(mass)))

#%%===========================================================================#
#===================== COMPARE CONCURRENCE VS. DISTANCE ======================#
#=============================================================================#

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
DeltaE = 1        # Omega
nmax = 1            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dRA = np.linspace(0.1,3,num=60)
# Plot PA/lam^2 vs dRA from rh to 10rh
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector

PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
 = [], [], [], [], [], [], []
                    # Create y-axis array for BTZ and geon

# Start summing
for n in np.arange(0,nmax+1):
    print('n =',n)
    print(' i = ', end='')
    for i in range(len(RA)):
        print(i,end=' ',flush=True)
        if n == 0:
            PA_btz.append( P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
            PB_btz.append( P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
            X_btz_re.append( XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            X_btz_im.append( XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            
            PA_geon.append(2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig))#PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_geon.append(2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig))#PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_geon.append(2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))#XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        else:     # when summing nonzero n's, multiply by 2
            PA_btz[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_btz[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_re[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_im[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            
            PA_geon[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_geon[i] += 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_geon[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
    print('')

PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
 = np.array(PA_btz), np.array(PA_geon), np.array(PB_btz), np.array(PB_geon)\
 , np.array(X_btz_re), np.array(X_btz_im), np.array(X_geon)

X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2 )
conc_btz = 2*np.maximum(0,np.abs(X_btz) - np.sqrt(np.multiply(PA_btz,PB_btz)))

PA_geon += PA_btz
PB_geon += PB_btz
X_geon += X_btz_re       # add transition probability addition from geon to BTZ
X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
conc_geon = 2*np.maximum(0,np.abs(X_geon) - np.sqrt(np.multiply(PA_geon,PB_geon)))

# Plotting
fig = plt.figure(figsize=(9,5))

# btz
plt.plot(dRA,PA_btz,'c', label='PA BTZ', )
plt.plot(dRA,PB_btz,'orange', label='PB BTZ')
plt.plot(dRA,X_btz,'g', label='X BTZ')
plt.plot(dRA,conc_btz,'r', label='conc BTZ')

# geon
plt.plot(dRA,PA_geon,'c:', label='PA BTZ', )
plt.plot(dRA,PB_geon,color='orange',linestyle=':', label='PB BTZ')
plt.plot(dRA,X_geon,'g:', label='X BTZ')
plt.plot(dRA,conc_geon,'r:', label='conc BTZ')

plt.legend()
plt.xlabel(r'$d(r_h,R_A)$')
plt.title('Concurrence with proper distance')
#plt.xlim([0,6])
#plt.ylim([1.075,1.175])
#print('PA_btz: ', PA_btz)
#print('PB_btz: ', PB_btz)
#print('X_btz: ', X_btz)
plt.show()

#print('relative difference between concurrence',\
#      relative_diff_mpf(conc_btz,conc_geon))

#%%=========================================================================%%#
#======================== COMPARE PD FOR BTZ AND GEON ========================#
#=============================================================================#
dirname = "P_v_dR_btzgeon/"
if not os.path.exists(dirname):
    os.makedirs(dirname)

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
#DeltaE = 1          # Omega

l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

# Plot PA/lam^2 vs R from rh to 10rh

dR = np.linspace(0.1, 3,num=50)
                    # proper distance of the closest detector

R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
                    # Array for distance from horizon

nmax = converge_cutoff(M,0.0001)            # summation limit
print('nmax is',nmax)

def get_PvdRA_for_Om(DeltaE):
    P_btz, P_geon = [], []
                        # Create y-axis array for BTZ and geon
    for n in np.arange(0,nmax+1):
        print('n = ',n,'; i = ', end='')
        for i in range(np.size(dR)):
            print(i,end=' ',flush=True)
            if n == 0:
                P_btz.append(P_BTZn(n,R[i],rh,l,pm1,DeltaE,lam,sig))
                P_geon.append(2*PGEON_n(n,R[i],rh,l,pm1,DeltaE,lam,sig))
            else:
                P_btz[i] += 2*P_BTZn(n,R[i],rh,l,pm1,DeltaE,lam,sig)
                P_geon[i] += 2*PGEON_n(n,R[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    
    P_btz, P_geon = np.array(P_btz), np.array(P_geon)
    P_geon += P_btz       # add transition probability addition from geon to BTZ
    return P_btz, P_geon

Pbtz1, Pgeon1 = get_PvdRA_for_Om(1)
Pbtz01, Pgeon01 = get_PvdRA_for_Om(0.1)
Pbtz001, Pgeon001 = get_PvdRA_for_Om(0.01)
np.save(dirname+"Pbtz1.npy",Pbtz1)
np.save(dirname+"Pbtz01.npy",Pbtz01)
np.save(dirname+"Pbtz001.npy",Pbtz001)
np.save(dirname+"Pgeon1.npy",Pgeon1)
np.save(dirname+"Pgeon01.npy",Pgeon01)
np.save(dirname+"Pgeon001.npy",Pgeon001)
np.save(dirname+"dR",dR)

# Plotting
plt.figure()
plt.plot(dR,Pbtz1,'b',label='BTZ, Om=1')
plt.plot(dR,Pgeon1,'b:',label='geon, Om=1')
plt.plot(dR,Pbtz01,'r',label='BTZ, Om=0.1')
plt.plot(dR,Pgeon01,'r:',label='geon, Om=0.1')
plt.plot(dR,Pbtz001,'g',label='BTZ, Om=0.01')
plt.plot(dR,Pgeon001,'g:',label='geon, Om=0.01')
plt.xlabel('dRA')
plt.ylabel('P')
#plt.ylim([0,0.5])
plt.legend()
plt.show()

#%%=========================================================================%%#
#=================== COMPARE PD v M BTZ AND GEON, diff Om ====================#
#=============================================================================#
dirname = "P_v_M_btzgeon_diffOm/"
if not os.path.exists(dirname):
    os.makedirs(dirname)

sig = 1             # width of Gaussian
#M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
#DeltaE = 1          # Omega

l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dR = 1
                    # proper distance of the closest detector
                    
lowm, him = 1e-4, 1
masses = np.exp( np.linspace( np.log(lowm), np.log(him), num = 30 ) )
                    
def get_PvM_for_Om(DeltaE):
    
    P_btz, P_geon = [], []
                        # Create y-axis array for BTZ and geon
    for i in range(np.size(masses)):
        rh = np.sqrt(masses[i])*l
        R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
        nmax = converge_cutoff(masses[i],0.0001)            # summation limit
        print('>>> i =',i,'nmax is',nmax,'\nn = ',end='')
    
        for n in np.arange(0,nmax+1):
            print(n, end=', ', flush=True)
            if n == 0:
                P_btz.append(P_BTZn(n,R,rh,l,pm1,DeltaE,lam,sig))
                P_geon.append(2*PGEON_n(n,R,rh,l,pm1,DeltaE,lam,sig))
            else:
                P_btz[i] += 2*P_BTZn(n,R,rh,l,pm1,DeltaE,lam,sig)
                P_geon[i] += 2*PGEON_n(n,R,rh,l,pm1,DeltaE,lam,sig)
        print('\n')
    
    P_btz, P_geon = np.array(P_btz), np.array(P_geon)
    P_geon += P_btz       # add transition probability addition from geon to BTZ
    return P_btz, P_geon

Pbtz1, Pgeon1 = get_PvM_for_Om(1)
Pbtz01, Pgeon01 = get_PvM_for_Om(0.1)
Pbtz001, Pgeon001 = get_PvM_for_Om(0.01)
np.save(dirname+"Pbtz1.npy",Pbtz1)
np.save(dirname+"Pbtz01.npy",Pbtz01)
np.save(dirname+"Pbtz001.npy",Pbtz001)
np.save(dirname+"Pgeon1.npy",Pgeon1)
np.save(dirname+"Pgeon01.npy",Pgeon01)
np.save(dirname+"Pgeon001.npy",Pgeon001)
np.save(dirname+"dR",dR)

# Plotting
plt.figure()
plt.plot(masses,Pbtz1,'b',label='BTZ, Om=1')
plt.plot(masses,Pgeon1,'b:',label='geon, Om=1')
plt.plot(masses,Pbtz01,'r',label='BTZ, Om=0.1')
plt.plot(masses,Pgeon01,'r:',label='geon, Om=0.1')
plt.plot(masses,Pbtz001,'g',label='BTZ, Om=0.01')
plt.plot(masses,Pgeon001,'g:',label='geon, Om=0.01')
plt.xlabel('M')
plt.ylabel('P')
plt.semilogx()
#plt.ylim([0,0.5])
plt.legend()
plt.show()

#%%=========================================================================%%#
#=================== COMPARE PD v Om BTZ AND GEON, DIFF M ====================#
#=============================================================================#
dirname = "P_v_Om_btzgeon_diffM/"
if not os.path.exists(dirname):
    os.makedirs(dirname)

sig = 1             # width of Gaussian
#M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
#DeltaE = 1          # Omega

l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dR = 1
                    # proper distance of the closest detector
lowOm, hiOm = 1e-4, 1
Om = np.exp( np.linspace( np.log(lowOm), np.log(hiOm), num = 30 ) )
                    
def get_PvOm_for_M(M):
    rh = np.sqrt(M)*l
    R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
    nmax = converge_cutoff(M,0.0001)
    print('nmax =',nmax)
    P_btz, P_geon = [], []
                        # Create y-axis array for BTZ and geon
    for n in np.arange(0,nmax+1):
        print('n = ',n,'; i = ', end='')
        for i in range(np.size(Om)):
            print(i,end=' ',flush=True)
            if n == 0:
                P_btz.append(P_BTZn(n,R,rh,l,pm1,Om[i],lam,sig))
                P_geon.append(2*PGEON_n(n,R,rh,l,pm1,Om[i],lam,sig))
            else:
                P_btz[i] += 2*P_BTZn(n,R,rh,l,pm1,Om[i],lam,sig)
                P_geon[i] += 2*PGEON_n(n,R,rh,l,pm1,Om[i],lam,sig)
        print('')
    
    P_btz, P_geon = np.array(P_btz), np.array(P_geon)
    P_geon += P_btz       # add transition probability addition from geon to BTZ
    return P_btz, P_geon

Pbtz1, Pgeon1 = get_PvOm_for_M(1)
Pbtz01, Pgeon01 = get_PvOm_for_M(0.1)
Pbtz001, Pgeon001 = get_PvOm_for_M(0.01)
np.save(dirname+"Pbtz1.npy",Pbtz1)
np.save(dirname+"Pbtz01.npy",Pbtz01)
np.save(dirname+"Pbtz001.npy",Pbtz001)
np.save(dirname+"Pgeon1.npy",Pgeon1)
np.save(dirname+"Pgeon01.npy",Pgeon01)
np.save(dirname+"Pgeon001.npy",Pgeon001)
np.save(dirname+"dR",dR)

# Plotting
plt.figure()
plt.plot(Om,Pbtz1,'b',label='BTZ, Om=1')
plt.plot(Om,Pgeon1,'b:',label='geon, Om=1')
plt.plot(Om,Pbtz01,'r',label='BTZ, Om=0.1')
plt.plot(Om,Pgeon01,'r:',label='geon, Om=0.1')
plt.plot(Om,Pbtz001,'g',label='BTZ, Om=0.01')
plt.plot(Om,Pgeon001,'g:',label='geon, Om=0.01')
plt.xlabel(r'$\Omega\sigma$')
plt.ylabel('P')
plt.semilogx()
#plt.ylim([0,0.5])
plt.legend()
plt.show()

#%%===========================================================================#
#========================== COMPARE MATRIX ELEMENT X =========================#
#=============================================================================#

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 1          # Omega

#sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dR = np.linspace(0.1, 3,num=100)
                    # proper distance of the closest detector

RA = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector
nmax = converge_cutoff(M,0.0001)            # summation limit
print('nmax is',nmax)

def get_XvdRA_for_Om_sep(DeltaE,sep):
    X_btz_re, X_btz_im, X_geon = 0*RA, 0*RA, 0*RA
                        # Create y-axis array for BTZ and geon
    for n in np.arange(0,nmax+1):
        print('n = ',n)
        print('i =',end=' ')
        for i in range(len(RA)):
            print(i,end=' ',flush=True)
            if n == 0:
                fac = 1
            else:
                fac = 2     # when summing nonzero n's, multiply by 2
            X_btz_re[i] += fac*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_im[i] += fac*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    X_geon += X_btz_re       # add transition probability addition from geon to BTZ
    X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2)
    X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
    return X_btz, X_geon

Xb2, Xg2 = get_XvdRA_for_Om_sep(1,1)
Xb3, Xg3 = get_XvdRA_for_Om_sep(0.1,1)
Xb5, Xg5 = get_XvdRA_for_Om_sep(0.01,1)

dirname = "X_v_dR_btzgeon/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
np.save(dirname+"Xb1",Xb2)
np.save(dirname+"Xb01",Xb3)
np.save(dirname+"Xb001",Xb5)
np.save(dirname+"Xg1",Xg2)
np.save(dirname+"Xg01",Xg3)
np.save(dirname+"Xg001",Xg5)
np.save(dirname+"dRA_new",dR)

#plt.figure()
#plt.plot(dR,Xb2,'b',label='BTZ, Om=1')
#plt.plot(dR,Xg2,'b:',label='geon, Om=1')
#plt.plot(dR,Xb3,'r',label='BTZ, Om=0.1')
#plt.plot(dR,Xg3,'r:',label='geon, Om=0.1')
#plt.plot(dR,Xb5,'g',label='BTZ, Om=0.01')
#plt.plot(dR,Xg5,'g:',label='geon, Om=0.01')
#plt.xlabel('dRA')
#plt.ylabel('|X|')
#plt.legend()
#plt.title('X Element')
#plt.show()

#%%===========================================================================#
#============================ PLOT DELTAX VS. DRA ============================#
#=============================================================================#

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 1          # Omega

#sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dR = np.linspace(0, 40,num=1000)[1:]
                    # proper distance of the closest detector

RA = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector
def get_DeltaX(DeltaE):
    X_geon = 0*dR
    nmax = 2            # summation limit
    for n in np.arange(0,nmax+1):
        print('n = ',n)
        print('i =',end=' ')
        for i in range(len(RA)):
            if i%50 == 0:
                print(i,end=' ',flush=True)
            if n == 0:
                fac = 1
            else:
                fac = 2     # when summing nonzero n's, multiply by 2
            X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    return X_geon

deltaX_Om1 = get_DeltaX(1.)
#deltaX_Om1, deltaX_Om01, deltaX_Om2 = get_DeltaX(1.), get_DeltaX(0.1), get_DeltaX(2.)
#deltaX_Om3, deltaX_Om5, deltaX_Om001 = get_DeltaX(3.), get_DeltaX(5.), get_DeltaX(0.01)

dirname = "DeltaX_v_dR/"
if not os.path.exists(dirname):
    os.makedirs(dirname)

np.save(dirname+"longerDeltaX_Om1",deltaX_Om1)
#np.save(dirname+"longDeltaX_Om01",deltaX_Om01)
#np.save(dirname+"longDeltaX_Om2",deltaX_Om2)
#np.save(dirname+"longDeltaX_Om3",deltaX_Om3)
#np.save(dirname+"longDeltaX_Om5",deltaX_Om5)
#np.save(dirname+"longDeltaX_Om001",deltaX_Om001)
np.save(dirname+'longerdRA',dR)


#%%===========================================================================#
#========== COMPARE CONCURRENCE VS. DISTANCE FOR DIFFERENT ENERGIES ==========#
#=============================================================================#

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#DeltaE = .01          # Omega
nmax = 2            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dRA = np.linspace(0.01,4,num=50)
# Plot PA/lam^2 vs dRA from rh to 10rh
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector
                    
#-------------------------------- function -----------------------------------#
def get_conc_vs_dist(DeltaE):
    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
     = [], [], [], [], [], [], []
                        # Create y-axis array for BTZ and geon

    # Start summing
    for n in np.arange(0,nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(RA)):
            print(i,' ',end='',flush=True)
            if n == 0:
                PA_btz.append(P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig))
                PB_btz.append(P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig))
                X_btz_re.append(XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
                X_btz_im.append(XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
                
                PA_geon.append(2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig))
                PB_geon.append(2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig))
                X_geon.append(2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
            else:
                PA_btz[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
                PB_btz[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
                X_btz_re[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
                X_btz_im[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
                
                PA_geon[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
                PB_geon[i] += 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
                X_geon[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
     = np.array(PA_btz), np.array(PA_geon), np.array(PB_btz), np.array(PB_geon)\
     , np.array(X_btz_re), np.array(X_btz_im), np.array(X_geon)
    X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2 )
    conc_btz = 2*np.maximum(0,np.abs(X_btz) - np.sqrt(np.multiply(PA_btz,PB_btz)))
    
    PA_geon += PA_btz
    PB_geon += PB_btz
    X_geon += X_btz_re       # add transition probability addition from geon to BTZ
    X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
    conc_geon = 2*np.maximum(0,np.abs(X_geon) - np.abs(np.sqrt(np.multiply(PA_geon,PB_geon))))
    
    return conc_btz, conc_geon
#-----------------------------------------------------------------------------#
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('\n***Energy Difference 1...')
    conc_btz_e1, conc_geon_e1 = get_conc_vs_dist(1)
    print('\n***Energy Difference 0.1...')
    conc_btz_e01, conc_geon_e01 = get_conc_vs_dist(0.1)
    print('\n***Energy Difference 0.01..')
    conc_btz_e001, conc_geon_e001 = get_conc_vs_dist(0.01)

dirname = "Conc_v_dR_diffOm/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
np.save(dirname+"conc_btz_e1",conc_btz_e1)
np.save(dirname+"conc_btz_e01",conc_btz_e01)
np.save(dirname+"conc_btz_e001",conc_btz_e001)
np.save(dirname+"conc_geon_e1",conc_geon_e1)
np.save(dirname+"conc_geon_e01",conc_geon_e01)
np.save(dirname+"conc_geon_e001",conc_geon_e001)
np.save(dirname+"dRA",dRA)

## Plotting
#fig = plt.figure(figsize=(9,4.5))
#
## btz
#plt.plot(dRA,conc_btz_e1,'r', label=r'BTZ; $\Omega\sigma=1$')
#plt.plot(dRA,conc_btz_e01,'b', label=r'BTZ; $\Omega\sigma=0.1$')
#plt.plot(dRA,conc_btz_e001,'y', label=r'BTZ; $\Omega\sigma=0.01$')
##plt.plot(dRA,conc_btz_e4,'c', label=r'BTZ; $\Omega\sigma=2.5$')
##plt.plot(dRA,conc_btz_e10,'y', label=r'BTZ; $\Omega\sigma=10$')
#
## geon
#plt.plot(dRA,conc_geon_e1,'r:', label=r'BTZ; $\Omega\sigma=1$')
#plt.plot(dRA,conc_geon_e01,'b:', label=r'BTZ; $\Omega\sigma=0.1$')
#plt.plot(dRA,conc_geon_e001,'y:', label=r'BTZ; $\Omega\sigma=0.01$')
##plt.plot(dRA,conc_geon_e4,'c:', label=r'BTZ; $\Omega\sigma=2.5$')
##plt.plot(dRA,conc_geon_e10,'y', label=r'BTZ; $\Omega\sigma=10$')
#
##plt.ylim([-0.002,0.02])
#plt.legend(loc=4)
#plt.xlabel(r'$d(r_h,R_A)$')
#plt.title('Concurrence with proper distance')
##plt.ylim([0,5e-8])
##plt.xlim([1.075,1.175])
##print('PA_btz: ', PA_btz)
##print('PB_btz: ', PB_btz)
##print('X_btz: ', X_btz)
#plt.show()

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    print('relative difference between concurrence',\
#          relative_diff_mpf(conc_btz_e1,conc_geon_e1))

#%%===========================================================================#
#====================== COMPARE CONCURRENCE VS. ENERGY =======================#
#=============================================================================#
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
dRA = 100             # proper distance of RA from the horizon
nmax = 2            # summation limit

#sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

Om = np.linspace(0.01,3,num=50)

#-------------------------------- function -----------------------------------#
def get_conc_vs_energy(sep):
    sep *= sig
    # Plot PA/lam^2 vs R from rh to 10rh
    RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                        # Array for distance from horizon
    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                     + RA - np.sqrt(RA**2-rh**2))
                        # Distance of the further detector

    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon =\
     [], [], [], [], [], [], []
                        # Create y-axis array for BTZ and geon

    # Start summing
    for n in np.arange(0,nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(np.size(Om)):
            print(i,' ',end='',flush=True)
            if n == 0:
                PA_btz.append(P_BTZn(n,RA,rh,l,pm1,Om[i],lam,sig))
                PB_btz.append(P_BTZn(n,RB,rh,l,pm1,Om[i],lam,sig))
                X_btz_re.append(XBTZ_n_re(n,RA,RB,rh,l,pm1,Om[i],lam,sig))
                X_btz_im.append(XBTZ_n_im(n,RA,RB,rh,l,pm1,Om[i],lam,sig))
                
                PA_geon.append(2*PGEON_n(n,RA,rh,l,pm1,Om[i],lam,sig))
                PB_geon.append(2*PGEON_n(n,RB,rh,l,pm1,Om[i],lam,sig))
                X_geon.append(2*XGEON_n(n,RA,RB,rh,l,pm1,Om[i],lam,sig))
            else:
                PA_btz[i] += 2*P_BTZn(n,RA,rh,l,pm1,Om[i],lam,sig)
                PB_btz[i] += 2*P_BTZn(n,RB,rh,l,pm1,Om[i],lam,sig)
                X_btz_re[i] += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
                X_btz_im[i] += 2*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
                
                PA_geon[i] += 2*PGEON_n(n,RA,rh,l,pm1,Om[i],lam,sig)
                PB_geon[i] += 2*PGEON_n(n,RB,rh,l,pm1,Om[i],lam,sig)
                X_geon[i] += 2*XGEON_n(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
        print('')
    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
         = np.array(PA_btz), np.array(PA_geon), np.array(PB_btz), np.array(PB_geon)\
         , np.array(X_btz_re), np.array(X_btz_im), np.array(X_geon)
    X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2 )
    conc_btz = 2*np.maximum(0,X_btz - np.sqrt(np.multiply(PA_btz,PB_btz)))

    PA_geon += PA_btz
    PB_geon += PB_btz
    X_geon += X_btz_re       # add transition probability addition from geon to BTZ
    X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
    conc_geon = 2*np.maximum(0,np.abs(X_geon) - np.sqrt(np.multiply(PA_geon,PB_geon)))
    
    return conc_btz, conc_geon
#-----------------------------------------------------------------------------#

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    print('\n***Separation 0.8 ...')
    conc_btz_sep08, conc_geon_sep08 = get_conc_vs_energy(0.8)
    print('\n***Separation 1 ...')
    conc_btz_sep1, conc_geon_sep1 = get_conc_vs_energy(1)
    print('\n***Separation 1.2 ...')
    conc_btz_sep12, conc_geon_sep12 = get_conc_vs_energy(1.2)

dirname = "Conc_v_En_diffsep/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
np.save(dirname+"conc_btz_sep08",conc_btz_sep08)
np.save(dirname+"conc_btz_sep1",conc_btz_sep1)
np.save(dirname+"conc_btz_sep12",conc_btz_sep12)
np.save(dirname+"conc_geon_sep08",conc_geon_sep08)
np.save(dirname+"conc_geon_sep1",conc_geon_sep1)
np.save(dirname+"conc_geon_sep12",conc_geon_sep12)
np.save(dirname+"Om",Om)

# Plotting
fig = plt.figure(figsize=(9,4.5))

# btz

#plt.plot(Om,PA_btz,'c', label='PA BTZ', )
#plt.plot(Om,PB_btz,'orange', label='PB BTZ')
#plt.plot(Om,X_btz,'g', label='X BTZ')
plt.plot(Om,conc_btz_sep08,'r', label=r'BTZ with $d(R_A,R_B)/\sigma = 0.8$')
plt.plot(Om,conc_btz_sep1,'b', label=r'BTZ with $d(R_A,R_B)/\sigma = 1$')
plt.plot(Om,conc_btz_sep12,'c', label=r'BTZ with $d(R_A,R_B)/\sigma = 1.2$')

# geon

#plt.plot(Om,PA_geon,'c:', label='PA BTZ', )
#plt.plot(Om,PB_geon,color='orange',linestyle=':', label='PB BTZ')
#plt.plot(Om,X_geon,'g:', label='X BTZ')
plt.plot(Om,conc_geon_sep08,'r:', label=r'geon with $d(R_A,R_B)/\sigma = 0.8$')
plt.plot(Om,conc_geon_sep1,'b:', label=r'geon with $d(R_A,R_B)/\sigma = 1$')
plt.plot(Om,conc_geon_sep12,'c:', label=r'geon with $d(R_A,R_B)/\sigma = 1.2$')

plt.legend()
plt.title('Concurrence vs Energy for different separations')
plt.xlabel(r'$\Omega\sigma$')
#plt.ylim([0.088,0.092])
#plt.xlim([1.075,1.175])

plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('relative difference between concurrence for separation=1',\
          relative_diff_mpf(conc_btz_sep1,conc_geon_sep1))

#%%===========================================================================#
#==================== D_DEATH VERSUS ENERGY FOR DIFF SEP =====================#
#=============================================================================#
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
nmax = 1            # summation limit

#sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

Om = np.linspace(0.1,4,num=30)

prec = 5e-3

def get_ddeath_for_sep(sep, guess):
    sep *= sig
    d_death_BTZ = []
    d_death_GEON = []
    for i in range(np.size(Om)):
        print('\n***i = ',i)
        if i == 0:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,guess,prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,guess,prec))
        else:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,d_death_BTZ[-1],prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,d_death_GEON[-1],prec))
    return d_death_BTZ, d_death_GEON

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#    print('\n==>Separation 0.8 ...')
#    ddbtz_sep08, ddgeon_sep08 = get_ddeath_for_sep(0.8, 0.2)
    print('\n==>Separation 1 ...')
    ddbtz_sep1, ddgeon_sep1 = get_ddeath_for_sep(1.0, 0.3)
#    print('\n==>Separation 1.2 ...')
#    ddbtz_sep12, ddgeon_sep12 = get_ddeath_for_sep(1.2, 0.5)

dirname = "ddeath_v_En_diffsep/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
##np.save(dirname+"ddbtz_sep08_M1",ddbtz_sep08)
np.save(dirname+"ddbtz_sep1_M1_l1",ddbtz_sep1)
##np.save(dirname+"ddbtz_sep12_M1",ddbtz_sep12)
##np.save(dirname+"ddgeon_sep08_M1",ddgeon_sep08)
np.save(dirname+"ddgeon_sep1_M1_l1",ddgeon_sep1)
##np.save(dirname+"ddgeon_sep12_M1",ddgeon_sep12)
np.save(dirname+"Om_M1_l1",Om)
#sep = 1.2
#sep *= sig
#ddbtz_sep12 = []
#ddgeon_sep12 = []
#for i in range(np.size(Om)):
#    print('\n***i = ',i)
#    if i == 0:
#        print('btz')
#        ddbtz_sep12.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,1.1,prec))
#        print('geon')
#        ddgeon_sep12.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,0.4,prec))
#    else:
#        print('btz')
#        ddbtz_sep12.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,ddbtz_sep12[-1],prec))
#        print('geon')
#        ddgeon_sep12.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,ddgeon_sep12[-1],prec))
    
Om *= sig

## Plotting
fig = plt.figure(figsize=(9,4.5))
#
## btz
#plt.plot(Om[:47],ddbtz_sep08[:47],'r', label=r'BTZ with $d(R_A,R_B)/\sigma = 0.8$')
#plt.plot(Om,ddbtz_sep1,'b', label=r'BTZ with $d(R_A,R_B)/\sigma = 1$')
#plt.plot(Om,ddbtz_sep12,'c', label=r'BTZ with $d(R_A,R_B)/\sigma = 1.2$')
#
## geon
#plt.plot(Om[:47],ddgeon_sep08[:47],'r:', label=r'geon with $d(R_A,R_B)/\sigma = 0.8$')
plt.plot(Om,ddgeon_sep1,'b:', label=r'geon with $d(R_A,R_B)/\sigma = 1$')
#plt.plot(Om,ddgeon_sep12,'c:', label=r'geon with $d(R_A,R_B)/\sigma = 1.2$')
plt.legend() 
#
plt.xlabel(r'$\Omega\sigma$')
#plt.ylabel(r'$d_{death}(r_h,R_A)/\sigma$')
#
##fig.savefig('ddeath_vs_energy.pdf',dpi=170)
#plt.ylim([0.0,1.4])

#%%===========================================================================#
#==================== D_DEATH VERSUS ENERGY FOR DIFF MASS ====================#
#=============================================================================#
sig = 1             # width of Gaussian
#M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
#nmax = 3            # summation limit

#sep *= sig
l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

Om = np.linspace(0.02,6,num=80)

prec = 5e-3

def get_ddeath_for_mass(M, guess):
    #sep *= sig
    #d_death_BTZ = []
    rh = np.sqrt(M)*l   # radius of horizon
    nmax = converge_cutoff(M,0.01)
    d_death_GEON = []
    for i in range(np.size(Om)):
        print('\n***i = ',i)
        if i == 0:
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,guess,prec))
        else:
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,d_death_GEON[-1],prec))
    return d_death_GEON

dirname = "ddeath_v_En_diffmasses/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
    
np.save(dirname+"Om_sep1",Om)

print('\n==>Mass 10 ...')
ddgeon_m10 = get_ddeath_for_mass(10, 0.2)
np.save(dirname+"ddgeon_m10",ddgeon_m10)

print('\n==>Mass 0.03 ...')
ddgeon_m003 = get_ddeath_for_mass(0.03, 0.2)
np.save(dirname+"ddgeon_m003",ddgeon_m003)

print('\n==>Mass 0.3 ...')
ddgeon_m03 = get_ddeath_for_mass(0.3, 0.2)
np.save(dirname+"ddgeon_m03",ddgeon_m03)

print('\n==>Mass 0.01 ...')
ddgeon_m001 = get_ddeath_for_mass(0.01, 0.2)
np.save(dirname+"ddgeon_m001",ddgeon_m001)

print('\n==>Mass 0.1 ...')
ddgeon_m01 = get_ddeath_for_mass(0.1, 0.19)
np.save(dirname+"ddgeon_m01",ddgeon_m01)

#sep = 1.2
#sep *= sig
#ddbtz_sep12 = []
#ddgeon_sep12 = []
#for i in range(np.size(Om)):
#    print('\n***i = ',i)
#    if i == 0:
#        print('btz')
#        ddbtz_sep12.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,1.1,prec))
#        print('geon')
#        ddgeon_sep12.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,0.4,prec))
#    else:
#        print('btz')
#        ddbtz_sep12.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,ddbtz_sep12[-1],prec))
#        print('geon')
#        ddgeon_sep12.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,ddgeon_sep12[-1],prec))
    
Om *= sig

# Plotting
#fig = plt.figure(figsize=(9,4.5))
#
## btz
#plt.plot(Om[:47],ddbtz_sep08[:47],'r', label=r'BTZ with $d(R_A,R_B)/\sigma = 0.8$')
#plt.plot(Om,ddbtz_sep1,'b', label=r'BTZ with $d(R_A,R_B)/\sigma = 1$')
#plt.plot(Om,ddbtz_sep12,'c', label=r'BTZ with $d(R_A,R_B)/\sigma = 1.2$')
#
## geon
#plt.plot(Om[:47],ddgeon_sep08[:47],'r:', label=r'geon with $d(R_A,R_B)/\sigma = 0.8$')
#plt.plot(Om,ddgeon_sep1,'b:', label=r'geon with $d(R_A,R_B)/\sigma = 1$')
#plt.plot(Om,ddgeon_sep12,'c:', label=r'geon with $d(R_A,R_B)/\sigma = 1.2$')
#plt.legend() 
#
#plt.xlabel(r'$\Omega\sigma$')
#plt.ylabel(r'$d_{death}(r_h,R_A)/\sigma$')
#
##fig.savefig('ddeath_vs_energy.pdf',dpi=170)
#plt.ylim([0.0,1.4])

#%%===========================================================================#
#=================== D_DEATH VERSUS ENERGY LOG FOR DIFF M ====================#
#=============================================================================#
sig = 1             # width of Gaussian
#M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
#nmax = 3            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

Om= np.exp(np.linspace(np.log(1e-4),np.log10(1.),num=30))

prec = 5e-3

def get_ddeath_for_mass_btzgeon(M, guess):
    #sep *= sig
    d_death_BTZ = []
    rh = np.sqrt(M)*l   # radius of horizon
    nmax = converge_cutoff(M,0.01)
    d_death_GEON = []
    for i in range(np.size(Om)):
        print('\n***i = ',i)
        if i == 0:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,guess,prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,guess,prec))
        else:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh,l,pm1,Om[i],lam,sig,d_death_BTZ[-1],prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh,l,pm1,Om[i],lam,sig,d_death_GEON[-1],prec))
    return d_death_BTZ, d_death_GEON

dirname = "ddeath_v_LogEn_diffmasses/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
    
#np.save(dirname+"LogOmLonger",Om)

print('\n==>Mass 0.038 ...')
ddbtz_m0038, ddgeon_m0038 = get_ddeath_for_mass_btzgeon(0.038, 5.6)
np.save(dirname+"ddbtz_m0038longer",ddbtz_m0038)
np.save(dirname+"ddgeon_m0038longer",ddgeon_m0038)

print('\n==>Mass 0.034 ...')
ddbtz_m0034, ddgeon_m0034 = get_ddeath_for_mass_btzgeon(0.034, 6.7)
np.save(dirname+"ddbtz_m0034longer",ddbtz_m0034)
np.save(dirname+"ddgeon_m0034longer",ddgeon_m0034)
#
#print('\n==>Mass 0.3 ...')
#ddbtz_m03, ddgeon_m03 = get_ddeath_for_mass_btzgeon(0.3, 2.)
#np.save(dirname+"ddbtz_m03longer",ddbtz_m03)
#np.save(dirname+"ddgeon_m03longer",ddgeon_m03)
#
#print('\n==>Mass 1 ...')
#ddbtz_m1, ddgeon_m1 = get_ddeath_for_mass_btzgeon(1., 0.8)
#np.save(dirname+"ddbtz_m1longer",ddbtz_m1)
#np.save(dirname+"ddgeon_m1longer",ddgeon_m1)

#plt.figure(figsize=(6,4))
#plt.plot(Om,ddbtz_m1,'b',label='M = 1; BTZ')
#plt.plot(Om,ddgeon_m1,'b:',label='M = 1; geon')
#plt.plot(Om,ddbtz_m03,'c',label='M = 0.3; BTZ')
#plt.plot(Om,ddgeon_m03,'c:',label='M = 0.3; geon')
#plt.plot(Om,ddbtz_m01,'g',label='M = 0.1; BTZ')
#plt.plot(Om,ddgeon_m01,'g:',label='M = 0.1; geon')
#plt.plot(Om,ddbtz_m003,'y',label='M = 0.01; BTZ')
#plt.plot(Om,ddgeon_m003,'y:',label='M = 0.01; geon')
#plt.xlabel(r'$\Omega\sigma$')
#plt.semilogx()
#plt.ylabel(r'$d_{death}$')
#plt.legend()
#plt.savefig('plots/diffm_ddvLogOm.png',dpi=170)

#%%===========================================================================#
#=================== D_DEATH VERSUS MASS LOG FOR DIFF OMs ====================#
#=============================================================================#
sig = 1             # width of Gaussian
#M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
#nmax = 3            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

M = np.exp(np.linspace(np.log(1e-3),np.log10(1.),num=40))
rh = np.sqrt(M)*l   # radius of horizon

prec = 1e-2

def get_ddeath_for_om_btzgeon(Om, guess):
    #sep *= sig
    d_death_BTZ = []
    d_death_GEON = []
    for i in range(np.size(M)):
        nmax = converge_cutoff(M[i],0.01)
        print('\n***i = ',i)
        if i == 0:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh[i],l,pm1,Om,lam,sig,guess,prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh[i],l,pm1,Om,lam,sig,guess,prec))
        else:
            print('btz')
            d_death_BTZ.append(find_d_death_BTZ(nmax,sep,rh[i],l,pm1,Om,lam,sig,d_death_BTZ[-1],prec))
            print('geon')
            d_death_GEON.append(find_d_death_GEON(nmax,sep,rh[i],l,pm1,Om,lam,sig,d_death_GEON[-1],prec))
    return d_death_BTZ, d_death_GEON

dirname = "ddeath_v_LogM_diffenergies/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
    
np.save(dirname+"LogM",M)

print('\n==> Energy 1 ...')
ddbtz_Om1, ddgeon_Om1 = get_ddeath_for_om_btzgeon(1., 1.)
np.save(dirname+"ddbtz_Om1",ddbtz_Om1)
np.save(dirname+"ddgeon_Om1",ddgeon_Om1)

print('\n==> Energy 0.1 ...')
ddbtz_Om01, ddgeon_Om01 = get_ddeath_for_om_btzgeon(0.1, 2.)
np.save(dirname+"ddbtz_Om01",ddbtz_Om01)
np.save(dirname+"ddgeon_Om01",ddgeon_Om01)

print('\n==> Energy 0.01 ...')
ddbtz_Om001, ddgeon_Om001 = get_ddeath_for_om_btzgeon(0.01, 4.)
np.save(dirname+"ddbtz_Om001",ddbtz_Om001)
np.save(dirname+"ddgeon_Om001",ddgeon_Om001)

plt.plot(M,ddbtz_Om1,'b',label=r'$\Omega\sigma = 1$; BTZ')
plt.plot(M,ddgeon_Om1,'b:',label=r'$\Omega\sigma = 1$; geon')
plt.plot(M,ddbtz_Om01,'c',label=r'$\Omega\sigma = 0.1$; BTZ')
plt.plot(M,ddgeon_Om01,'c:',label=r'$\Omega\sigma = 0.1$; geon')
plt.plot(M,ddbtz_Om001,'g',label=r'$\Omega\sigma = 0.01$; BTZ')
plt.plot(M,ddgeon_Om001,'g:',label=r'$\Omega\sigma = 0.01$; geon')
plt.legend()
plt.xlabel('M')
plt.semilogx()
#plt.semilogy()
plt.ylabel(r'$d_{death}$')
plt.ylim([0,15])

#%%===========================================================================#
##======================= PA VERSUS dRA FOR DIFFERENT OM ======================#
##=============================================================================#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
##dRA = 100             # proper distance of RA from the horizon
#nmax = 0            # summation limit
#
##sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#lam = 1             # coupling constant
#
#Om = [4]
#P_btz = []
#P_geon = []
#colors = ['b','c','g','orange','red']
#
#dRA = np.linspace(0.02,10,num=200)
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#
#print('\nCreating P_A arrays...')
#for en in Om:
#    print('\n>>> Energy = %s <<<'%(en))
#    
#    thisP_btz = []
#    thisP_geon = []
#    
#    print('i(50) =',end=' ')
#    for i in range(np.size(RA)):
#        print(i,end=' ',flush=True)
#        thisP_btz.append(P_BTZn(0,RA[i],rh,l,pm1,en,lam,sig))
#        thisP_geon.append(thisP_btz[-1]+PGEON_n(0,RA[i],rh,l,pm1,en,lam,sig))
#    P_btz.append(thisP_btz)
#    P_geon.append(thisP_geon)
#
#diff = np.divide( np.abs( np.array(P_btz[0]) - np.array(P_geon[0]) ) , np.array(P_geon[0]))
#
##%%
#figp, (ax1,ax2) = plt.subplots(2,1,figsize=(6,6))
#for i in range(len(P_btz)):
#    ax1.plot(dRA,P_btz[i],color=colors[i],label=r'BTZ with $\Omega = %s$'%(Om[i]))
#    ax1.plot(dRA,P_geon[i],color=colors[i],linestyle=':',label=r'geon with $\Omega = %s$'%(Om[i]))
#
##ax1.xlabel(r'$d(r_h,R_A)$')
#ax1.set_ylabel(r'$P_A/\lambda^2$')
#ax1.legend(loc=1)
#ax1.set_ylim([0,2e-8])
#
#ax2.plot(dRA,diff,label='relative difference')
#ax2.legend()
#
##figp.savefig('PA_vs_dRA_diffOms.pdf',dpi=170)
#plt.show()
#

##%%===========================================================================#
##======================= X VERSUS dRA FOR DIFFERENT OM =======================#
##=============================================================================#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
##dRA = 100             # proper distance of RA from the horizon
#nmax = 0            # summation limit
#
##sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#lam = 1             # coupling constant
#
#Om = [2,3,4,5,10]
#X_btz = []
#X_geon = []
#colors = ['b','c','g','orange','red']
#
#dRA = np.linspace(0.02,1,num=50)
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                 + RA - np.sqrt(RA**2-rh**2))
#
#print('\nCreating X arrays...')
#for en in Om:
#    print('\n>>> Energy = %s <<<'%(en))
#    
#    thisX_btz = []
#    thisX_geon = []
#    
#    print('i(50) =',end=' ')
#    for i in range(np.size(RA)):
#        print(i,end=' ',flush=True)
#        thisX_btzre = XBTZ_n_re(0,RA[i],RB[i],rh,l,pm1,en,lam,sig)
#        thisX_btzim = XBTZ_n_im(0,RA[i],RB[i],rh,l,pm1,en,lam,sig)
#        thisX_btz.append(fp.sqrt(thisX_btzre**2 + thisX_btzim**2))
#        thisX_geon.append(fp.sqrt((thisX_btzre+XGEON_n(0,RA[i],RB[i],rh,l,pm1,en,lam,sig))**2 + thisX_btzim**2))
#    X_btz.append(thisX_btz)
#    X_geon.append(thisX_geon)
#
#figp = plt.figure(figsize=(9,4.5))
#for i in range(len(X_btz)):
#    plt.plot(dRA,X_btz[i],color=colors[i],label=r'BTZ with $\Omega = %s$'%(Om[i]))
#    plt.plot(dRA,X_geon[i],color=colors[i],linestyle=':',label=r'geon with $\Omega = %s$'%(Om[i]))
#
#plt.xlabel(r'$d(r_h,R_A)$')
#plt.ylabel(r'$X/\lambda^2$')
##plt.legend(loc=1)
#plt.ylim([-0.000001,0.000004])
#plt.show()
#
#figp = plt.figure(figsize=(9,6))
#for i in [2,3]:
#    plt.plot(dRA,P_btz[i],color=colors[i],label=r'P_BTZ with $\Omega = %s$'%(Om[i]))
#    #plt.plot(dRA,P_geon[i],color=colors[i],linestyle='--',label=r'geon with $\Omega = %s$'%(Om[i]))
#    plt.plot(dRA,X_btz[i],color=colors[i],linestyle ='--',label=r'X_BTZ with $\Omega = %s$'%(Om[i]))
#    plt.plot(dRA,X_geon[i],color=colors[i],linestyle=':',label=r'X_geon with $\Omega = %s$'%(Om[i]))
#
#plt.xlabel(r'$d(r_h,R_A)$')
#plt.ylabel(r'$X/\lambda^2$')
#plt.legend(loc=3)
#plt.ylim([-0.000001,0.000006])
#plt.show()
#figp.savefig('X_PA_vs_d.pdf',dpi=170)
#
#
#%%             CHECK IF FUNCTIONS ARE WORKING CORRECTLY

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
Om = 1

def diff_X_PAPB(dRA):
    RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
    RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
                     + RA - mp.sqrt(RA**2-rh**2))
    Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
    Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig)
    PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig)
    PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig)
    Xre_geon = XGEON_n(0,RA,RB,rh,l,pm1,Om,lam,sig)
    
    for n in range(1,nmax):
        Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)
        Xim += 2*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)
        PA += 2*P_BTZn(n,RA,rh,l,pm1,Om,lam,sig)
        PB += 2*P_BTZn(n,RB,rh,l,pm1,Om,lam,sig)
        Xre_geon += 2*XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig)
    return fp.sqrt(PA*PB), fp.sqrt(Xre**2+Xim**2), fp.sqrt((Xre+Xre_geon)**2 + Xim**2)

dRA = np.linspace(0.2,7,num=400)
diffxpapb = [[],[],[]]
print('')
for i in range(np.size(dRA)):
    print(i)
    x = diff_X_PAPB(dRA[i])
    for j in range(len(x)):
        diffxpapb[j].append(x[j])

conc_btz = np.maximum(0 , np.array(diffxpapb[1]) - np.array(diffxpapb[0]))
conc_geon = np.maximum(0 , np.array(diffxpapb[2]) - np.array(diffxpapb[0]))
#ddbtz = find_d_death_BTZ(0,sep,rh,l,pm1,Om,lam,sig,0.3,1e-3)
#ddgeon = find_d_death_GEON(0,sep,rh,l,pm1,Om,lam,sig,0.3,1e-3)

#%%===========================================================================#
#=============== X_IM COMPARED TO root(PAPB) FOR DIFF ENERGIES ===============#
#=============================================================================#
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
nmax = 0            # summation limit
lam = 1

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon

dRA = np.linspace(0.1,10,num=200)
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))

def get_xim_and_root_papb(en):
    xim = []
    rootpapb = []
    
    print('jj =',end=' ')
    for jj in range(np.size(dRA)):
        print(jj,end=' ',flush=True)
        rootpapb.append(np.abs(fp.sqrt(P_BTZn(0,RA[jj],rh,l,pm1,en,lam,sig)*P_BTZn(0,RB[jj],rh,l,pm1,en,lam,sig))))
        xim.append(np.abs(XBTZ_n_im(0,RA[jj],RB[jj],rh,l,pm1,en,lam,sig)))
    
    return np.array(xim), np.array(rootpapb)

Om = [1]
colors = ['b','c','r','g','orange','k']
rootpapbs, xim, xim_rootpapb = [], [], []

for i in range(len(Om)):
    print('\n >>> Energy #',Om[i])
    x = get_xim_and_root_papb(Om[i])
    rootpapbs.append(x[1])
    xim.append(x[0])
    xim_rootpapb.append(x[0]-x[1])

#%%
fig = plt.figure(figsize=(9,4.5))
for i in range(len(Om)):
    plt.plot(dRA,rootpapbs[i],color=colors[i],label=r'$\sqrt{P_A P_B}$ BTZ, energy$=%s$'%(Om[i]))
    plt.plot(dRA,xim[i],color=colors[i],linestyle='--',label=r'$X_{im}$ BTZ, energy$=%s$'%(Om[i]))
plt.legend(loc=5)
#plt.ylim([0,0.00001])
plt.show()

##%%===========================================================================#
##======================= CONC_GEON DOWN TO ZERO AGAIN? =======================#
##=============================================================================#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
##dRA = 100             # proper distance of RA from the horizon
#nmax = 0            # summation limit
#
#sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#
#dRA = np.concatenate((np.concatenate((np.linspace(0.75,0.8,num=100),np.linspace(2.12,2.18,num=100)),axis=0),\
#                      np.linspace(4.3, 4.5, num=100)),axis=0)
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                 + RA - np.sqrt(RA**2-rh**2))
#
#def get_all(en):
#    xim = []
#    rootpapb = []
#    xre = []
#    xge = []
#    
#    print('jj =',end=' ')
#    for jj in range(np.size(dRA)):
#        print(jj,end=' ',flush=True)
#        rootpapb.append(np.abs(fp.sqrt(P_BTZn(0,RA[jj],rh,l,pm1,en,lam,sig)*P_BTZn(0,RB[jj],rh,l,pm1,en,lam,sig))))
#        xim.append(np.abs(XBTZ_n_im(0,RA[jj],RB[jj],rh,l,pm1,en,lam,sig)))
#        xre.append(XBTZ_n_re(0,RA[jj],RB[jj],rh,l,pm1,en,lam,sig))
#        xge.append(XGEON_n(0,RA[jj],RB[jj],rh,l,pm1,en,lam,sig))
#        
#        xbtz = np.sqrt(np.array(xim)**2 + np.array(xre)**2)
#        xgeon = np.sqrt(np.array(xim)**2 + (np.array(xre) + np.array(xge))**2)
#    
#    return np.array(rootpapb), np.array(xim), xbtz, xgeon
#
#Om = 5
#returned = get_all(Om)
#rootpapb = returned[0]
#xim = returned[1]
#xbtz = returned[2]
#xgeon = returned[3]
#
##%%
#fig = plt.figure(figsize=(10,4))
#plt.plot(dRA,xim,'b',label=r'|X_{im}|')
#plt.plot(dRA,rootpapb,'r',label=r'$\sqrt{P_A P_B}$')
#plt.plot(dRA,xbtz,'c',label=r'$|X_{BTZ}|')
#plt.plot(dRA,xgeon,'c--',label=r'$|X_{geon}|$')
#plt.legend()
#plt.ylim([0, 4e-8])
#plt.xlim([0.7878,0.788])
#
#
##f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, facecolor='w',figsize=(10,5))
##ax1.plot(dRA,xim,'b',label=r'|X_{im}|')
##ax1.plot(dRA,rootpapb,'r',label=r'$\sqrt{P_A P_B}$')
##ax1.plot(dRA,xbtz,'c',label=r'$|X_{BTZ}|')
##ax1.plot(dRA,xgeon,'c--',label=r'$|X_{geon}|$')
##
##ax2.plot(dRA,xim,'b',label=r'|X_{im}|')
##ax2.plot(dRA,rootpapb,'r',label=r'$\sqrt{P_A P_B}$')
##ax2.plot(dRA,xbtz,'c',label=r'$|X_{BTZ}|')
##ax2.plot(dRA,xgeon,'c--',label=r'$|X_{geon}|$')
##
##ax3.plot(dRA,xim,'b',label=r'|X_{im}|')
##ax3.plot(dRA,rootpapb,'r',label=r'$\sqrt{P_A P_B}$')
##ax3.plot(dRA,xbtz,'c',label=r'$|X_{BTZ}|$')
##ax3.plot(dRA,xgeon,'c--',label=r'$|X_{geon}|$')
##ax3.legend()
##ax3.set_ylim(0,4e-9)
##
##ax1.set_xlim(0.75, 0.8)
##ax2.set_xlim(2.12, 2.18)
##ax3.set_xlim(4.3,4.5)
##
##ax1.spines['right'].set_visible(False)
##ax2.spines['left'].set_visible(False)
##ax2.spines['right'].set_visible(False)
##ax3.spines['left'].set_visible(False)
##
##ax1.yaxis.tick_left()
##ax1.tick_params(labelright='off')
##ax2.tick_params(labelleft='off')
##ax2.tick_params(axis='y',which='both',left='off',right='off')
##ax2.tick_params(labelright='off')
##ax3.yaxis.tick_right()
#
##%%===========================================================================#
##------------------ COMPARE INTEGRANDS BEFORE AFTER SIGN(X) ------------------#
##=============================================================================#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 0.1        # Omega
#nmax = 1            # summation limit
#
#sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#lam = 1             # coupling constant
#
#dRA = .2
## Plot PA/lam^2 vs dRA from rh to 10rh
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#                    # Array for distance from horizon
#RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                 + RA - np.sqrt(RA**2-rh**2))
#print('acoshZp=',np.arccosh((RA**2+rh**2)/(RA**2-rh**2)))
#
#
#y = np.linspace(-25,25,num=100)
#yintgnd_f02 = []
#yintgndnew_f02 = []
#print('i =',end=' ')
#for i in range(len(y)):
#    print(i,end=',',flush=True)
#    yintgnd_f02.append(f02(y[i],0,RA,rh,l,pm1,DeltaE,lam,sig))
#    yintgndnew_f02.append(f02_new(y[i],0,RA,rh,l,pm1,DeltaE,lam,sig))
#
#fig = plt.figure(figsize=(9,5))
#plt.plot(y,yintgnd_f02,label='before')
#plt.plot(y,yintgndnew_f02,label='after')
#plt.legend()
#
#
##%%===========================================================================#
##------------------- COMPARE P AND X BEFORE AFTER SIGN(X) --------------------#
##=============================================================================#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 1          # Omega
#nmax = 1            # summation limit
#
#sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#lam = 1             # coupling constant
#
#dRA = np.linspace(0.01,1.2,num=50)
## Plot PA/lam^2 vs dRA from rh to 10rh
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#                    # Array for distance from horizon
#RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                 + RA - np.sqrt(RA**2-rh**2))
#                    # Distance of the further detector
#
#PA_bef, PA_aft, PB_bef, PB_aft, X_bef_re, X_bef_im, X_aft_re, X_aft_im\
# = [], [], [], [], [], [], [], []
#                    # Create y-axis array for BTZ and geon
#
## Start summing
#for n in np.arange(0,nmax+1):
#    print('n =',n)
#    print(' i = ', end='')
#    for i in range(len(RA)):
#        print(i,end=' ',flush=True)
#        if n == 0:
#            PA_bef.append(P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig))
#            PB_bef.append(P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig))
#            X_bef_re.append(XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#            X_bef_im.append(XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#            
#            PA_aft.append(P_BTZn_new(n,RA[i],rh,l,pm1,DeltaE,lam,sig))
#            PB_aft.append(P_BTZn_new(n,RB[i],rh,l,pm1,DeltaE,lam,sig))
#            X_aft_re.append(XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#            X_aft_im.append(XBTZ_n_im_new(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#
#        else:     # when summing nonzero n's, multiply by 2
#            PA_bef[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
#            PB_bef[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
#            X_bef_re[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#            X_bef_im[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#
#            PA_aft[i] += 2*P_BTZn_new(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
#            PB_aft[i] += 2*P_BTZn_new(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
#            X_aft_re[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#            X_aft_im[i] += 2*XBTZ_n_im_new(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#            
#    print('')
#
#PA_bef, PA_aft, PB_bef, PB_aft, X_bef_re, X_bef_im, X_aft_re, X_aft_im\
# = np.array(PA_bef), np.array(PA_aft), np.array(PB_bef), np.array(PB_aft)\
# , np.array(X_bef_re), np.array(X_bef_im), np.array(X_aft_re), np.array(X_aft_im)
#
#X_bef = np.sqrt( X_bef_re**2 + X_bef_im**2 )
#X_aft = np.sqrt( X_aft_re**2 + X_aft_im**2 )
##conc_btz = 2*np.maximum(0,np.abs(X_btz) - np.sqrt(np.multiply(PA_btz,PB_btz)))
#
## Plotting
#fig = plt.figure(figsize=(9,5))
#
## btz
#plt.plot(dRA,PA_bef,'c', label='PA bef', )
##plt.plot(dRA,PB_bef,'orange', label='PB bef')
##plt.plot(dRA,X_bef,'g', label='X bef')
#plt.plot(dRA,PA_aft,'c', linestyle=':', label='PA aft', )
##plt.plot(dRA,PB_aft,'orange', linstyle=':', label='PB aft')
##plt.plot(dRA,X_aft,'g', linstyle=':', label='X aft')
#
#plt.legend()
#plt.xlabel(r'$d(r_h,R_A)$')
#plt.title('Concurrence with proper distance')
##plt.ylim([0.134,0.16])
##plt.xlim([1.075,1.175])
##print('PA_btz: ', PA_btz)
##print('PB_btz: ', PB_btz)
##print('X_btz: ', X_btz)
#plt.show()
#
##print('relative difference between concurrence',\
##      relative_diff_mpf(conc_btz,conc_geon))

#%%===========================================================================#
#------------------------------ MASS INSPECTION ------------------------------#
#=============================================================================#
sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
DeltaE = 1        # Omega

sep *= sig
l = 10*sig          # cosmological parameter
lam = 1             # coupling constant

dRA = np.linspace(0.1,10,num=80)
# Plot PA/lam^2 vs dRA from rh to 10rh


M = 0.01
nmax = converge_cutoff(M, 0.02)            # summation limit
print('nmax is ', nmax,'\n')
rh = np.sqrt(M)*l   # radius of horizon
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))

###
PA, PB, Xre, Xim = [], [], [], []
PAg, PBg, Xreg = [], [], []

for n in range(nmax+1):
    print('n =',n)
    print(' i = ', end='')
    for i in range(len(dRA)):
        print(i,end=' ',flush=True)
        if n == 0:    
            PA.append( P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
            PB.append( P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
            Xre.append( XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            Xim.append( XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            
            PAg.append( PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
            PBg.append( PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
            Xreg.append( XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
        
        else:
            PA[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
            PB[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
            Xre[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
            Xim[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
            
            PAg[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
            PBg[i] += 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
            Xreg[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
    print('')

papb = np.sqrt( np.multiply(np.array(PA),np.array(PB)) )
papbg = np.sqrt( np.multiply(np.array(PAg)+np.array(PA) , np.array(PBg)+np.array(PB) ) )
Xbtz = np.sqrt( np.array(Xre)**2 + np.array(Xim)**2 )
Xgeon = np.sqrt( (np.array(Xre) + np.array(Xreg))**2 + np.array(Xim)**2 )

concbtz = 2 * np.maximum( 0 , Xbtz - papb )
concgeon = 2 * np.maximum( 0 , Xgeon - papbg )

dirname = "Everything_with_diffmass/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
np.save(dirname+'Om1m001_dRA',dRA)
np.save(dirname+'Om1m001_PAb',np.array(PA))
np.save(dirname+'Om1m001_PAg',np.array(PAg)+np.array(PA))
np.save(dirname+'Om1m001_PBb',np.array(PB))
np.save(dirname+'Om1m001_PBg',np.array(PBg)+np.array(PB))
np.save(dirname+'Om1m001_rootpapb',papb)
np.save(dirname+'Om1m001_rootpapbg',papbg)
np.save(dirname+'Om1m001_Xb',Xbtz)
np.save(dirname+'Om1m001_Xg',Xgeon)
np.save(dirname+'Om1m001_concb',concbtz)
np.save(dirname+'Om1m001_concg',concgeon)


fig1 = plt.figure(1, figsize = (8,4))

#plt.plot(dRA,papb,'r',label='papb btz')
#plt.plot(dRA,papbg,'r:',label='papb geon')
#plt.plot(dRA,Xbtz, 'g', label='|X| btz')
#plt.plot(dRA,Xgeon, 'g:', label='|X| geon')
plt.plot(dRA,concbtz,'grey',label=r'$C_{BTZ}$')
plt.plot(dRA,concgeon,'grey',linestyle=':',label=r'$C_{geon}$')

plt.legend()
#plt.xlabel(r'$d(rh,R_A)$')

fig2 = plt.figure(2, figsize = (8,4))
plt.plot(dRA,papbg-papb,label=r'$\Delta \sqrt{P_A P_B}$')
plt.plot(dRA,Xgeon-Xbtz,label=r'$\Delta |X|$')
plt.legend()
plt.xlabel(r'$d(rh,R_A)$')
#plt.ylim([-1e-8,1e-8])

fig3 = plt.figure(3, figsize = (8,4))
plt.plot(dRA, PA, 'b', label=r'$P_A$; BTZ')
plt.plot(dRA, np.array(PAg)+np.array(PA), 'b:', label=r'$P_A$; geon')
plt.plot(dRA, PB, 'r', label=r'$P_B$; BTZ')
plt.plot(dRA, np.array(PBg)+np.array(PB), 'r:', label=r'$P_B$; geon')
plt.legend()
plt.xlabel('dRA')

fig4 = plt.figure(4, figsize = (8,4))
plt.plot(dRA, Xbtz, 'c', label=r'$|X|$; BTZ')
plt.plot(dRA, Xgeon, 'c:', label=r'$|X|$; geon')
plt.legend()
plt.xlabel('dRA')

fig5 = plt.figure(5, figsize = (8,4))
plt.plot(dRA, papb, 'g', label=r'$\sqrt{P_A P_B}$; BTZ')
plt.plot(dRA, papbg, 'g:', label=r'$\sqrt{P_A P_B}$; geon')
plt.plot(dRA, Xbtz, 'c', label=r'$|X|$; BTZ')
plt.plot(dRA, Xgeon, 'c:', label=r'$|X|$; geon')
plt.legend(loc=1)
plt.xlabel('dRA')
#plt.ylim([0.40,0.57])

#%%===========================================================================#
#----------------------- CONC vs. DRA FOR DIFF MASSES ------------------------#
#=============================================================================#
sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
DeltaE = 0.1        # Omega
#nmax = 5            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
lam = 1             # coupling constant

dRA = np.linspace(0.05,6,num=80)

#---------------------------------function------------------------------------#
def get_conc_vs_dRA_for_m(M,nmax):
    rh = np.sqrt(M)*l   # radius of horizon
    RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                        # Array for distance from horizon
    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                     + RA - np.sqrt(RA**2-rh**2))
    
    ###
    PA, PB, Xre, Xim = [], [], [], []
    PAg, PBg, Xreg = [], [], []
    
    for n in range(nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(dRA)):
            print(i,end=' ',flush=True)
            if n == 0:    
                PA.append( P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
                PB.append( P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xre.append( XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xim.append( XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
                
                PAg.append( PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
                PBg.append( PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xreg.append( XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            
            else:
                PA[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
                PB[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xre[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xim[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
                
                PAg[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
                PBg[i] += 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xreg[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
        print('')
    
    papb = np.sqrt( np.multiply(np.array(PA),np.array(PB)) )
    papbg = np.sqrt( np.multiply(np.array(PAg)+np.array(PA) , np.array(PBg)+np.array(PB) ) )
    Xbtz = np.sqrt( np.array(Xre)**2 + np.array(Xim)**2 )
    Xgeon = np.sqrt( (np.array(Xre) + np.array(Xreg))**2 + np.array(Xim)**2 )
    
    concbtz = 2 * np.maximum( 0 , Xbtz - papb )
    concgeon = 2 * np.maximum( 0 , Xgeon - papbg )
    return concbtz, concgeon

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('\n==> Mass 0.08 ...')
    cbtz_m008, cgeon_m008 = get_conc_vs_dRA_for_m(0.08, 2)
    print('\n==> Mass 0.06 ...')
    cbtz_m006, cgeon_m006 = get_conc_vs_dRA_for_m(0.06, 2)
    print('\n==> Mass 0.03 ...')
    cbtz_m003, cgeon_m003 = get_conc_vs_dRA_for_m(0.03, 3)
    print('\n==> Mass 0.01 ...')
    cbtz_m001, cgeon_m001 = get_conc_vs_dRA_for_m(0.01, 5)
    print('\n==> Mass 1 ...')
    cbtz_m1, cgeon_m1 = get_conc_vs_dRA_for_m(1, 1)
    print('\n==> Mass 0.1 ...')
    cbtz_m01, cgeon_m01 = get_conc_vs_dRA_for_m(0.1, 2)
    print('\n==> Mass 0.0001 ...')
    cbtz_m0001, cgeon_m0001 = get_conc_vs_dRA_for_m(0.0001, 20)

dirname = "Conc_v_dRA_diffM/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
np.save(dirname+"cbtz_m008_E01b",cbtz_m008)
np.save(dirname+"cbtz_m006_E01b",cbtz_m006)
np.save(dirname+"cbtz_m003_E01b",cbtz_m003)
np.save(dirname+"cbtz_m001_E01b",cbtz_m001)
np.save(dirname+"cbtz_m1_E01b",cbtz_m1)
np.save(dirname+"cbtz_m01_E01b",cbtz_m01)
np.save(dirname+"cbtz_m0001_E01b",cbtz_m0001)
np.save(dirname+"cgeon_m008_E01b",cgeon_m008)
np.save(dirname+"cgeon_m006_E01b",cgeon_m006)
np.save(dirname+"cgeon_m003_E01b",cgeon_m003)
np.save(dirname+"cgeon_m001_E01b",cgeon_m001)
np.save(dirname+"cgeon_m1_E01b",cgeon_m1)
np.save(dirname+"cgeon_m01_E01b",cgeon_m01)
np.save(dirname+"cgeon_m0001_E01b",cgeon_m0001)
np.save(dirname+"dRA_E01b",dRA)

fig = plt.figure(figsize=(9,5))

plt.plot(dRA,cbtz_m008,'r',label=r'$M=0.08$, BTZ')
plt.plot(dRA,cbtz_m006,'orange',label=r'$M=0.06$, BTZ')
plt.plot(dRA,cbtz_m003,'g',label=r'$M=0.03$, BTZ')
plt.plot(dRA,cbtz_m001,'b',label=r'$M=0.01$, BTZ')

plt.plot(dRA,cgeon_m008,'r',linestyle=':',label=r'$M=0.08$, Geon')
plt.plot(dRA,cgeon_m006,'orange',linestyle=':',label=r'$M=0.06$, Geon')
plt.plot(dRA,cgeon_m003,'g',linestyle=':',label=r'$M=0.03$, Geon')
plt.plot(dRA,cgeon_m001,'b',linestyle=':',label=r'$M=0.01$, Geon')

plt.xlabel(r'$d(r_h,R_A)$')
plt.ylabel('Concurrence')
plt.legend()

##%%
##root_papb = []
##for i in range(np.size(dRA)):
##    print(i)
##    root_papb.append(diff_X_PAPB(dRA[i])[0])
#
#fig = plt.figure(figsize = (10,5.5))
##plt.plot(dRA,diffxpapb[0],'b',label=r'$\sqrt{P_A P_B}$ BTZ')
##plt.plot(dRA,diffxpapb[1],'c',label='|X| BTZ')
##plt.plot(dRA,diffxpapb[2],'c--',label='|X| Geon')
##plt.plot([0.2,0.7],[0,0],'c:')
#
#plt.plot(dRA,conc_btz,label='conc btz')
#plt.plot(dRA,conc_geon,label='conc geon')
#
##plt.scatter(ddbtz,0,color='r',label=r'$d_{death}$ BTZ')
##plt.scatter(ddgeon,0,color='r',label=r'$d_{death}$ geon')
#
#plt.xlabel(r'$d(r_h,R_A)$')
##plt.ylabel(r'$\Delta X$')
#plt.ylim([0,0.000001])
##plt.xlim([5,7])
#plt.legend()
#plt.show()
#fig.savefig('conc-btz-geon-HE.pdf',dpi=170)
##plt.xticks([0.250392886,0.250392887,0.250392888])

#%%===========================================================================#
#------------------ CONC vs. COSM. CONST. FOR DIFF MASSES --------------------#
#=============================================================================#
sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 0.1        # Omega

sep *= sig
#l = 10*sig          # cosmological parameter
lam = 1             # coupling constant

dRA = np.linspace(0.05,10,num=99)

#---------------------------------function------------------------------------#
def get_all_vs_dRA_for_l(M,DeltaE,l):
    l *= sig
    rh = np.sqrt(M)*l   # radius of horizon
    nmax = converge_cutoff(M,0.002)
    RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                        # Array for distance from horizon
    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                     + RA - np.sqrt(RA**2-rh**2))
    
    ###
    PA, PB, Xre, Xim = [], [], [], []
    PAg, PBg, Xreg = [], [], []
    
    for n in range(nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(dRA)):
            if i%10 == 0:
                print(i,end=' ',flush=True)
            if n == 0:    
                PA.append( P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
                PB.append( P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xre.append( XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xim.append( XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
                
                PAg.append( 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) )
                PBg.append( 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) )
                Xreg.append( 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) )
            
            else:
                PA[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
                PB[i] += 2*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xre[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xim[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
                
                PAg[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig) 
                PBg[i] += 2*PGEON_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig) 
                Xreg[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig) 
        print('')
    
    papb = np.sqrt( np.multiply(np.array(PA),np.array(PB)) )
    papbg = np.sqrt( np.multiply(np.array(PAg)+np.array(PA) , np.array(PBg)+np.array(PB) ) )
    Xbtz = np.sqrt( np.array(Xre)**2 + np.array(Xim)**2 )
    Xgeon = np.sqrt( (np.array(Xre) + np.array(Xreg))**2 + np.array(Xim)**2 )
    
    concbtz = 2 * np.maximum( 0 , Xbtz - papb )
    concgeon = 2 * np.maximum( 0 , Xgeon - papbg )
    return np.array(PA), np.array(PAg), Xbtz, Xgeon, concbtz, concgeon

dirname = "all_v_dRA_diffl_long/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
    
np.save(dirname+'dRA',dRA)

energies = [1,0.1,0.01]
masses = [1,0.1,0.01]
ls = [100,20,10,8,7,6,5,1]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for DeltaE in energies:
        print('>>>>>>> ENERGY = '+str(DeltaE))
        
        for M in masses:
            print('    >>> MASS = '+str(M))
            
            for l in ls:
                print('      > l = '+str(l))
                fileend = ('_E' + str(DeltaE) + '_M' + str(M) + '_l' + str(l))\
                          .replace('.','')
                #names & execute
                n1, n2, n3, n4, n5, n6 = \
                    'pb'+fileend, 'pg'+fileend, 'xb'+fileend, 'xg'+fileend,\
                    'cb'+fileend, 'cg'+fileend
                execname = '= get_all_vs_dRA_for_l(M,DeltaE,l)'
                inputname = n1+','+n2+','+n3+','+n4+','+n5+','+n6
                exec(inputname+execname)
                
                exec('np.save(dirname+"'+n1+'",'+n1+')')
                exec('np.save(dirname+"'+n2+'",'+n2+')')
                exec('np.save(dirname+"'+n3+'",'+n3+')')
                exec('np.save(dirname+"'+n4+'",'+n4+')')
                exec('np.save(dirname+"'+n5+'",'+n5+')')
                exec('np.save(dirname+"'+n6+'",'+n6+')')

#fig = plt.figure(figsize=(9,5))
#
#plt.plot(dRA,cbtz_m008,'r',label=r'$M=0.08$, BTZ')
#plt.plot(dRA,cbtz_m006,'orange',label=r'$M=0.06$, BTZ')
#plt.plot(dRA,cbtz_m003,'g',label=r'$M=0.03$, BTZ')
#plt.plot(dRA,cbtz_m001,'b',label=r'$M=0.01$, BTZ')
#
#plt.plot(dRA,cgeon_m008,'r',linestyle=':',label=r'$M=0.08$, Geon')
#plt.plot(dRA,cgeon_m006,'orange',linestyle=':',label=r'$M=0.06$, Geon')
#plt.plot(dRA,cgeon_m003,'g',linestyle=':',label=r'$M=0.03$, Geon')
#plt.plot(dRA,cgeon_m001,'b',linestyle=':',label=r'$M=0.01$, Geon')
#
#plt.xlabel(r'$d(r_h,R_A)$')
#plt.ylabel('Concurrence')
#plt.legend()

#%%===========================================================================#
#---------------------- CHECK DELTAP DELTAX WITH LAURA -----------------------#
#=============================================================================#

#sig = 1             # width of Gaussian
#pm1 = 1             # zeta = +1, 0, or -1
#sep = 1             # distance between two detectors in terms of sigma
#DeltaE = 0.1        # Omega
#
##nmax = 5            # summation limit
#l = 10*sig          # cosmological parameter
#M = 0.1
#
#rh = fp.sqrt(M)*l
#sep *= sig
#lam = 1             # coupling constant
#
#dRA = np.linspace(0.1,6,num=80)
#RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#                    # Array for distance from horizon
#RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                 + RA - np.sqrt(RA**2-rh**2))
#
#deltaPA, deltaX = 0*dRA, 0*dRA
#PA, Xre, Xim = 0*dRA, [], []
#
#nmax = converge_cutoff(M,0.001)
#print('nmax is: ', nmax, '\n')
#
#for n in np.arange(0,nmax+1):
#    print('n = ',n,'; i = ', end='')
#    for i in range(np.size(dRA)):
#        print(i,end=' ',flush=True)
#        deltaX[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#        deltaPA[i] += 2*PGEON_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
#        if n == 0:
#            PA[i] += P_BTZn(0,RA[i],rh,l,pm1,DeltaE,lam,sig)
#            Xre.append(XBTZ_n_re(0,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#            Xim.append(XBTZ_n_im(0,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig))
#        else:
#            PA[i] += 2*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
#            Xre[i] += 2*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#            Xim[i] += 2*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
#    print('')
#
#X = np.sqrt(np.array(Xre)**2 + np.array(Xim)**2)
#
#fig1 = plt.figure(1)
#plt.plot(dRA, deltaPA, label=r'$\Delta P_A$')
#plt.xlabel('dRA')
#plt.legend()
#
#fig2 = plt.figure(2)
#plt.plot(dRA, deltaX, label=r'$\Delta X$')
#plt.xlabel('dRA')
#plt.legend()
#
#fig3 = plt.figure(3)
#plt.plot(dRA, PA, label=r'$P_A$; BTZ')
#plt.xlabel('dRA')
#plt.legend()
#
#fig4 = plt.figure(4)
#plt.plot(dRA, Xre, label=r'Re($X$)')
#plt.plot(dRA, Xim, label=r'Im($X$)')
#plt.xlabel('dRA')
#plt.legend()
#
#fig5 = plt.figure(5)
#plt.plot(dRA, X, label=r'$|X|$; BTZ')
#plt.xlabel('dRA')
#plt.legend()
