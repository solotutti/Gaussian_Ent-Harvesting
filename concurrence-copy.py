# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:53:52 2018
concurrence copy for testing
@author: Admin
"""
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
import warnings
from mpmath import mp


#%%===========================================================================#
#======================== BTZ TRANSITION PROBABILITY =========================#
#=============================================================================#

### Using Laura's formula

# First integrand 
def f01(y,n,R,rh,l,pm1,Om,lam,sig):
    return lam**2*sig**2/2 * mp.exp(-sig**2*(y-Om)**2)\
     / (mp.exp(y * 2*mp.pi*l*mp.sqrt(R**2-rh**2)/rh) + 1)

# Second integrand
def f02(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/mp.sqrt(2*mp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = mp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)
    #print(y,end=" ",flush=True)
    if Zp - mp.cosh(y) > 0:
        return K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y))
    else:
        return -K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp)

# First integrand in the sum
def fn1(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/mp.sqrt(2*mp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = mp.sqrt(R**2-rh**2)*Om*l/rh
    Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * mp.cosh(2*mp.pi*rh/l * n) - 1)
    if Zm - mp.cosh(y) > 0:
        return K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zm - mp.cosh(y))
    else:
        return -K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zm)
    
# Second integrand in the sum
def fn2(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/mp.sqrt(2*mp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = mp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 * mp.cosh(2*mp.pi*rh/l * n) + 1)
    if Zp - mp.cosh(y) > 0:
        return K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y))
    else:
        return -K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp)
    
def P_BTZn(n,R,rh,l,pm1,Om,lam,sig):
    b = mp.sqrt(R**2-rh**2)/l
    lim = 20*sig*rh/b/l**2
#    Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * mp.cosh(2*mp.pi*rh/l * n) - 1)
#    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 * mp.cosh(2*mp.pi*rh/l * n) + 1)
    #print('Zm: ',Zm, 'Zp: ',Zp)
    if pm1==-1 or pm1==1 or pm1==0:
        if n==0:
            return mp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-mp.inf,mp.inf])\
                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
#            if lim > mp.acosh(Zp):
#                return mp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-mp.inf,mp.inf])\
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[0,mp.acosh(Zm)])
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acosh(Zm),mp.acosh(Zp)])\
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acos(Zp),lim])
#            elif lim > mp.acosh(Zm):
#                return mp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-mp.inf,mp.inf])\
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[0,mp.acosh(Zm)])\
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acosh(Zm),lim])
#            else:
#                return mp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-mp.inf,mp.inf])\
#                 - mp.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
        else:
            return mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
#            if lim > mp.acosh(Zp):
#                return mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,mp.acosh(Zm)])\
#                        + mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acosh(Zm),mp.acosh(Zp)])\
#                        + mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acosh(Zp),lim])
#            elif lim > mp.acosh(Zm):
#                return mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,mp.acosh(Zm)])\
#                        + mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[mp.acosh(Zm),lim])
#            else:
#                return mp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
#                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#================= GEON ADDITION TO TRANSITION PROBABILITY ===================#
#=============================================================================#

def PGEON_gaussian(x,sig):
    return mp.exp(-x**2/4/sig**2)

def sigma_geon(x,n,R,rh,l):
    return R**2/rh**2 * mp.cosh(2*mp.pi * rh/l * (n+1/2)) - 1 + (R**2-rh**2)/rh**2\
    * mp.cosh(rh/l**2 * x)
    
def h_n(x,n,R,rh,l,pm1):
    return 1/(4*mp.sqrt(2)*mp.pi*l) * (1/mp.sqrt(sigma_geon(x,n,R,rh,l)) \
              - pm1 * 1/mp.sqrt(sigma_geon(x,n,R,rh,l) + 2))

def PGEON_n(n,R,rh,l,pm1,Om,lam,sig):
    """
    Om = energy difference
    lam = coupling constant
    """
    if pm1==-1 or pm1==1 or pm1==0:
        return lam**2/mp.sqrt(2)*sig * mp.exp(-sig**2 * Om**2) *\
        mp.quad(lambda x: h_n(x,n,R,rh,l,pm1) * PGEON_gaussian(x,sig), [-mp.inf, mp.inf])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#=========================== BTZ MATRIX ELEMENT X ============================#
#=============================================================================#

### Using Laura's formula

# Denominators of wightmann functions
def XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) + 1)
                        # Z-plus
    if Zm < mp.cosh(y) and Zp < mp.cosh(y):
        return 0
    elif Zm < mp.cosh(y):
        return - pm1/mp.sqrt(Zp - mp.cosh(y))
    else:
        return 1/mp.sqrt(Zm - mp.cosh(y)) - pm1/mp.sqrt(Zp - mp.cosh(y))

def XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) + 1)
                        # Z-plus
    #print(Zm-np.cosh(y),Zp-np.cosh(y))
    if Zm < mp.cosh(y) and Zp < mp.cosh(y):
        return - ( 1/mp.sqrt(mp.cosh(y) - Zm) - pm1/mp.sqrt(mp.cosh(y) - Zp) )
    elif Zm < mp.cosh(y):
        return - 1/mp.sqrt(mp.cosh(y) - Zm)
    else:
        return 0

# gaussian exponential multiplied by 
def XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*mp.exp(-alp*y**2)*mp.cos(bet*y) * XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

def XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*mp.exp(-alp*y**2)*mp.cos(bet*y) * XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

# Integrate everything
def XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    uplim = 10*sig*mp.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > mp.acosh(Zp):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), mp.acosh(Zp)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zp), uplim])
        return -integral
    elif uplim > mp.acosh(Zm):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), uplim])
        return -integral
    else:
        return -mp.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [-uplim, uplim])/2

def XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    uplim = 10*sig*mp.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > mp.acosh(Zp):
        integral = mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), mp.acosh(Zp)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zp), uplim])
        return -integral
    elif uplim > mp.acosh(Zm):
        integral = mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), uplim])
        return -integral
    else:
        return -mp.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [-uplim, uplim])/2


#=============================================================================#
#=========================== Geon MATRIX ELEMENT X ===========================#
#=============================================================================#
    
# Denominators of wightmann functions
def XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * (n+1/2)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * (n+1/2)) + 1)
                        # Z-plus
    return 1/mp.sqrt(Zm + mp.cosh(y)) - pm1/mp.sqrt(Zp + mp.cosh(y))

def XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*rh*sig/4/l**2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-(bA-bB)**2/2/(bA**2+bB**2)*sig**2*Om**2)
    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2
    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2)
    
    return K*mp.exp(-alp2*y**2)*mp.cos(bet2*Om*y) * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

def XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig):
    return -mp.quad(lambda y: XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig), [0, np.inf])

#=============================================================================#
#================================== D_Death ==================================#
#=============================================================================#
    
def find_zero(f,p1,d):
    """
    Return the zero 'c' of a monotonically increasing or decreasing function.
    p1 is an estimate of where the zero might be.
    """
    print(p1)
    c = p1 - f(p1)*d/(f(p1+d)-f(p1)) # simple
    # if p1 and p2 are close together
    
    if np.abs(c-p1)<d or f(c)==0:
        return c
    else:
        return find_zero(f,c,d)
   
#def find_d_death_BTZ(nmax,sep,rh,l,pm1,Om,lam,sig):
#    def diff_X_PAPB(RA):
#        RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                         + RA - np.sqrt(RA**2-rh**2))
#        Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#        Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig)
#        PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig)
#        PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig)
#        
#        for n in range(1,nmax):
#            Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)
#            Xim += XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)
#            PA += P_BTZn(n,RA,rh,l,pm1,Om,lam,sig)
#            PB += P_BTZn(n,RB,rh,l,pm1,Om,lam,sig)
#        return Xre**2+Xim**2 - PA*PB
#    
#    return find_zero(diff_X_PAPB,1,1e-10)

#=============================================================================#
#============================== Random Functions =============================#
#=============================================================================#

def relative_diff(arr1,arr2):
    return np.divide( np.abs(arr1-arr2) , (arr1+arr2)/2 )


#nmax = 2
#def testX(dRA):
#    RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
#    RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
#                     + RA - mp.sqrt(RA**2-rh**2))
#    X = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#    for i in range(1,nmax):
#        X += 2*XBTZ_n_re(i,RA,RB,rh,l,pm1,Om,lam,sig)
#    return X
#
#find_zero(testX,0.1,1e-6)

#%%

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
n = 0            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant
Om = 1

length = 200
dRA = np.linspace(0.5,0.5000001,num=length)
wtvs = []

print("i = ")
for i in range(length):
    print(i)
    RA = 1/2*np.exp(-dRA[i]/l) * ( rh*np.exp(2*dRA[i]/l) + rh)
    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                         + RA - np.sqrt(RA**2-rh**2))
    wtvs.append(XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig))
    #wtvs.append(mp.sqrt(XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)**2+XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)**2))

plt.plot(dRA,wtvs)

#%%

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

dRA = np.linspace(0.01,3,num=50)
# Plot PA/lam^2 vs dRA from rh to 10rh
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector
                    
#-------------------------------- function -----------------------------------#
def get_conc_vs_dist(DeltaE):
    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
     = 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA
                        # Create y-axis array for BTZ and geon

    # Start summing
    for n in np.arange(0,nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(RA)):
            print(i,' ',end='',flush=True)
            if n == 0:
                fac = 1
            else:
                fac = 2     # when summing nonzero n's, multiply by 2
            PA_btz[i] += fac*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_btz[i] += fac*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_re[i] += fac*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_im[i] += fac*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            
            PA_geon[i] += fac*deltaP_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_geon[i] += fac*deltaP_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    
    X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2 )
    conc_btz = 2*np.maximum(0,np.abs(X_btz) - np.sqrt(np.multiply(PA_btz,PB_btz)))
    
    PA_geon += PA_btz
    PB_geon += PB_btz
    X_geon += X_btz_re       # add transition probability addition from geon to BTZ
    X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
    conc_geon = 2*np.maximum(0,np.abs(X_geon) - np.sqrt(np.multiply(PA_geon,PB_geon)))
    
    return conc_btz, conc_geon
#-----------------------------------------------------------------------------#
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('\n***Energy Difference 0.01...')
    conc_btz_e001, conc_geon_e001 = get_conc_vs_dist(0.01)
    print('\n***Energy Difference 0.1...')
    conc_btz_e01, conc_geon_e01 = get_conc_vs_dist(0.1)
    print('\n***Energy Difference 1...')
    conc_btz_e1, conc_geon_e1 = get_conc_vs_dist(1)

# Plotting
fig = plt.figure(figsize=(9,5))

# btz
plt.plot(dRA,conc_btz_e001,'r', label=r'BTZ; $\Omega\sigma=0.01$')
plt.plot(dRA,conc_btz_e01,'b', label=r'BTZ; $\Omega\sigma=0.1$')
plt.plot(dRA,conc_btz_e1,'y', label=r'BTZ; $\Omega\sigma=1$')

# geon
plt.plot(dRA,conc_geon_e001,'r:', label=r'geon; $\Omega\sigma=0.01$')
plt.plot(dRA,conc_geon_e01,'b:', label=r'geon; $\Omega\sigma=0.1$')
plt.plot(dRA,conc_geon_e1,'y:', label=r'geon; $\Omega\sigma=1$')

plt.legend()
plt.xlabel(r'$d(r_h,R_A)$')
plt.title('Concurrence with proper distance')
#plt.ylim([0.134,0.16])
#plt.xlim([1.075,1.175])
#print('PA_btz: ', PA_btz)
#print('PB_btz: ', PB_btz)
#print('X_btz: ', X_btz)
plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(r'relative difference between concurrences for $\Omega\sigma = 1$',\
          relative_diff(conc_btz_e1,conc_geon_e1))

#%%
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #
#                                   OLD SHIT                                  #

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


#%%===========================================================================#
#======================== BTZ TRANSITION PROBABILITY =========================#
#=============================================================================#

### Using Laura's formula

# First integrand 
def f01(y,n,R,rh,l,pm1,Om,lam,sig):
    return lam**2*sig**2/2 * np.exp(-sig**2*(y-Om)**2)\
     / (np.exp(y * 2*np.pi*l*np.sqrt(R**2-rh**2)/rh) + 1)

# Second integrand
def f02(y,n,R,rh,l,pm1,Om,lam,sig):
#    return (-pm1*lam**2*sig/2/np.sqrt(2*np.pi) * np.exp(-(R**2-rh**2)*l**2\
#     /4/sig**2/rh**2 * y**2)/c.sqrt(rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)\
#     - np.cosh(y)) * np.exp(complex(0,-np.sqrt(R**2-rh**2)*Om*l/rh * y))).real
    K = lam**2*sig/2/np.sqrt(2*np.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = np.sqrt(R**2-rh**2)*Om*l/rh
    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)
    if Zp - np.cosh(y) > 0:
        return K * np.exp(-a*y**2) * np.cos(b*y) / np.sqrt(Zp - np.cosh(y))
    else:
        return -K * np.exp(-a*y**2) * np.sin(b*y) / np.sqrt(np.cosh(y) - Zp)

# First integrand in the sum
def fn1(y,n,R,rh,l,pm1,Om,lam,sig):
#    return (2*lam**2*sig/2/np.sqrt(2*np.pi) * np.exp(-(R**2-rh**2)*l**2\
#     /4/sig**2/rh**2 * y**2)/c.sqrt(rh**2/(R**2-rh**2) * (R**2/rh**2 * np.cosh(\
#     rh*2*np.pi/l * n) - 1) - np.cosh(y)) *\
#     np.exp(complex(0,-np.sqrt(R**2-rh**2)*Om*l/rh * y))).real
    K = lam**2*sig/2/np.sqrt(2*np.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = np.sqrt(R**2-rh**2)*Om*l/rh
    Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * np.cosh(2*np.pi*rh/l * n) - 1)
    if Zm - np.cosh(y) > 0:
        return K * np.exp(-a*y**2) * np.cos(b*y) / np.sqrt(Zm - np.cosh(y))
    else:
        return -K * np.exp(-a*y**2) * np.sin(b*y) / np.sqrt(np.cosh(y) - Zm)
    
# Second integrand in the sum
def fn2(y,n,R,rh,l,pm1,Om,lam,sig):
#    return (-pm1*2*lam**2*sig/2/np.sqrt(2*np.pi) * np.exp(-(R**2-rh**2)*l**2\
#     /4/sig**2/rh**2 * y**2)/c.sqrt(rh**2/(R**2-rh**2) * (R**2/rh**2 * np.cosh(\
#     rh*2*np.pi/l * n) + 1) - np.cosh(y)) *\
#     np.exp(complex(0,-np.sqrt(R**2-rh**2)*Om*l/rh * y))).real
    K = lam**2*sig/2/np.sqrt(2*np.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = np.sqrt(R**2-rh**2)*Om*l/rh
    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 * np.cosh(2*np.pi*rh/l * n) + 1)
    if Zp - np.cosh(y) > 0:
        return K * np.exp(-a*y**2) * np.cos(b*y) / np.sqrt(Zp - np.cosh(y))
    else:
        return -K * np.exp(-a*y**2) * np.sin(b*y) / np.sqrt(np.cosh(y) - Zp)
    
def P_BTZn(n,R,rh,l,pm1,Om,lam,sig):
    b = np.sqrt(R**2-rh**2)/l
    lim = 20*sig*rh/b/l**2
    Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * np.cosh(2*np.pi*rh/l * n) - 1)
    if pm1==-1 or pm1==1 or pm1==0:
        if n==0:
            if np.arccosh(Zm) < lim:
                return integ.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),-np.inf,np.inf)[0]\
                 - integ.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),0,lim, points=[np.arccosh(Zm)])[0]
            else:
                return integ.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),-np.inf,np.inf)[0]\
                 - integ.quad(lambda x: f02(x,n,R,rh,l,pm1,Om,lam,sig),0,lim)[0]
        else:
            if np.arccosh(Zm) < lim:
                return integ.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),0,lim, points=[np.arccosh(Zm)])[0]
            else:
                return integ.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                  - fn2(x,n,R,rh,l,pm1,Om,lam,sig),0,lim)[0]
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#================= GEON ADDITION TO TRANSITION PROBABILITY ===================#
#=============================================================================#

def gaussian4(x,sig):
    return np.exp(-x**2/4/sig**2)

def sigma_geon(x,n,R,rh,l):
    return R**2/rh**2 * np.cosh(2*np.pi * rh/l * (n+1/2)) - 1 + (R**2-rh**2)/rh**2\
    * np.cosh(rh/l**2 * x)
    
def h_n(x,n,R,rh,l,pm1):
    return 1/(4*np.sqrt(2)*np.pi*l) * (1/np.sqrt(sigma_geon(x,n,R,rh,l)) \
              - pm1 * 1/np.sqrt(sigma_geon(x,n,R,rh,l) + 2))

def deltaP_n(n,R,rh,l,pm1,Om,lam,sig):
    """
    Om = energy difference
    lam = coupling constant
    """
    if pm1==-1 or pm1==1 or pm1==0:
        return lam**2/np.sqrt(2)*sig * np.exp(-sig**2 * Om**2) *\
        integ.quad(lambda x: h_n(x,n,R,rh,l,pm1) * gaussian4(x,sig), -np.inf, np.inf)[0]
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#=============================================================================#
#=========================== BTZ MATRIX ELEMENT X ============================#
#=============================================================================#

### Using Laura's formula

# Denominators of wightmann functions
def XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    #print(Zm-np.cosh(y),Zp-np.cosh(y))
    if Zm < np.cosh(y) and Zp < np.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y)) - pm1/c.sqrt(Zp - np.cosh(y))).real
        return 0
        #return 1/np.sqrt(np.cosh(y) - Zm) - pm1/np.sqrt(np.cosh(y) - Zp)
    elif Zm < np.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y))).real - pm1/np.sqrt(Zp - np.cosh(y))
        return - pm1/np.sqrt(Zp - np.cosh(y))
    else:
        return 1/np.sqrt(Zm - np.cosh(y)) - pm1/np.sqrt(Zp - np.cosh(y))

def XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    #print(Zm-np.cosh(y),Zp-np.cosh(y))
    if Zm < np.cosh(y) and Zp < np.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y)) - pm1/c.sqrt(Zp - np.cosh(y))).real
        return - ( 1/np.sqrt(np.cosh(y) - Zm) - pm1/np.sqrt(np.cosh(y) - Zp) )
        #return 1/np.sqrt(np.cosh(y) - Zm) - pm1/np.sqrt(np.cosh(y) - Zp)
    elif Zm < np.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y))).real - pm1/np.sqrt(Zp - np.cosh(y))
        return - 1/np.sqrt(np.cosh(y) - Zm)
    else:
        return 0

# gaussian exponential multiplied by 
def XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/np.sqrt(np.pi)*np.sqrt(bA*bB/(bA**2+bB**2))\
         * np.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*np.exp(-alp*y**2)*np.cos(bet*y) * XBTZ_denoms_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

def XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/np.sqrt(np.pi)*np.sqrt(bA*bB/(bA**2+bB**2))\
         * np.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*np.exp(-alp*y**2)*np.cos(bet*y) * XBTZ_denoms_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

# Integrate everything
def XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) - 1)
                        # Z-minus
    #Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    uplim = 10*sig*np.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    #print(np.arccosh(Zm))
    if uplim > np.arccosh(Zm):
        return -integ.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           -uplim, uplim, points=[-np.arccosh(Zm),np.arccosh(Zm)])[0]/2
    else:
        return -integ.quad(lambda y: XBTZ_integrand_n_re(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           -uplim, uplim)[0]/2

def XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) - 1)
                        # Z-minus
    #Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    uplim = 10*sig*np.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > np.arccosh(Zm):
        return -integ.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           -uplim, uplim, points=[-np.arccosh(Zm),np.arccosh(Zm)])[0]/2
    else:
        return -integ.quad(lambda y: XBTZ_integrand_n_im(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           -uplim, uplim)[0]/2

#=============================================================================#
#=========================== Geon MATRIX ELEMENT X ===========================#
#=============================================================================#
    
# Denominators of wightmann functions
def XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * (n+1/2)) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * (n+1/2)) + 1)
                        # Z-plus
    return 1/np.sqrt(Zm + np.cosh(y)) - pm1/np.sqrt(Zp + np.cosh(y))

def XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = np.sqrt(RA**2-rh**2)/l
    bB = np.sqrt(RB**2-rh**2)/l
    K = lam**2*rh*sig/4/l**2/np.sqrt(np.pi)*np.sqrt(bA*bB/(bA**2+bB**2))\
         * np.exp(-(bA-bB)**2/2/(bA**2+bB**2)*sig**2*Om**2)
    alp2 = bA**2*bB**2/2/(bA**2+bB**2)/sig**2
    bet2 = (bA+bB)*bA*bB/(bA**2+bB**2)
    
    return K*np.exp(-alp2*y**2)*np.cos(bet2*Om*y) * XGEON_denoms_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig)

def XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,sig):
    return -integ.quad(lambda y: XGEON_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig), 0, np.inf)[0]

#=============================================================================#
#================================== D_Death ==================================#
#=============================================================================#
    
def find_zero(f,p1,d):
    """
    Return the zero 'c' of a monotonically increasing or decreasing function.
    p1 is an estimate of where the zero might be.
    """
    print(p1)
    c = p1 - f(p1)*d/(f(p1+d)-f(p1)) # simple
    # if p1 and p2 are close together
    
    if np.abs(c-p1)<d or f(c)==0:
        return c
    else:
        return find_zero(f,c,d)
   
#def find_d_death_BTZ(nmax,sep,rh,l,pm1,Om,lam,sig):
#    def diff_X_PAPB(RA):
#        RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                         + RA - np.sqrt(RA**2-rh**2))
#        Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#        Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig)
#        PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig)
#        PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig)
#        
#        for n in range(1,nmax):
#            Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)
#            Xim += XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)
#            PA += P_BTZn(n,RA,rh,l,pm1,Om,lam,sig)
#            PB += P_BTZn(n,RB,rh,l,pm1,Om,lam,sig)
#        return Xre**2+Xim**2 - PA*PB
#    
#    return find_zero(diff_X_PAPB,1,1e-10)

#=============================================================================#
#============================== Random Functions =============================#
#=============================================================================#

def relative_diff(arr1,arr2):
    return np.divide( np.abs(arr1-arr2) , (arr1+arr2)/2 )


#%%=========================================================================%%#
#======================== COMPARE PD FOR BTZ AND GEON ========================#
#=============================================================================#
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
DeltaE = .1          # Omega

l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

# Plot PA/lam^2 vs R from rh to 10rh

R = np.linspace(rh*1.0005,1.05*rh,num=50)
                    # Array for distance from horizon
dR = l*np.log((R + np.sqrt(R**2 - rh**2))/rh)
                    # proper distance of the closest detector

P_btz, P_geon = 0*R, 0*R
                    # Create y-axis array for BTZ and geon
nmax = 2            # summation limit
for n in np.arange(0,nmax+1):
    for i in range(len(P_btz)):
        if n == 0:
            fac = 1
        else:
            fac = 2     # when summing nonzero n's, multiply by 2
        P_btz[i] += fac*P_BTZn(n,R[i],rh,l,pm1,DeltaE,lam,sig)
        P_geon[i] += fac*deltaP_n(n,R[i],rh,l,pm1,DeltaE,lam,sig)

P_geon += P_btz       # add transition probability addition from geon to BTZ

# Plotting
plt.figure()
plt.plot(dR,P_btz,label='BTZ')
plt.plot(dR,P_geon,label='geon')
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
DeltaE = 1          # Omega

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

# Plot PA/lam^2 vs R from rh to 10rh
RA = np.linspace(rh*1.001,1.1*rh,num=20)
                    # Array for distance from horizon
RB = 1/2*np.exp(2*sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector

X_btz_re, X_btz_im, X_geon = 0*RA, 0*RA
                    # Create y-axis array for BTZ and geon
nmax = 2            # summation limit
for n in np.arange(0,nmax+1):
    for i in range(len(RA)):
        if n == 0:
            fac = 1
        else:
            fac = 2     # when summing nonzero n's, multiply by 2
        X_btz_re[i] += fac*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        X_btz_im[i] += fac*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)

X_geon += X_btz_re       # add transition probability addition from geon to BTZ
X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2)

print('')
print('discrepancy: ',X_geon-X_btz)
print('discrepancy/X_btz: ', np.divide(X_geon-X_btz,X_btz))
# Plotting
plt.figure()
plt.plot(RA,X_btz,label='BTZ')
plt.plot(RA,X_geon,label='geon')
plt.legend()
plt.title('X Element')
plt.show()

#%%===========================================================================#
#===================== COMPARE CONCURRENCE VS. DISTANCE ======================#
#=============================================================================#

sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
DeltaE = .01          # Omega
nmax = 2            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

dRA = np.linspace(0.01,3,num=50)
# Plot PA/lam^2 vs dRA from rh to 10rh
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector

PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
 = 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA
                    # Create y-axis array for BTZ and geon

# Start summing
for n in np.arange(0,nmax+1):
    print('n =',n)
    print(' i = ', end='')
    for i in range(len(RA)):
        print(i,' ',end='',flush=True)
        if n == 0:
            fac = 1
        else:
            fac = 2     # when summing nonzero n's, multiply by 2
        PA_btz[i] += fac*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
        PB_btz[i] += fac*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
        X_btz_re[i] += fac*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        X_btz_im[i] += fac*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        
        PA_geon[i] += fac*deltaP_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
        PB_geon[i] += fac*deltaP_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
        X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
    print('')

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
#plt.ylim([0.134,0.16])
#plt.xlim([1.075,1.175])
#print('PA_btz: ', PA_btz)
#print('PB_btz: ', PB_btz)
#print('X_btz: ', X_btz)
plt.show()

print('relative difference between concurrence',\
      np.divide( np.abs( conc_btz - conc_geon), (conc_btz+conc_geon)/2))

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

dRA = np.linspace(0.01,3,num=50)
# Plot PA/lam^2 vs dRA from rh to 10rh
RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
                    # Array for distance from horizon
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Distance of the further detector
                    
#-------------------------------- function -----------------------------------#
def get_conc_vs_dist(DeltaE):
    PA_btz, PA_geon, PB_btz, PB_geon, X_btz_re, X_btz_im, X_geon\
     = 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA, 0*RA
                        # Create y-axis array for BTZ and geon

    # Start summing
    for n in np.arange(0,nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(RA)):
            print(i,' ',end='',flush=True)
            if n == 0:
                fac = 1
            else:
                fac = 2     # when summing nonzero n's, multiply by 2
            PA_btz[i] += fac*P_BTZn(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_btz[i] += fac*P_BTZn(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_re[i] += fac*XBTZ_n_re(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_btz_im[i] += fac*XBTZ_n_im(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
            
            PA_geon[i] += fac*deltaP_n(n,RA[i],rh,l,pm1,DeltaE,lam,sig)
            PB_geon[i] += fac*deltaP_n(n,RB[i],rh,l,pm1,DeltaE,lam,sig)
            X_geon[i] += fac*XGEON_n(n,RA[i],RB[i],rh,l,pm1,DeltaE,lam,sig)
        print('')
    
    X_btz = np.sqrt( X_btz_re**2 + X_btz_im**2 )
    conc_btz = 2*np.maximum(0,np.abs(X_btz) - np.sqrt(np.multiply(PA_btz,PB_btz)))
    
    PA_geon += PA_btz
    PB_geon += PB_btz
    X_geon += X_btz_re       # add transition probability addition from geon to BTZ
    X_geon = np.sqrt(X_geon**2 + X_btz_im**2)
    conc_geon = 2*np.maximum(0,np.abs(X_geon) - np.sqrt(np.multiply(PA_geon,PB_geon)))
    
    return conc_btz, conc_geon
#-----------------------------------------------------------------------------#
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('\n***Energy Difference 0.01...')
    conc_btz_e001, conc_geon_e001 = get_conc_vs_dist(0.01)
    print('\n***Energy Difference 0.1...')
    conc_btz_e01, conc_geon_e01 = get_conc_vs_dist(0.1)
    print('\n***Energy Difference 1...')
    conc_btz_e1, conc_geon_e1 = get_conc_vs_dist(1)

# Plotting
fig = plt.figure(figsize=(9,5))

# btz
plt.plot(dRA,conc_btz_e001,'r', label=r'BTZ; $\Omega\sigma=0.01$')
plt.plot(dRA,conc_btz_e01,'b', label=r'BTZ; $\Omega\sigma=0.1$')
plt.plot(dRA,conc_btz_e1,'y', label=r'BTZ; $\Omega\sigma=1$')

# geon
plt.plot(dRA,conc_geon_e001,'r:', label=r'geon; $\Omega\sigma=0.01$')
plt.plot(dRA,conc_geon_e01,'b:', label=r'geon; $\Omega\sigma=0.1$')
plt.plot(dRA,conc_geon_e1,'y:', label=r'geon; $\Omega\sigma=1$')

plt.legend()
plt.xlabel(r'$d(r_h,R_A)$')
plt.title('Concurrence with proper distance')
#plt.ylim([0.134,0.16])
#plt.xlim([1.075,1.175])
#print('PA_btz: ', PA_btz)
#print('PB_btz: ', PB_btz)
#print('X_btz: ', X_btz)
plt.show()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(r'relative difference between concurrences for $\Omega\sigma = 1$',\
          relative_diff(conc_btz_e1,conc_geon_e1))

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
     0*Om, 0*Om, 0*Om, 0*Om, 0*Om, 0*Om, 0*Om
                        # Create y-axis array for BTZ and geon

    # Start summing
    for n in np.arange(0,nmax+1):
        print('n =',n)
        print(' i = ', end='')
        for i in range(len(Om)):
            print(i,' ',end='',flush=True)
            if n == 0:
                fac = 1
            else:
                fac = 2     # when summing nonzero n's, multiply by 2
            PA_btz[i] += fac*P_BTZn(n,RA,rh,l,pm1,Om[i],lam,sig)
            PB_btz[i] += fac*P_BTZn(n,RB,rh,l,pm1,Om[i],lam,sig)
            X_btz_re[i] += fac*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
            X_btz_im[i] += fac*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
    #    print('real ',X_btz_re)
    #    print('im ',X_btz_im)
            PA_geon[i] += fac*deltaP_n(n,RA,rh,l,pm1,Om[i],lam,sig)
            PB_geon[i] += fac*deltaP_n(n,RB,rh,l,pm1,Om[i],lam,sig)
            X_geon[i] += fac*XGEON_n(n,RA,RB,rh,l,pm1,Om[i],lam,sig)
        print('')
    
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

#%%===========================================================================#
#=========================== D_DEATH VERSUS ENERGY ===========================#
#=============================================================================#
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
nmax = 2            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant

#Om = np.linspace(0.01,3,num=50)


#%%===========================================================================#
#=================================== TEST ====================================#
#=============================================================================#
#def f022(y,n,R,rh,l,pm1,Om,lam,sig):
#    K = lam**2*sig/2/np.sqrt(2*np.pi)
#    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
#    b = np.sqrt(R**2-rh**2)*Om*l/rh
#    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)
#    
#    return (K * np.exp(-a*y**2) * c.exp(complex(0,-b*y)) / c.sqrt(Zp - np.cosh(y))).real
#
#def f023(y,n,R,rh,l,pm1,Om,lam,sig):
##    return (-pm1*lam**2*sig/2/np.sqrt(2*np.pi) * np.exp(-(R**2-rh**2)*l**2\
##     /4/sig**2/rh**2 * y**2)/c.sqrt(rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)\
##     - np.cosh(y)) * np.exp(complex(0,-np.sqrt(R**2-rh**2)*Om*l/rh * y))).real
#    g = np.sqrt(R**2 - rh**2)/l
#    K = lam**2*sig/2/np.sqrt(2*np.pi)
#    a = g**2*l**4/4/sig**2/rh**2
#    b = g*Om*l**2/rh
#    Zp = rh**2/(R**2-rh**2)*(R**2/rh**2 + 1)
#    if Zp - np.cosh(y) > 0:
#        return K * np.exp(-a*y**2) * np.cos(b*y) / np.sqrt(Zp - np.cosh(y))
#    else:
#        return -K * np.exp(-a*y**2) * np.sin(b*y) / np.sqrt(np.cosh(y) - Zp)
#
#def Xbtz_nlim(n,RA,RB,rh,l,pm1,Om,lam,sig,lim):
#    bA = np.sqrt(RA**2-rh**2)/l
#    bB = np.sqrt(RB**2-rh**2)/l
#    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) - 1)
#                        # Z-minus
#    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*np.cosh(2*np.pi*rh/l * n) + 1)
#                        # Z-plus
#    return integ.quad(lambda y: Xbtz_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig),0, Zm-lim)[0]\
#        + integ.quad(lambda y: Xbtz_integrand_n(y,n,RA,RB,rh,l,pm1,Om,lam,sig),Zm+lim, Zp)[0]

#def f1(x):
#    return x**2-1
#
#def find_zero(f,p1,p2,d):
#    """
#    Return value q s.t. p1<q<p2 where func becomes positive.
#    """
#    print(p1,p2)
#    c = (p1 + p2) / 2 # simple
#    # if p1 and p2 are close together
#    if p2-p1 < d or f(c) == 0:
#        return c
#    elif f(p1)*f(c) < 0:
#        return find_zero(f,p1,c,d)
#    else:
#        return find_zero(f,c,p2,d)


#def diff_X_PAPB(dRA):
#    RA = 1/2*np.exp(-dRA/l) * ( rh*np.exp(2*dRA/l) + rh)
#    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
#                     + RA - np.sqrt(RA**2-rh**2))
#    Xre = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#    Xim = XBTZ_n_im(0,RA,RB,rh,l,pm1,Om,lam,sig)
#    PA = P_BTZn(0,RA,rh,l,pm1,Om,lam,sig)
#    PB = P_BTZn(0,RB,rh,l,pm1,Om,lam,sig)
#    
#    for n in range(1,nmax):
#        Xre += 2*XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)
#        Xim += 2*XBTZ_n_im(n,RA,RB,rh,l,pm1,Om,lam,sig)
#        PA += 2*P_BTZn(n,RA,rh,l,pm1,Om,lam,sig)
#        PB += 2*P_BTZn(n,RB,rh,l,pm1,Om,lam,sig)
#    return Xre**2+Xim**2 - PA*PB
#    
#find_zero(diff_X_PAPB,0.2,1e-7)

#%%===========================================================================#
#================================== TEST 2 ===================================#
#=============================================================================#
"""
Get a better numerical integration so funnction looks less rugged.
"""
import numpy as np

### Starting with X_BTZ_n_re

def XBTZ_denoms_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) + 1)
                        # Z-plus
    #print(Zm-np.cosh(y),Zp-np.cosh(y))
    if Zm < mp.cosh(y) and Zp < mp.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y)) - pm1/c.sqrt(Zp - np.cosh(y))).real
        return 0
        #return 1/np.sqrt(np.cosh(y) - Zm) - pm1/np.sqrt(np.cosh(y) - Zp)
    elif Zm < mp.cosh(y):
        #return (1/c.sqrt(Zm - np.cosh(y))).real - pm1/np.sqrt(Zp - np.cosh(y))
        return - pm1/mp.sqrt(Zp - mp.cosh(y))
    else:
        return 1/mp.sqrt(Zm - mp.cosh(y)) - pm1/mp.sqrt(Zp - mp.cosh(y))

def XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    K = lam**2*sig/2/mp.sqrt(mp.pi)*mp.sqrt(bA*bB/(bA**2+bB**2))\
         * mp.exp(-sig**2*Om**2/2 * (bA+bB)**2/(bA**2+bB**2))
    alp = 1/2/sig**2 * bA**2*bB**2/(bA**2+bB**2) * l**4/rh**2
    bet = Om * bA*bB*(bB-bA)/(bA**2+bB**2) * l**2/rh
    return K*mp.exp(-alp*y**2)*mp.cos(bet*y) * XBTZ_denoms_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig)
    
def XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    Zm = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*mp.pi*rh/l * n) - 1)
                        # Z-minus
    Zp = rh**2/l**2/bA/bB * (RA*RB/rh**2*mp.cosh(2*np.pi*rh/l * n) + 1)
                        # Z-plus
    uplim = 10*sig*mp.sqrt(bA**2+bB**2)*rh/(bA*bB*l**2)
    if uplim > mp.acosh(Zp):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), mp.acosh(Zp)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zp), uplim])
        #print("precision of integration: ",int1[1],int2[1])
        return -integral
    elif uplim > mp.acosh(Zm):
        integral = mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [0, mp.acosh(Zm)])
        integral += mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [mp.acosh(Zm), uplim])
        #print("precision of integration: ",int1[1],int2[1])
        return -integral
    else:
        return -mp.quad(lambda y: XBTZ_integrand_n_re2(y,n,RA,RB,rh,l,pm1,Om,lam,sig),\
                           [-uplim, uplim])/2
                           
sig = 1             # width of Gaussian
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
#dRA = 100             # proper distance of RA from the horizon
n = 0            # summation limit

sep *= sig
l = 10*sig          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant
Om = 1

#nmax = 2
#def testX(dRA):
#    RA = 1/2*mp.exp(-dRA/l) * ( rh*mp.exp(2*dRA/l) + rh)
#    RB = 1/2*mp.exp(-sep/l) * ( (RA + mp.sqrt(RA**2-rh**2))*mp.exp(2*sep/l)\
#                     + RA - mp.sqrt(RA**2-rh**2))
#    X = XBTZ_n_re(0,RA,RB,rh,l,pm1,Om,lam,sig)
#    for i in range(1,nmax):
#        X += 2*XBTZ_n_re(i,RA,RB,rh,l,pm1,Om,lam,sig)
#    return X
#
#find_zero(testX,0.1,1e-6)

length = 200
dRA = np.linspace(0.5,0.500001,num=length)
Xreal = 0*dRA

print("i = ")
for i in range(length):
    print(i)
    RA = 1/2*np.exp(-dRA[i]/l) * ( rh*np.exp(2*dRA[i]/l) + rh)
    RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                         + RA - np.sqrt(RA**2-rh**2))
    Xreal[i] = XBTZ_n_re(n,RA,RB,rh,l,pm1,Om,lam,sig)

plt.plot(dRA,Xreal)

#%%
def g(x):
    return integ.quad(lambda y: (y**2-np.sin(x))/y**(-1/3), -0.2, 1)[0]
xarr = np.linspace(0.5,0.50001,num=100)
yarr = [g(xi) for xi in xarr]

plt.plot(xarr,yarr)

