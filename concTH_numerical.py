# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:43:18 2018
        Compute Concurrence for Top Hat switching function
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
#================================ FUNCTIONS ==================================#
#=============================================================================#

# Wightman function in BTZ of DeltaPhi and y=t-t'

def sigma_btz_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    if isP:
        return RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1\
                - fp.sqrt(RA**2-rh**2)*fp.sqrt(RB**2-rh**2)/rh**2\
                  * fp.mpf(mp.cosh(rh/l/mp.sqrt(RA*RB-rh**2) * y) )
    else:
        return RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1\
                - fp.sqrt(RA**2-rh**2)*fp.sqrt(RB**2-rh**2)/rh**2\
                  * fp.mpf(mp.cosh(rh/l**2 * y) )

def g_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    sigma = sigma_btz_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP)
    #print('y',y,' sigma',sigma)
    if n == 0 and deltaphi == 0 and RA==RB:        #Deal with this case separately
        return 0
    elif sigma == 0:
        return 0
    elif sigma > 0:
        return 1/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(sigma)
    else:
        return -fp.j/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(-sigma) * fp.sign(y)

def g_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    sigma = sigma_btz_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP) + 2
    if sigma == 0:
        return 0
    elif sigma > 0:
        return 1/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(sigma)
    else:
        return -fp.j/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(-sigma) * fp.sign(y)

def g_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    return g_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP) - pm1 * g_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP)

# Wightman function in geon of Deltaphi and y = t+t'

def sigma_geon_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    if isP:
        return RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1\
                + fp.sqrt(RA**2-rh**2)*fp.sqrt(RB**2-rh**2)/rh**2 \
                  * fp.mpf(mp.cosh(rh/l/mp.sqrt(RA*RB-rh**2) * y) )
    else:
        return RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1\
                + fp.sqrt(RA**2-rh**2)*fp.sqrt(RB**2-rh**2)/rh**2 \
                  * fp.mpf(mp.cosh(rh/l**2 * y) )

def h_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    return 1/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(sigma_geon_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP))

def h_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    return 1/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(sigma_geon_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP) + 2)

def h_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP):
    return h_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP) - pm1 * h_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP)

#########
# P_BTZ #
#########
    
def integrandof_P_BTZn(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    return 2*lam**2 * (fp.exp(-fp.j*Om*y) * g_n(y,n,RA,RB,rh,l,pm1,deltaphi,isP)).real * (width-y)

def integrandPBTZ_0m(y,R,rh,l,pm1,Om,lam,tau0,width,eps):
    #print( mp.sinh(rh/2/l**2 * (y - fp.j*eps)) )
    return (- fp.j*rh*lam**2 / (8*fp.pi*l * fp.sqrt(R**2-rh**2))\
            * fp.exp(-fp.j*Om*y)/mp.sinh(rh/2/l**2 * (y - fp.j*eps)) * (width - np.abs(y))).real

def P_BTZn(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi=0,eps=1e-8,isP=True):
    f = lambda y: integrandof_P_BTZn(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)
    
    if n==0 and deltaphi==0 and RA==RB:
        # P0m
        P0m = 2 * fp.quad(lambda x: integrandPBTZ_0m(x,RA,rh,l,pm1,Om,lam,tau0,width,eps), [0, width])

        intlim = l**2/rh * mp.acosh( (RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) + 1)\
                / fp.sqrt(RA**2-rh**2)/fp.sqrt(RB**2-rh**2)*rh**2) 
        if intlim < width:
            return fp.quad(f, [0,intlim]) + fp.quad(f, [intlim,width]) + P0m
        else:
            return fp.quad(f, [0,width]) + P0m
        
    else:

        intlim1 = l**2/rh * mp.acosh( (RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1)\
                / fp.sqrt(RA**2-rh**2)/fp.sqrt(RB**2-rh**2)*rh**2) 
        intlim2 = l**2/rh * mp.acosh( (RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) + 1)\
                / fp.sqrt(RA**2-rh**2)/fp.sqrt(RB**2-rh**2)*rh**2) 
        #print('%.4f, %.4f'%(intlim1,intlim2))
        if intlim2 < width:
            return fp.quad(f, [0,intlim1]) + fp.quad(f, [intlim1,intlim2]) + fp.quad(f, [intlim2,width])
        elif intlim1 < width:
            return fp.quad(f, [0,intlim1]) + fp.quad(f, [intlim1,width])
        else:
            return fp.quad(f, [0,width])

#########
# PGEON #
#########

def integrandof_PGEON_n(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    return lam**2/Om * fp.sin(Om*y)\
             * ( h_n(y+2*tau0,n,RA,RB,rh,l,pm1,deltaphi,isP) - h_n(2*(tau0+width)-y,n,RA,RB,rh,l,pm1,deltaphi,isP) )

def PGEON_n(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi=0,isP=True):
    return fp.quad(lambda y: integrandof_PGEON_n(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)\
                   , [0,width])

#########
# X_BTZ #
#########
    
def integrandof_Xn_minus(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    return fp.j*4 * fp.exp(-fp.j*Om*(bB+bA)*(tau0+width/2))/(Om*(bB+bA))\
            * fp.cos( Om/2*(bB+bA)*(y-width) ) * fp.cos(Om/2*(bB-bA)*y)\
            * g_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP)

def Xn_minus(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    
    intlim1 = l**2/rh * mp.acosh( (RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) - 1)\
                / fp.sqrt(RA**2-rh**2)/fp.sqrt(RB**2-rh**2)*rh**2) 
    #print('minus, %.4f ;'%intlim1, end=' ')
    f = lambda y: integrandof_Xn_minus(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)
    
    if intlim1 < width:
        return fp.quad(f, [0,intlim1]) + fp.quad(f, [intlim1, width])
    else:
        return fp.quad(f, [0, width])

def integrandof_Xn_plus(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    return fp.j*4 * fp.exp(-fp.j*Om*(bB+bA)*(tau0+width/2))/(Om*(bB+bA))\
            * fp.cos( Om/2*(bB+bA)*(y-width) ) * fp.cos(Om/2*(bB-bA)*y)\
            * g_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP)

def Xn_plus(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    
    intlim2 = l**2/rh * mp.acosh( (RA*RB/rh**2 * fp.mpf( mp.cosh(rh/l * (deltaphi-2*fp.pi*n)) ) + 1)\
                / fp.sqrt(RA**2-rh**2)/fp.sqrt(RB**2-rh**2)*rh**2) 
    #print('plus, %.4f ;'%intlim2)
    f = lambda y: integrandof_Xn_plus(y,n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)
    
    if intlim2 < width:
        return fp.quad(f, [0,intlim2]) + fp.quad(f, [intlim2, width])
    else:
        return fp.quad(f, [0, width])

def XBTZ_n(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi=0,isP=False):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    tau0 *= bA
    width *= bA
    
    return -lam**2*bA*bB * (Xn_minus(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)\
                            - pm1*Xn_plus(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP))

#########
# XGEON #
#########

def XG_minusAB(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    f = lambda y: 2/Om/(bB-bA) * fp.exp(-fp.j*Om/2*(bB+bA)*y)\
                    * h_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP)\
                    * fp.sin( Om/2*(bB-bA)*(y-2*tau0))
    return fp.quad(f, [2*tau0, 2*tau0+width])

def XG_minusBA(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    f = lambda y: 2/Om/(bB-bA) * fp.exp(-fp.j*Om/2*(bB+bA)*y)\
                    * h_n1(y,n,RA,RB,rh,l,pm1,deltaphi,isP)\
                    * fp.sin( Om/2*(bB-bA)*(2*(tau0+width)-y))
    return fp.quad(f, [2*tau0 + width, 2*(tau0+width)])

def XG_plusAB(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    f = lambda y: 2/Om/(bB-bA) * fp.exp(-fp.j*Om/2*(bB+bA)*y)\
                    * h_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP)\
                    * fp.sin( Om/2*(bB-bA)*(y-2*tau0))
    return fp.quad(f, [2*tau0, 2*tau0+width])

def XG_plusBA(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    f = lambda y: 2/Om/(bB-bA) * fp.exp(-fp.j*Om/2*(bB+bA)*y)\
                    * h_n2(y,n,RA,RB,rh,l,pm1,deltaphi,isP)\
                    * fp.sin( Om/2*(bB-bA)*(2*(tau0+width)-y))
    return fp.quad(f, [2*tau0 + width, 2*(tau0+width)])

def XGEON_n(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi=0,isP=False):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    tau0 *= bA
    width *= bA
    
    return -lam**2*bA*bB * ( (XG_minusAB(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)\
                              - XG_minusBA(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP))\
                       - pm1*(XG_plusAB(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)\
                              - XG_plusBA(n,RA,RB,rh,l,pm1,Om,lam,tau0,width,deltaphi,isP)))

### P_BTZ_0 GAUSSIAN

def integrandPBTZ_0m_GAUSS(y,R,rh,l,pm1,Om,lam,sig,eps):
    #print(mp.sqrt(1 - mp.cosh( y -fp.j*eps )))
    return fp.exp(-fp.j*fp.sqrt(R**2-rh**2)*Om*l/rh * y)\
            / mp.sqrt(1 - mp.cosh( y -fp.j*eps ))\
            * lam**2*sig/(2*fp.sqrt(2*fp.pi))\
            * fp.exp(-(R**2-rh**2)*l**2/(4*sig**2*rh**2) * y**2)

def P0m_GAUSS(R,rh,l,pm1,Om,lam,sig,eps=1e-8):
    f = lambda y: integrandPBTZ_0m_GAUSS(y,R,rh,l,pm1,Om,lam,sig,eps)
    return fp.quad(f, [0, fp.inf])
#%%===========================================================================#
#=============================== Check things ================================#
#=============================================================================#
sig = 5
tau0 = 0
width = 2*np.sqrt(2*sig**2*np.log(2))       # width of tophat
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
Om = 1          # Omega

l = 10          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant
sep = 1
# Plot PA/lam^2 vs R from rh to 10rh

dR = np.linspace(0.05,25,num=800)
                    # proper distance of the closest detector

RA = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))
                    # Array for distance from horizon

nmax = 0            # summation limit

#y = np.linspace(-0.0001,10,num=60)
#checkfcn = []
#checkfcn2 = []
#for i in range(np.size(y)):
#    print(i, end=' ',flush=True)
#    Ri = R[61]
#    #checkfcn.append( integrandof_P_BTZn(y[i],0,Ri,Ri,rh,l,pm1,Om,lam,tau0,width,0) )
#    checkfcn.append( (g_n1(y[i],0,Ri,Ri,rh,l,pm1,0)*fp.exp(fp.j*Om*y[i])).real )
#    checkfcn2.append( (g_n1(y[i],0,Ri,Ri,rh,l,pm1,0)*fp.exp(fp.j*Om*y[i])).imag )
#plt.plot(y,checkfcn,label='real of g')
##plt.plot(y,checkfcn2,label='imag of g')
#plt.legend()

P0_gauss = []
P_btz, P_geon = [], []
X_btz, X_geon = [], []
PB_btz, PB_geon = [], []
                    # Create y-axis array for BTZ and geon
for n in np.arange(0,nmax+1):
    print('n = ',n,'; i = ', end='')
    for i in range(np.size(dR)):
        print(i,end=' ',flush=True)
        bA = fp.sqrt(RA[i]**2-rh**2)/l
        bB = fp.sqrt(RB[i]**2-rh**2)/l
        tau0B = tau0*bB/bA
        widthB = width*bB/bA
        if n == 0:
            P0_gauss.append(P0m_GAUSS(RA[i],rh,l,pm1,Om,lam,sig).real)
            P_btz.append(P_BTZn(0,RA[i],RA[i],rh,l,pm1,Om,lam,tau0,width))
            P_geon.append(2*PGEON_n(0,RA[i],RA[i],rh,l,pm1,Om,lam,tau0,width))
            PB_btz.append(P_BTZn(0,RB[i],RB[i],rh,l,pm1,Om,lam,tau0B,widthB))
            PB_geon.append(2*PGEON_n(0,RB[i],RB[i],rh,l,pm1,Om,lam,tau0B,widthB))
            X_btz.append(XBTZ_n(0,RA[i],RB[i],rh,l,pm1,Om,lam,tau0,width))
            X_geon.append(XGEON_n(0,RA[i],RB[i],rh,l,pm1,Om,lam,tau0,width))
        else:
            P_btz[i] += 2*P_BTZn(n,RA[i],RA[i],rh,l,pm1,Om,lam,tau0,width)
            P_geon[i] += 2*PGEON_n(n,RA[i],RA[i],rh,l,pm1,Om,lam,tau0,width)
            PB_btz[i] += 2*P_BTZn(n,RB[i],RB[i],rh,l,pm1,Om,lam,tau0B,widthB)
            PB_geon[i] += 2*PGEON_n(n,RB[i],RB[i],rh,l,pm1,Om,lam,tau0B,widthB)
            X_btz[i] += 2*XBTZ_n(n,RA[i],RB[i],rh,l,pm1,Om,lam,tau0,width)
            X_geon[i] += 2*XGEON_n(n,RA[i],RB[i],rh,l,pm1,Om,lam,tau0,width)
    print('')

Xb_re,Xb_im = [xi.real for xi in X_btz], [xi.imag for xi in X_btz]
Xg_re,Xg_im = [xi.real for xi in X_geon], [xi.imag for xi in X_geon]
X_btz = [fp.sqrt(xi.real**2 + xi.imag**2) for xi in X_btz]
deltax = [fp.sqrt(xi.real**2 + xi.imag**2) for xi in X_geon]
X_geon = [fp.sqrt(xi.real**2 + xi.imag**2) for xi in np.array(X_geon) + np.array(X_btz)]

P_btz, P_geon = np.array(P_btz), np.array(P_geon)
PB_btz, PB_geon = np.array(PB_btz), np.array(PB_geon)
#print('P_geon = ',P_geon)
P_geon = P_btz+P_geon       # add transition probability addition from geon to BTZ
PB_geon = PB_btz+PB_geon

PAPB_btz = np.sqrt(np.multiply(P_btz,PB_btz))
PAPB_geon = np.sqrt(np.multiply(P_geon,PB_geon))

fig1 = plt.figure(1,figsize=(5,2))
plt.plot(dR,P_btz,'b',label='PA BTZ')
plt.plot(dR,P_geon,'b:',label='PA Geon')
plt.legend()
plt.xlabel('dR')
plt.ylabel(r'$P/\lambda^2$')
#plt.ylim([0,1])

fig2 = plt.figure(2,figsize=(5,2))
plt.plot(dR,PB_btz,'b',label='PB BTZ')
plt.plot(dR,PB_geon,'b:',label='PB Geon')
plt.legend()
plt.xlabel('dR')
plt.ylabel(r'$P/\lambda^2$')

fig3 = plt.figure(3,figsize=(5,2))
plt.plot(dR,Xb_re,'b:',label='real')
plt.plot(dR,Xb_im,'orange',linestyle=':',label='imag')
plt.plot(dR,X_btz,'r',label='|X| BTZ')
plt.legend()
plt.xlabel('dR')
plt.ylabel(r'$X/\lambda^2$')
#plt.xlim([25,35])

fig4 = plt.figure(4, figsize=(5,2))
plt.plot(dR,Xg_re,'b:',label='real')
plt.plot(dR,Xg_im,'orange',linestyle=':',label='imag')
plt.plot(dR,deltax,'r',label=r'$\Delta |X|$')
plt.legend()
plt.xlabel('dR')
plt.ylabel(r'$X/\lambda^2$')

fig5 = plt.figure(5,figsize=(5,2))
plt.plot(dR,X_btz,'r',label='BTZ')
plt.plot(dR,X_geon,'r:',label='geon')
plt.legend()
plt.xlabel('dR')
plt.ylabel(r'$|X|/\lambda^2$')

conc_btz = np.maximum(0, np.array(X_btz) - np.array(PAPB_btz))
conc_geon = np.maximum(0, np.array(X_geon) - np.array(PAPB_geon))
fig6 = plt.figure(6,figsize=(7,4))
plt.plot(dR,conc_btz,'orange',label='BTZ')
plt.plot(dR,conc_geon,'orange',linestyle=':',label='geon')
plt.xlabel('dR')
plt.title('Concurrence')
plt.legend()

fig7 = plt.figure(7)
plt.plot(dR,P_btz,label='Top Hat P0 BTZ')
plt.plot(dR,P0_gauss,label='Gaussian P0 BTZ')
plt.legend()
plt.xlabel('dR')
plt.xlim([-0.5,3])
plt.ylim([0,5])

#%%
#y = np.linspace(-0.1,1.1*width,num=100)
#checkX = [integrandof_Xn_minus(yi,0,RA[20],RB[20],rh,l,pm1,Om,lam,tau0, width, 0, False) \
#          for yi in y]
#checkXre = [x.real for x in checkX]
#checkXim = [x.imag for x in checkX]
#
#
#plt.plot(y,checkXre,label='real')
#plt.plot(y,checkXim,label='imag')
#plt.legend()

#%% CHECK P0-

eps = 1e-8
x = np.linspace(-5,5,num=100)
yreal = []
yimag = []
print("i =", end='')
for i in range(len(x)):
    print(i,end=',',flush=True)
    val = integrandPBTZ_0m(x[i],10.05,10,10,1,1,1,1,1,eps)
    yreal.append(val)
    #yimag.append(val.imag)

plt.plot(x,yreal,label='real')
#plt.plot(x,yimag,label='imag')
plt.legend()
#plt.ylim([-1,1])

#print('\n',fp.quad(lambda y: integrandPBTZ_0m(y,10.05,10,10,1,1,1,1,1,eps), [-1,0])\
#      +  fp.quad(lambda y: integrandPBTZ_0m(y,10.05,10,10,1,1,1,1,1,eps), [0,1]))

#integrandPBTZ_0m(y,10.05,10,10,1,1,1,1,1,eps)

#%% Plot integrand(???) of X_btz

deltaphi = 0
def g(x,RA,RB):
#    bA = mp.sqrt(RA**2-rh**2)/l
#    bB = mp.sqrt(RB**2-rh**2)/l
#    return fp.j*4 * fp.exp(-fp.j*Om*(bB+bA)*(tau0+width/2))/(Om*(bB+bA))\
#            * fp.cos( Om/2*(bB+bA)*(x-width) ) * fp.cos(Om/2*(bB-bA)*x)\
#            * g_n2(x,0,RA,RB,rh,l,pm1,deltaphi)
    sigma = sigma_btz_n(x,n,RA,RB,rh,l,pm1,deltaphi) + 2
    if sigma == 0:
        return 0
    elif sigma > 0:
        return 1/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(sigma)
    else:
        return -fp.j/(4*fp.pi*fp.sqrt(2)*l) * 1/fp.sqrt(-sigma) * fp.sign(x)

def f(x,RA,RB):
    bA = mp.sqrt(RA**2-rh**2)/l
    bB = mp.sqrt(RB**2-rh**2)/l
    return fp.cos(Om/2*(bB+bA)*(x-width))*fp.cos(Om/2*(bB-bA)*x)

plt.figure(figsize=(10,6))
indicesofdR = np.linspace(130,160,num=5)
for ind in indicesofdR:
    ind = int(ind)
    xvals = np.linspace(0,1.1*width,num=200)
    yvals = [g(xi,RA[ind],RB[ind]) for xi in xvals]
    yplot = [yi.imag for yi in yvals]    
    plt.plot(xvals,yplot,label='dR is %.2f'%dR[ind])
plt.legend()
plt.title('Imaginary part of integrand')

#%% Integrands of P0m for Gauss and tophat
tau0 = 0
sig = 1
width = 2*np.sqrt(2*sig**2*np.log(2))            
M = 1               # mass
pm1 = 1             # zeta = +1, 0, or -1
Om = 1          # Omega

l = 10          # cosmological parameter
rh = np.sqrt(M)*l   # radius of horizon
lam = 1             # coupling constant
sep = 1

dR = 1. #np.linspace(25,35,num=200)
                    # proper distance of the closest detector

RA = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
RB = 1/2*np.exp(-sep/l) * ( (RA + np.sqrt(RA**2-rh**2))*np.exp(2*sep/l)\
                 + RA - np.sqrt(RA**2-rh**2))

################################
eps = 1e-2
ylowlim, yuplim = -0.5, 4
plotlowlim, plotuplim = -2, 2
################################

y = np.linspace(-0.5,4,num=101)

th_re, th_im, gau_re, gau_im = [], [], [], []

for yi in y:
    th = integrandPBTZ_0m(yi,RA,rh,l,pm1,Om,lam,0,width,eps)
    th_re.append(th.real)
    th_im.append(th.imag)
    gau = integrandPBTZ_0m_GAUSS(yi,RA,rh,l,pm1,Om,lam,sig,eps)
    gau_re.append(gau.real)
    gau_im.append(gau.imag)

gau_integral = fp.quad(lambda y: integrandPBTZ_0m_GAUSS(y,RA,rh,l,pm1,Om,lam,sig,eps),[0,10])
th_integral = fp.quad(lambda y: integrandPBTZ_0m(y,RA,rh,l,pm1,Om,lam,tau0,width,eps),[0,width])

###
fig2 = plt.figure(figsize=(5,4))
plt.plot(y,th_im,label='top hat imaginary')
plt.plot(y,gau_im,label='gaussian imaginary')
plt.legend()

###
fig = plt.figure(figsize=(9,5))
plt.plot(y,th_re,label='top hat')
plt.plot(y,gau_re,label='gaussian')
plt.plot([ylowlim,yuplim],[0,0],'c:')
plt.legend()

plt.ylim([plotlowlim,plotuplim])
plt.plot([width,width],[plotlowlim,plotuplim],'c:')

print('Integral, Gaussian:',gau_integral)
print('Integral, Top Hat: ',th_integral)
