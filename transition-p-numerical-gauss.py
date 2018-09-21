# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:15:29 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpmath import mp
from mpmath import fp

#%%############################################################################
#=============================== FUNCTIONS ===================================#
###############################################################################
# First integrand 
def f01(y,n,R,rh,l,pm1,Om,lam,sig):
    return lam**2*sig**2/2 * fp.exp(-sig**2*(y-Om)**2)\
     / fp.mpf((mp.exp(y * 2*mp.pi*l*mp.sqrt(R**2-rh**2)/rh) + 1))

# Second integrand
def f02(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = mp.mpf((R**2+rh**2)/(R**2-rh**2))
    if Zp - mp.cosh(y) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y)))
    elif Zp - mp.cosh(y) < 0:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp))
    else:
        return 0

# First integrand in the sum
def fn1(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zm = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) - 1))
    if Zm == mp.cosh(y):
        return 0
    elif Zm - fp.cosh(y) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zm - mp.cosh(y)))
    else:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zm))
    
# Second integrand in the sum
def fn2(y,n,R,rh,l,pm1,Om,lam,sig):
    K = lam**2*sig/2/fp.sqrt(2*fp.pi)
    a = (R**2-rh**2)*l**2/4/sig**2/rh**2
    b = fp.sqrt(R**2-rh**2)*Om*l/rh
    Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) + 1))
    if Zp == mp.cosh(y):
        return 0
    elif Zp - fp.cosh(y) > 0:
        return fp.mpf(K * mp.exp(-a*y**2) * mp.cos(b*y) / mp.sqrt(Zp - mp.cosh(y)))
    else:
        return fp.mpf(-K * mp.exp(-a*y**2) * mp.sin(b*y) / mp.sqrt(mp.cosh(y) - Zp))
    
def P_BTZn(n,R,rh,l,pm1,Om,lam,sig):
    b = fp.sqrt(R**2-rh**2)/l
    lim = 20*sig*rh/b/l**2
    #Zp = mp.mpf(rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) + 1))
    #print('Zm: ',Zm, 'Zp: ',Zp)
    #print(Zm,fp.acosh(Zm))
    if pm1==-1 or pm1==1 or pm1==0:
        if n==0:
            return fp.quad(lambda x: f01(x,n,R,rh,l,pm1,Om,lam,sig),[-fp.inf,fp.inf])
        else:
            Zm = rh**2/(R**2-rh**2)*(R**2/rh**2 * fp.cosh(2*fp.pi*rh/l * n) - 1)
            if fp.cosh(lim) < Zm or Zm < 1:
                return fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,lim])
            else:
                return fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[0,fp.mpf(mp.acosh(Zm))])\
                     - fp.quad(lambda x: fn1(x,n,R,rh,l,pm1,Om,lam,sig)\
                                      - fn2(x,n,R,rh,l,pm1,Om,lam,sig),[fp.mpf(mp.acosh(Zm)),lim])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#%%############################################################################
###############################################################################
###############################################################################

#time dependent

def wightman_btz_n(s,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0,eps=1e-4):
    if n == 0:
        Bm = (R**2 - rh**2)/rh**2
        Bp = (R**2 + rh**2)/rh**2
    else:
        Bm = R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1
        Bp = R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1
    Scosh = (R**2-rh**2)/rh**2 *mp.cosh(rh/l**2 * (s - fp.j*eps))
    
#    if s == 0:
#        return 0
#    elif Bm == Scosh or Bp == Scosh:
#        return 0
#    else:
    return 1/(4*fp.sqrt(2)*fp.pi*l) * (-fp.j/fp.sqrt(2*Bm)/\
              mp.sinh(rh/2/l**2 * (s - fp.j*eps)) - pm1/fp.sqrt(Bp - Scosh))


def integrand_btz_n(u,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    s_int_fnc = lambda s: lam**2 * fp.exp(-s**2/4/sig**2)\
                * (fp.exp(-fp.j*Om*s) * wightman_btz_n(s,n,R,rh,l,pm1,Om,lam,sig))
    
    if n != 0 or deltaphi !=0:    
        pt1 = fp.mpf( mp.acosh( rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1) ) )
    pt2 = fp.mpf( mp.acosh( rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1) ) )
    uplim = 2*tau - u
    
    if n== 0:
        if uplim < pt2:
            output = fp.quad(s_int_fnc,[0,uplim])
        else:
            output = fp.quad(s_int_fnc,[0,pt2]) + fp.quad(s_int_fnc,[pt2,uplim])

    else:
        if uplim < pt1:
            output = fp.quad(s_int_fnc,[0,uplim])
        elif uplim < pt2:
            output = fp.quad(s_int_fnc,[0,pt1]) + fp.quad(s_int_fnc,[pt1,uplim])
        else:
            output = fp.quad(s_int_fnc,[0,pt1]) + fp.quad(s_int_fnc,[pt1,pt2])\
                        + fp.quad(s_int_fnc,[pt2,uplim])
    
    output *= fp.exp(-u**2/4/sig**2)
    return output

def P_BTZ_n_tau(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return fp.quad(lambda u: integrand_btz_n(u,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0), [-10*sig,2*tau])

#%%############################################################################
###############################################################################
###############################################################################
def PGEON_gaussian(x,sig):
    return fp.mpf(mp.exp(-x**2/4/sig**2))

def sigma_geon(x,n,R,rh,l):
    return R**2/rh**2 * fp.mpf(mp.cosh(2*mp.pi * rh/l * (n+1/2))) - 1 + (R**2-rh**2)/rh**2\
    * fp.mpf(mp.cosh(rh/l**2 * x))
    
def h_n(x,n,R,rh,l,pm1):
    return 1/(4*fp.sqrt(2)*fp.pi*l) * (1/fp.sqrt(sigma_geon(x,n,R,rh,l)) \
              - pm1 * 1/fp.sqrt(sigma_geon(x,n,R,rh,l) + 2))

def T_P_int_geon(x,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return lam**2*fp.exp(-sig**2*Om**2) * fp.sqrt(fp.pi)*sig * fp.erf(-(x-2*tau/2/sig)-fp.j*sig*Om)\
            * fp.exp(-x**2/4/sig**2) * h_n(x,n,R,rh,l,pm1)
            
def T_PGEON_n(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
    return fp.quad(lambda x: T_P_int_geon(x,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0), [-10*sig,2*tau])

def PGEON_n(n,R,rh,l,pm1,Om,lam,sig):
    """
    Om = energy difference
    lam = coupling constant
    """
    if pm1==-1 or pm1==1 or pm1==0:
        return lam**2*sig*fp.sqrt(fp.pi) * fp.mpf(mp.exp(-sig**2 * Om**2)) *\
        fp.quad(lambda x: h_n(x,n,R,rh,l,pm1) * PGEON_gaussian(x,sig), [-fp.inf, fp.inf])
    else:
        print("Please enter a value of 1, -1, or 0 for pm1.")

#%% do stuff

sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
Om = [1,0.3,0.1,0.03,0.01,0.003,0.001]        # Omega
nmax = 3            # summation limit

#sep *= sig
l = 10*sig          # cosmological parameter
lam = 1             # coupling constant
M = 0.1
rh = np.sqrt(M)*l
dR = 1
R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)

lowlim, uplim = -3, 3
tau = np.linspace(lowlim,uplim,num=30)
#tau = np.exp(np.linspace(np.log(lowlim),np.log(uplim),num=20))
y = 0*tau
P = P_BTZn(0,R,rh,l,pm1,Om,lam,sig)
#for n in range(1,nmax+1):
#    P += 2*P_BTZn(n,R,rh,l,pm1,Om,lam,sig)

print("i=")
for i in range(len(tau)):
    print(i,end=', ',flush=True)
    for n in range(nmax+1):
        y[i] += fp.mpf(P_BTZ_n_tau(tau[i],n,R,rh,l,pm1,Om,lam,sig).real)
    print(y[-1])

#fig = plt.figure(figsize=(9,5))
plt.plot(tau,y)
plt.plot([lowlim,uplim],[P,P],linestyle=':',label=r'$P$')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$P(\tau)$')


#%%
x=np.linspace(-5,5,num=100)
#y=[T_P_int_geon(xi,-1,0,R,rh,l,pm1,Om,lam,sig) for xi in x]
y=[fp.erf(xi+fp.j).real for xi in x]
plt.plot(x,y)

#%%
# Checking that code is working ?

sig = 1             # width of Gaussian
pm1 = 1             # zeta = +1, 0, or -1
sep = 1             # distance between two detectors in terms of sigma
Om = 0.01        # Omega
#nmax = 3            # summation limit
n = 0

#sep *= sig
l = 10*sig          # cosmological parameter
lam = 1             # coupling constant
M = 0.1
rh = np.sqrt(M)*l
dR = 1
R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)

tau = 1

print('Making Plot...')
s = np.linspace(-3,3,num=80)
integrand_s_re = []
integrand_s_im = []
counter = 1
for si in s:
    print(counter,end=', ',flush=True)
    val = integrand_btz_n(si, tau, n, R, rh, l, pm1, Om, lam, sig)
    integrand_s_re.append( val.real )
    integrand_s_im.append( val.imag )
    counter += 1

plt.figure()
plt.plot(s,integrand_s_re,label='real')
plt.xlabel('integration variable')
plt.ylabel('integrand')
plt.title(r'$\tau$ = %s'%tau)
plt.legend()

plt.figure()
plt.plot(s,integrand_s_im,label='imag')
plt.xlabel('integration variable')
plt.ylabel('integrand')
plt.title(r'$\tau$ = %s'%tau)
plt.legend()

print('Computing integral...')
print(P_BTZ_n_tau(tau[0],1,R,rh,l,pm1,Om,lam,sig))

#%%

##time dependent 2
#
#def wm2_btz_01(y2,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    if n != 0:
#        return
#    else:
#        if y2 == 0:
#            return 0
#        else:
#            return 1/fp.mpf(mp.sinh(y2))
#    
#def wm2_btz_02(y2,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    #K = lam**2*l**3/( 4*fp.sqrt(2)*fp.pi*rh*fp.sqrt(R**2-rh**2) )
#    if n != 0:
#        return
##        Cm = rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.mpf(mp.cosh(rh/l * (deltaphi - 2*fp.pi*n))) - 1 )
##        Cp = rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.mpf(mp.cosh(rh/l * (deltaphi - 2*fp.pi*n))) + 1 )
#    else:
#        rho2 = rh**2/(R**2 - rh**2)
#        
#        if rho2 == fp.mpf(mp.sinh(y2))**2:
#            return 0
#        else:
#            return (-pm1/fp.sqrt(rho2 - fp.mpf(mp.sinh(y2))**2)).real
#
#def integrand2_btz_n(y,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    K = lam**2*l**3/( 4*fp.sqrt(2)*fp.pi*rh*fp.sqrt(R**2-rh**2) )
#    taut = rh/l**2 * tau
#    sigt = rh/l**2 * sig
#    Omt = rh/l**2 * Om
#    rho2 = rh**2/(R**2 - rh**2)
#    
#    if n != 0:
#        return
#    else:
#        #f = lambda x: K*2*fp.sqrt(2) * fp.exp(-y**2/sigt**2) * fp.exp(-x**2/sigt**2)\
#        #              * fp.cos(2*Omt*x) * wm2_btz_02(x,n,R,rh,l,pm1,Om,lam,sig)
#        f = lambda x:  K*2*fp.sqrt(2) * fp.exp(-y**2/sigt**2) * fp.exp(-x**2/sigt**2)\
#                       * fp.sin(2*Omt*x) * wm2_btz_01(x,n,R,rh,l,pm1,Om,lam,sig)
#        
##        x = np.linspace(0,0.1,num=50)
##        yvals = [f(xi) for xi in x]
##        plt.plot(x,yvals)
#        intlim = np.maximum(taut - y, 0)
#        #print(intlim)
#        return fp.quad(f, [0, intlim])
##        fcnlim = fp.mpf( mp.asinh( fp.sqrt(rho2)) )
##        
##        if intlim > fcnlim:
##            return fp.quad(f, [0, fcnlim])
##        else:
##            return fp.quad(f, [0, intlim])
#
#def P2_BTZ_n_tau(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    f = lambda y: integrand2_btz_n(y,tau,n,R,rh,l,pm1,Om,lam,sig)
#    taut = rh/l**2 * tau
#    sigt = rh/l**2 * sig
#    #fcnlim = fp.mpf( mp.asinh( fp.sqrt(rh**2/(R**2 - rh**2))) )
#    #print(taut,fcnlim)
#    return fp.quad(f, [-10*sigt,taut])
#        
##%%############################################################################
##=============================== PLOT THINGs =================================#
################################################################################
#
#sig = 1             # width of Gaussian
#M = 1               # mass
#pm1 = 1             # zeta = +1, 0, or -1
##sep = 1             # distance between two detectors in terms of sigma
#Om = 0.1          # Omega
#n = 0               # Order
#
##sep *= sig
#l = 10*sig          # cosmological parameter
#rh = np.sqrt(M)*l   # radius of horizon
#lam = 1             # coupling constant
#
#dR = 1
#R = 1/2*np.exp(-dR/l) * ( rh*np.exp(2*dR/l) + rh)
##Bp = (R**2 + rh**2)/rh**2
##print(l**2/rh*mp.acosh((R**2+rh**2)/(R**2-rh**2)))
#
##tau = []
#s_of_x = 50
#lolim = -5
#hilim = 5
#x = np.linspace(lolim,hilim,num=s_of_x)
#t2 = 0.5
#
#y = []
#y1 = []
#y2 = []
#y3 = []
#y4 = []
#y5 = []
## yim = []
#
#print('\n')
#for i in range(s_of_x):
#    print(i,end=", ",flush=True)
#    y.append(P2_BTZ_n_tau(x[i],0,R,rh,l,pm1,Om,lam,sig))
#    #y1.append(integrand2_btz_n(x[i],0,0,R,rh,l,pm1,Om,lam,sig)) 
#    #y2.append(integrand2_btz_n(x[i],1,0,R,rh,l,pm1,Om,lam,sig))
#    #y3.append(integrand2_btz_n(x[i],5,0,R,rh,l,pm1,Om,lam,sig))
#    #y5.append(integrand2_btz_n(x[i],50,0,R,rh,l,pm1,Om,lam,sig))
#    #y4.append(integrand2_btz_n(x[i],-1,0,R,rh,l,pm1,Om,lam,sig))
#    #yim.append(ytemp.imag)
#    #y1.append( Twightman_btz_n(x[i],t2,0,R,rh,l,pm1,Om,lam,sig)*fp.exp(-fp.j*Om*(x[i]-t2)))#*fp.exp(-fp.j*Om*x[i])))
#    #y2.append( fp.sin(-Om*()))
#    #y.append( wightman_btz_n(x[i],0,R,rh,l,pm1,Om,lam,sig)*fp.exp(-fp.j*Om*x[i])*fp.exp(-x[i]**2/4/sig**2)) #.imag
#    #y.append( TP_BTZ_n_tau(x[i],0,R,rh,l,pm1,Om,lam,sig))
#
#fig = plt.figure(figsize=(7,5))
#
##plt.plot(x,y4,'orange',label=r'$\tau=-1$')
##plt.plot(x,y1,'b',label=r'$\tau=0$')
##plt.plot(x,y2,'g',label=r'$\tau=1$')
##plt.plot(x,y3,'r',label=r'$\tau=5$',linewidth=5)
##plt.plot(x,y5,'k',label=r'$\tau=50$')
#
#plt.plot(x,y,'b',label='')
##plt.plot([-plotlim,plotlim],[0,0],'c:')
##plt.plot([lolim,hilim],[P_BTZn(0,R,rh,l,pm1,Om,lam,sig),P_BTZn(0,R,rh,l,pm1,Om,lam,sig)],'c:')
##plt.plot(x,y2,'orange',label='ReWs2*Cos')
##plt.plot(x,y3,'g',label='ImWs2*Sin')
##plt.ylim([-0.015,0.015])
#plt.legend()
##plt.plot(x,y2,'orange')
##plt.ylim([-0.06,0.02])
#plt.xlabel(r'$\tau$')
#plt.ylabel(r'$P(\tau)$')
##plt.ylabel('Integrand')
#
##%%############################################################################
################################################################################
################################################################################
## using t and t'
#def Twightman_btz_n(t1,t2,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    if n == 0:
#        Bm = (R**2 - rh**2)/rh**2
#        Bp = (R**2 + rh**2)/rh**2
#    else:
#        Bm = R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1
#        Bp = R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1
#    Scosh = (R**2-rh**2)/rh**2 * fp.mpf(mp.cosh(rh/l**2 * (t1-t2)))
#    
#    if Bm == Scosh or Bp == Scosh:
#        return 0
#    else:
#        return 1/(4*fp.sqrt(2)*fp.pi*l) * (1/fp.sqrt(Bm - Scosh) ) #- pm1/fp.sqrt(Bp - Scosh) 
#
#def Tintegrand_btz_n(t1,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    t2_int_fnc = lambda t2: lam**2 * fp.exp(-t2**2/2/sig**2)\
#                * (fp.exp(-fp.j*Om*(t1-t2)) * Twightman_btz_n(t1,t2,n,R,rh,l,pm1,Om,lam,sig))
#    
#    if n != 0 or deltaphi !=0:    
#        pt1 = fp.mpf( mp.acosh( rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) - 1) ) )
#    pt2 = fp.mpf( mp.acosh( rh**2/(R**2-rh**2) * (R**2/rh**2 * fp.cosh(rh/l * (deltaphi - 2*fp.pi*n)) + 1) ) )
#    
#    lowlim = -fp.inf
#    
#    if n==0:
#        output = fp.quad(t2_int_fnc,[lowlim,t1-pt2])
#        if tau < t1+pt2:
#            output += fp.quad(t2_int_fnc,[t1-pt2,tau])
#        else:
#            output += fp.quad(t2_int_fnc,[t1-pt2, t1+pt2])\
#                    + fp.quad(t2_int_fnc,[t1+pt2, tau])
#    else:
#        output = fp.quad(t2_int_fnc,[lowlim,t1-pt2])\
#               + fp.quad(t2_int_fnc,[t1-pt2,t1-pt1])
#        if tau < t1+pt1:
#            output += fp.quad(t2_int_fnc,[t1-pt1,tau])
#        elif tau < t1+pt2:
#            output += fp.quad(t2_int_fnc,[t1-pt1,t1+pt1])\
#                    + fp.quad(t2_int_fnc,[t1+pt1,tau])
#        else:
#            output += fp.quad(t2_int_fnc,[t1-pt1,t1+pt1])\
#                    + fp.quad(t2_int_fnc,[t1+pt1,t1+pt2])\
#                    + fp.quad(t2_int_fnc,[t1+pt2,tau])
#    
#    output *= fp.exp(-t1**2/2/sig**2)
#    return output
#
#def TP_BTZ_n_tau(tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0):
#    return fp.quad(lambda t1: Tintegrand_btz_n(t1,tau,n,R,rh,l,pm1,Om,lam,sig,deltaphi=0),[-fp.inf,tau])
    
