# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:36:34 2019
                                NEWTONIAN AND GR ORBITS
@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

#%% PARAMETERS
E = -0.4
L = 1
r0 = 1
c = 1.

ecc = np.sqrt(1 + 2*E*L**2)

#%% FUNCTIONS

def Newton_drdphi_over_r2_squared(E, L, r):
    return 2*E/L**2 - 1/r**2 + 2/r/L**2

def Newton_drdphi(dsquared, r, pm1):
    return pm1 * r**2 * np.sqrt( dsquared )
    
def SC_drdphi_over_r2_squared(E, L, r):
    #Schwarzchild
    return 2*E/L**2 - 1/r**2 + 2/r/L**2 + 2/c**2/r**3

def SC_drdphi(dsquared, r, pm1):
    return pm1 * r**2 * np.sqrt( dsquared )

def Thorne_drdphi_over_r2_squared(E, L, r):
    #Thorne wormhole metric in Interstellar
    return 2*E/L**2 - 1/r**2 -4*E/L**2/c**2/r + 2/c**2/r**3

def Thorne_drdphi(dsquared, r, pm1):
    return pm1 * r**2 * np.sqrt( dsquared )

def WH_drdphi_over_r2_squared(A, g, L, r):
    #Wormhole wormhole metric in Interstellar
    return -1/L**2 + 2/L**2/r - 1/r**2 + A**2*(1-2/r)/L**2/(1+4/c**2 *np.sqrt(2*(r-2)) )

def WH_drdphi(dsquared, r, pm1):
    return pm1 * r**2 * np.sqrt( dsquared )

def near_circular_d2(a, gamma, rH, r):
    return (a**2 - 1/r**2 - gamma*np.sqrt(r)) * (1 - rH/r)

def near_circular_drdphi(dsquared, r, pm1):
    return pm1 * r**2 * np.sqrt(dsquared)

#%%############################################################################
#                                 TESTS                                       #
###############################################################################
#%% TEST NEWTON

# check dr/dphi
start = 0.5
r_test = np.linspace(start, 2, num=400)


drdphi_over_r2_squared_test = Newton_drdphi_over_r2_squared(E, L, r_test)
drdphi_test = Newton_drdphi(E, L, r_test)

phi_test = np.linspace(0, 2*np.pi, num=400)
r_orbit_test = L**2 / (1 + ecc*np.cos(phi_test))

print('plottng')
plt.figure(figsize=(9,7))
plt.plot(r_test, drdphi_over_r2_squared_test, label=r'$(dr/d\phi)^2/r^4$')
plt.plot(r_test,drdphi_test, label=r'$dr/d\phi$')
plt.plot(r_orbit_test,(phi_test-np.pi)*0.7/2/np.pi,':')

plt.ylim([-0.4,0.4])
plt.xlim([0.5, 2.0])

def real_dr_dphi(E, L, phi):
    return L**2 * ecc*np.sin(phi) / (1 + ecc*np.cos(phi))**2

def real_r_orbit(E, L, phi):
    return L**2 / (1 + ecc*np.cos(phi))

real_r_test = real_r_orbit(E, L, phi_test)
real_drdphi_test = real_dr_dphi(E, L, phi_test)
plt.plot(real_r_test,real_drdphi_test)

plt.legend()

#%% TEST SCHWARZCHILD

start = 0.5
r_test = np.linspace(start, 2, num=400)


drdphi_over_r2_squared_test = SC_drdphi_over_r2_squared(E, L, r_test)
drdphi_test = SC_drdphi(E, L, r_test)

phi_test = np.linspace(0, 2*np.pi, num=400)
#r_orbit_test = L**2 / (1 + ecc*np.cos(phi_test))

print('plottng')
plt.figure(figsize=(9,7))
plt.plot(r_test, drdphi_over_r2_squared_test, label=r'$(dr/d\phi)^2/r^4$')
plt.plot(r_test,drdphi_test, label=r'$dr/d\phi$')
#plt.plot(r_orbit_test,(phi_test-np.pi)*0.7/2/np.pi,':')

#plt.ylim([-0.4,0.4])
#plt.xlim([0.5, 2.0])

#real_r_test = real_r_orbit(E, L, phi_test)
#real_drdphi_test = real_dr_dphi(E, L, phi_test)
#plt.plot(real_r_test,real_drdphi_test)

plt.legend()

#%%############################################################################
#                                 ORBITS                                      #
###############################################################################
#%% NEWTONIAN ORBIT

# Create radius array
radius = np.array([r0])
x, y = np.array([r0]), np.array([0])
dphi = 0.0005
pm1 = -1
imax = 50000

# Find dr/dphi
d2 = Newton_drdphi_over_r2_squared(E, L, r0)
print('\ndsquared initial value is',d2)
drdphi = Newton_drdphi(d2, r0, pm1)

for i in range(1,imax):
    if i%2000 == 0:
        print('i =', i)
    #print(i,'-th run')
    # add r_i to array r
    radius = np.append(radius, radius[-1] + dphi * drdphi)
    #print(' ', i, '-th radius: ', radius[-1])
    
    d2 = Newton_drdphi_over_r2_squared(E, L, radius[-1])
    #print('  d2: ', d2)
    if d2 < 0:
        pm1 *= -1
        radius[-1] -= 2*dphi * drdphi
        #print('    new radius ', radius[-1])
        d2 = Newton_drdphi_over_r2_squared(E, L, radius[-1])
        #print('     new d2: ', d2)
    #print('  d2 used to compute drdphi:', d2)
    drdphi = Newton_drdphi(d2, radius[-1], pm1)
    #print('  drdphi:', drdphi)
    
    x = np.append(x, radius[-1] * np.cos(i*dphi) )
    y = np.append(y, radius[-1] * np.sin(i*dphi) )

plt.figure(figsize=(9,9))
plt.plot(x,y)

#%% Schwarzchild ORBIT

# Create radius array
radius = np.array([r0])
x, y = np.array([r0]), np.array([0])
dphi = 0.0002
pm1 = 1
imax = 50000
broke = 0

# Find dr/dphi
d2 = SC_drdphi_over_r2_squared(E, L, r0)
print('\ndsquared initial value is',d2)
drdphi = SC_drdphi(d2, r0, pm1)

for i in range(1,imax):
    if i%2000 == 0:
        print('i =', i)
    #print(i,'-th run')
    # add r_i to array r
    radius = np.append(radius, radius[-1] + dphi * drdphi)
    #print(' ', i, '-th radius: ', radius[-1])
    
    # IF object falls into the horizon 
    if radius[-1] < 2/c**2:
        broke = 1
        break
    
    d2 = SC_drdphi_over_r2_squared(E, L, radius[-1])
    #print('  d2: ', d2)
    if d2 < 0:
        pm1 *= -1
        radius[-1] -= 2*dphi * drdphi
        #print('    new radius ', radius[-1])
        d2 = SC_drdphi_over_r2_squared(E, L, radius[-1])
        #print('     new d2: ', d2)
    #print('  d2 used to compute drdphi:', d2)
    drdphi = SC_drdphi(d2, radius[-1], pm1)
    #print('  drdphi:', drdphi)
    
    x = np.append(x, radius[-1] * np.cos(i*dphi) )
    y = np.append(y, radius[-1] * np.sin(i*dphi) )

plt.figure(figsize=(7,7))
plt.plot(x,y)
plt.scatter(0,0, c='r')
plotlim = np.maximum( np.max(np.abs(x)) , np.max(np.abs(y)) ) * 1.05
plt.xlim([-plotlim,plotlim])
plt.ylim([-plotlim,plotlim])
if broke:
    plt.text(0.1,0.1,'CRASHED!!!')

#%% Thorne ORBIT

# Create radius array
radius = np.array([r0])
x, y = np.array([r0]), np.array([0])
dphi = 0.0002
pm1 = 1
imax = 50000
broke = 0

# Find dr/dphi
d2 = Thorne_drdphi_over_r2_squared(E, L, r0)
print('\ndsquared initial value is',d2)
drdphi = Thorne_drdphi(d2, r0, pm1)

for i in range(1,imax):
    if i%2000 == 0:
        print('i =', i)
    #print(i,'-th run')
    # add r_i to array r
    radius = np.append(radius, radius[-1] + dphi * drdphi)
    #print(' ', i, '-th radius: ', radius[-1])
    
    # IF object hits the singularity, 
    if radius[-1] < 0:
        broke = 1
        break
    
    d2 = Thorne_drdphi_over_r2_squared(E, L,  radius[-1])
    #print('  d2: ', d2)
    if d2 < 0:
        pm1 *= -1
        radius[-1] -= 2*dphi * drdphi
        #print('    new radius ', radius[-1])
        d2 = Thorne_drdphi_over_r2_squared(E, L, radius[-1])
        #print('     new d2: ', d2)
    #print('  d2 used to compute drdphi:', d2)
    drdphi = Thorne_drdphi(d2, radius[-1], pm1)
    #print('  drdphi:', drdphi)
    
    x = np.append(x, radius[-1] * np.cos(i*dphi) )
    y = np.append(y, radius[-1] * np.sin(i*dphi) )

plt.figure(figsize=(7,7))
plt.plot(x,y)
plt.scatter(0,0, c='r')
if broke:
    plt.text(0.1,0.1,'CRASHED!!!')
#plotlim = np.maximum( np.max(np.abs(x)) , np.max(np.abs(y)) ) * 1.05
#plt.xlim([-plotlim,plotlim])
#plt.ylim([-plotlim,plotlim])

#%% Wormhole ORBIT

# Create radius array
r0 = 10.
radius = np.array([r0])
x, y = np.array([r0]), np.array([0])
dphi = 0.0002
pm1 = 1
imax = 50000

A = 0.2
g = 1.
# Find dr/dphi
d2 = WH_drdphi_over_r2_squared(A, g, L, r0)
print('\ndsquared initial value is',d2)
drdphi = WH_drdphi(d2, r0, pm1)

for i in range(1,imax):
    if i%2000 == 0:
        print('i =', i)
    #print(i,'-th run')
    # add r_i to array r
    radius = np.append(radius, radius[-1] + dphi * drdphi)
    #print(' ', i, '-th radius: ', radius[-1])
    
    # IF object hits the singularity, 
    #if radius[-1] < 0:
    #    break
    
    d2 = WH_drdphi_over_r2_squared(A, g, L, radius[-1])
    #print('  d2: ', d2)
    if d2 < 0:
        pm1 *= -1
        radius[-1] -= 2*dphi * drdphi
        #print('    new radius ', radius[-1])
        d2 = WH_drdphi_over_r2_squared(A, g, L, radius[-1])
        #print('     new d2: ', d2)
    #print('  d2 used to compute drdphi:', d2)
    drdphi = WH_drdphi(d2, radius[-1], pm1)
    #print('  drdphi:', drdphi)
    
    x = np.append(x, radius[-1] * np.cos(i*dphi) )
    y = np.append(y, radius[-1] * np.sin(i*dphi) )

plt.figure(figsize=(7,7))
plt.plot(x,y)
plt.scatter(0,0, c='r')
#plotlim = np.maximum( np.max(np.abs(x)) , np.max(np.abs(y)) ) * 1.05
#plt.xlim([-plotlim,plotlim])
#plt.ylim([-plotlim,plotlim])

#%% Nearly circular wormhole orbits
a = 0.02
rH = 1.
k = 0.000025

gamma = k/np.sqrt(rH)
u0 = 0.01
r0 = 1/u0

radius = np.array([r0])
x, y = np.array([r0]), np.array([0])
dphi = 0.002
pm1 = 1
imax = 25150
broke = 0

phi_perih = []

# Find dr/dphi
d2 = near_circular_d2(a, gamma, rH, r0)
print('\ndsquared initial value is',d2)
drdphi = near_circular_drdphi(d2, r0, pm1)

for i in range(1,imax):
    if i%2000 == 0:
        print('i =', i)
    #print(i,'-th run')
    # add r_i to array r
    radius = np.append(radius, radius[-1] + dphi * drdphi)
    #print(' ', i, '-th radius: ', radius[-1])
    
    # IF object hits the singularity, 
    if radius[-1] < rH:
        broke = 1
        break
    
    d2 = near_circular_d2(a, gamma, rH, radius[-1])
    #print('  d2: ', d2)
    if d2 < 0:
        pm1 *= -1
        radius[-1] -= 2*dphi * drdphi
        #print('    new radius ', radius[-1])
        d2 = near_circular_d2(a, gamma, rH, radius[-1])
        #print('     new d2: ', d2)
    #print('  d2 used to compute drdphi:', d2)
    drdphi = near_circular_drdphi(d2, radius[-1], pm1)
    #print('  drdphi:', drdphi)
    
    x = np.append(x, radius[-1] * np.cos(i*dphi) )
    y = np.append(y, radius[-1] * np.sin(i*dphi) )
    
    if i>1 and radius[-2]<radius[-1] and radius[-2]<radius[-3]:
        phi_perih.append(i*dphi)

plt.figure(figsize=(7,7))
plt.plot(x,y)
plt.scatter(0,0, c='r')
if broke:
    plt.text(0.1,0.1,'CRASHED!!!')

period = (phi_perih[-1]-phi_perih[0])/(len(phi_perih)-1)
print('Period of oscaillation in radius is :', period/np.pi, 'pi')

