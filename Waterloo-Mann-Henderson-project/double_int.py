# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:10:51 2018
check double integral
@author: Admin
"""
from mpmath import fp
from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

def integrand1(x,a):
    f = lambda x2: fp.exp(-(x-x2)**2) * fp.exp(-fp.j*x) * fp.exp(-np.abs(x2))
    return fp.quad(f,[-fp.inf,a])

def integral1(a):
    return fp.quad(lambda x: integrand1(x,a), [-fp.inf,a])

def integrand2(u,a):
    f = lambda s: fp.exp(-s**2) * (fp.exp(fp.j*(s+u)) * fp.mpf(mp.exp(-np.abs(u)))\
                                  + fp.exp(fp.j*u) * fp.mpf(mp.exp(-np.abs(s+u))) )
    return fp.quad(f,[0,fp.inf])

def integral2(a):
    return fp.quad(lambda u: integrand2(u,a), [-fp.inf,a])

#%% Plot
a = 2
x = np.linspace(-10,5,num=50)
y1re = [integrand1(xi,a).real for xi in x]
y1im = [integrand1(xi,a).imag for xi in x]
y2re = [integrand2(xi,a).real for xi in x]
y2im = [integrand2(xi,a).imag for xi in x]

plt.subplot(211)
plt.plot(x,y1re,label='real')
plt.plot(x,y1im,label='imag')
plt.legend()

plt.subplot(212)
plt.plot(x,y2re,label='real')
plt.plot(x,y2im,label='imag')
plt.legend()
