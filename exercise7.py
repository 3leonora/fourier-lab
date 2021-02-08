'''
Exercise 7

Analysis of ultra-short laserpulses

Dependencies: numpy, matplotlib

Authors:
@Eleonora Svanberg
@Henrik Jörnling
'''

#Modules
import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3*10**8 # speed of light [m/s]
lamb =  800*10**-9 # wavelength of laser [m]
T = lamb/c # period of laser [s]
omega = 2*np.pi/T

def E(t: np.ndarray, tau: float) -> np.ndarray:
    return np.cos(omega*(t-t0))*np.exp(-((t-t0)/tau)**2)

def plot_pulse(t: np.ndarray, Et: np.ndarray, tau: float) -> None:
    plt.plot(t, Et1, color='b')
    plt.title(r'$E(t)=\cos{[\omega(t-t0)]}e^{-[(t-t0)/\tau]^2} \tau=$'+f'{tau:.2f} s')
    plt.xlabel(r't')
    plt.ylabel(r'$E(t)$')
    plt.show()

# a)
x = 0
t0 = x/c # [s]

# Initiating variables
tau1 = 10*T
N = 1024 # number of samples
T0 = -1e-13 # start of sample [s]
T1 = 1e-13 # end of sample [s]
t = np.linspace(T0, T1, N, endpoint=False) # array with sample times
Et1 = E(t, tau1) # tau = approximate distribution in time [s]

plt.plot(t, Et1, color='b')
plt.title(r'$E(t)=\cos{[\omega(t-t0)]}e^{-[(t-t0)/\tau]^2}$')
plt.xlabel(r't')
plt.ylabel(r'$E(t)$')
plt.show()

# b)
clist = ['r', 'b', 'g', 'm', 'k', 'orange', 'c', 'y', 'lightgray', 'indigo']
for n in range(1, 11):
    tau = n*T
    Et = E(t, tau)
    Fft = np.fft.rfft(Et) # Fouriertransform F(f) of pulse
    plt.plot(np.real(Fft),'.', color=clist[n-1], label=r'$\tau=$'+f'{n}'+r'$T$') # Plot real part of Fft vs looping index 
    plt.legend()

plt.title('Real part of F(E(t))')
plt.show()

# d?
tau_d = 0.7*T
ta_d2 = 0.5*T
tau_d3 = 0.3*T
Et_d = E(t, tau_d)
Et_d2 = E(t, ta_d2)
Et_d3 = E(t, tau_d3)
Fft_d = np.fft.rfft(Et_d) # Fouriertransform F(f) of pulse
Fft_d2 = np.fft.rfft(Et_d2) 
Fft_d3 = np.fft.rfft(Et_d3) 
plt.plot(np.real(Fft_d),'.', color='m', label=r'$\tau=0.7T$') # Plot real part of Fft vs looping index 
plt.plot(np.real(Fft_d2),'.', color='b', label=r'$\tau=0.5T$') # Plot imaginary part of Fft vs looping index
plt.plot(np.real(Fft_d3),'.', color='c', label=r'$\tau=0.3T$') # Plot imaginary part of Fft vs looping index  
plt.legend()
plt.title(r'Real part of F(E(t)), $\tau < T$')
plt.show()

