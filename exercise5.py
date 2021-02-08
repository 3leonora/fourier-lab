'''
Exercise 5

Fourier Transform of non-integer frequencies

Dependencies: numpy, matplotlib

Authors:
@Eleonora Svanberg
@Henrik Jörnling
'''

#Modules
import numpy as np
import matplotlib.pyplot as plt

def f(t: np.ndarray, omega) -> np.ndarray:
    return np.sin(omega*t)

def sample(T1:int) -> np.ndarray:
    '''
    Gives a data point sample 
    '''
    N = 1024 # number of data points
    T0 = 0.0 # start of sample [s]
    t = np.linspace(T0, T1, N, endpoint=False) # array with sample time
    Ts = T1-T0
    deltat = Ts/N
    vs = 1/deltat
    deltav = 1/Ts
    print(f't[0]={t[0]:.2f}, t[1]={t[1]:.2f}, t[2]={t[2]:.2f}')
    print(f'Ts={Ts}, N={N}, deltat={deltat:.2f}, vs={vs:.2f}, deltav={deltav:.2f}')

    return t

T1 = 1.0
t = sample(T1)
ny1 = 100.4 # angular frequency [rad]
omega1 = 2*np.pi*ny1
ny2 = 100
omega2 = 2*np.pi*ny2
ft1 = f(t, omega1) # signal f(t)
ft2 = f(t, omega2) # signal f(t)
Fft1 = np.fft.rfft(ft1) # Fouriertransform F(f) of real signal f(t)
Fft2 = np.fft.rfft(ft2) # Fouriertransform F(f) of real signal f(t)

plt.subplot(121)
plt.plot(abs(Fft1), color='r', label=f'{ny1} Hz') # Plot real part of Fft vs looping index 
plt.legend()
plt.subplot(122)
plt.plot(abs(Fft2), color='b', label=f'{ny2} Hz') # Plot real part of Fft vs looping index 
plt.legend()
plt.suptitle(f'{ny1} Hz vs {ny2} Hz')
plt.show()