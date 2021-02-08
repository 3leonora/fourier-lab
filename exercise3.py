'''
Exercise 3

Fourier Transform of the square of f(t)

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

T1 = 0.05
print(f'T1_1={T1}')
t = sample(T1)
ny = 20.0 # frequency [Hz]
omega = 2*np.pi*ny # angular frequency [rad]
ft = f(t, omega) # signal f(t)
ft2 = ft**2
Fft = np.fft.rfft(ft2) # Fouriertransform F(f) of real signal f(t)
print(f'n=0 -> 0 Hz, n=2 -> 40 Hz')
plt.plot(np.real(Fft),'o', color='g', label='real') # Plot real part of Fft vs looping index 
plt.plot(np.imag(Fft),'o', color='r', label='imag') # Plot imaginary part of Fft vs looping index 
plt.legend()
plt.show()
