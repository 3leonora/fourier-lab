'''
Exercise 2

Fouriertransform of function with a period of T

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

# a) Test sample
T11 = 0.05
T12 = 0.1
print(f'T1_1={T11}')
t1 = sample(T11)
print(f'T1_2={T12}')
t2 = sample(T12)

# b) Signal function
ny = 20.0 # frequency [Hz]
omega = 2*np.pi*ny # angular frequency [rad]
ft1 = f(t1, omega) # signal f(t)
ft2 = f(t2, omega)
T = 2*np.pi/omega
print(f'T={T} s')

# c) Plot graph
plt.plot(t1, ft1, label='ft1') # plot signal f(t) vs t
plt.plot(t2, ft2, label='ft2')
plt.legend()
plt.show()

# d) Fourier Transform
Fft1 = np.fft.rfft(ft1) # Fouriertransform F(f) of real signal f(t)
plt.plot(np.real(Fft1),'o', color='g', label='ft1') # Plot real part of Fft vs looping index 
plt.plot(np.imag(Fft1),'o', color='r') # Plot imaginary part of Fft vs looping index 
Fft2 = np.fft.rfft(ft2) # Fouriertransform F(f) of real signal f(t)
plt.plot(np.real(Fft2),'x', color='g', label='ft2') # Plot real part of Fft vs looping index 
plt.plot(np.imag(Fft2),'x', color='r') # Plot imaginary part of Fft vs looping index 
plt.legend()
plt.show()
print(f'Fft1_1 (T1={T11}) = {Fft1[1]:.1f}')
print(f'Fft2_1 (T1={T12}) = {Fft2[2]:.1f}')

# e) If T1=0.1, what happens then?
# Directly proportional to time b_n != 0 -> b_2n !=0

# f) Correction of sample time
N = 1024
T0 = 0.0
ny_max1 = N/(2.*(T11-T0))
ny_max2 = N/(2.*(T12-T0))
frek1 = np.linspace(0, ny_max1,int((N/2)+1), endpoint=True)
frek2 = np.linspace(0, ny_max2, int((N/2)+1), endpoint=True)
plt.plot(frek1,np.real(Fft1),'o', color='g', label='ft1') # Plot real part of Fft vs frek 
plt.plot(frek1,np.imag(Fft1),'o', color='r') # Plot imaginary part of Fft vs frek 
plt.plot(frek2,np.real(Fft2),'x', color='g', label='ft2')
plt.plot(frek2,np.imag(Fft2),'x', color='r') 
plt.legend()
plt.show()
