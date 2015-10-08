__author__ = 'Luz'

import numpy as np
import matplotlib.pyplot as plt

# mu = 1.502

def f(y, v):
    return v, -y-1.502*((y**2)-1)*v

def K1(yn, vn, h, f):
    f_evaluado = f(yn, vn)
    return h * f_evaluado[0], h * f_evaluado[1]

def K2(yn, vn, h, f):
    k1 = K1(yn, vn, h, f)
    f_evaluado = f(yn + k1[0]/2, vn + k1[1]/2)
    return h * f_evaluado[0], h * f_evaluado[1]

def K3(yn, vn, h, f):
    k1 = K1(yn, vn, h, f)
    k2 = K2(yn, vn, h, f)
    f_evaluado = f(yn - k1[0] -2*k2[0], vn - k1[1] - 2*k2[1])
    return h*f_evaluado[0], h*f_evaluado[1]

def rungekutta3(yn, vn, h, f):
    k1 = K1(yn, vn, h, f)
    k2 = K2(yn, vn, h, f)
    k3 = K3(yn, vn, h, f)
    y_ni = yn + (1/6.0) * (k1[0] + 4*k2[0] + k3[0])
    v_ni = vn + (1/6.0) * (k1[1] + 4*k2[1] + k3[1])
    return y_ni, v_ni

### ------ P1 a) ----- ###

N = 5000

h = 20.0*np.pi / N
Y = np.zeros(N)
V = np.zeros(N)
s = np.linspace(0, 20. * np.pi, N)

Y[0] = 0.1
V[0] = 0

for i in range(1, N):
    Y[i] = rungekutta3(Y[i-1], V[i-1], h, f)[0]
    V[i] = rungekutta3(Y[i-1], V[i-1], h, f)[1]

plt.plot(Y, V, 'm')
plt.title('Trayectoria en el espacio (y=0.1, v=0)')
plt.xlabel('y')
plt.ylabel('dy/ds')
plt.show()

plt.plot(s, Y, 'g')
plt.title('Grafico de y vs s (y=0.1, v=0)')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.show()


### ------ P1 b) ----- ###

N = 5000

h = 20.0*np.pi / N
Y = np.zeros(N)
V = np.zeros(N)
s = np.linspace(0, 20. * np.pi, N)

Y[0] = 4.
V[0] = 0

for i in range(1, N):
    Y[i] = rungekutta3(Y[i-1], V[i-1], h, f)[0]
    V[i] = rungekutta3(Y[i-1], V[i-1], h, f)[1]

plt.plot(Y, V, 'm')
plt.title('Trayectoria en el espacio (y=4.0, v=0)')
plt.xlabel('y')
plt.ylabel('dy/ds')
plt.show()

plt.plot(s, Y, 'g')
plt.title('Grafico de y vs s (y=4.0, v=0)')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.show()

