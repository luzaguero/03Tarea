__author__ = 'Luz'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode

def funcion(t,w):
    #sistema de lorenz a integrar
    x, y, z = w
    return [10.*(y-x), x*(28.-z)-y, x*y-(8/3.)*z]

#condiciones iniciales
ci=[-5,1,-10]
t0=0

s=ode(funcion)
s.set_integrator('dopri5')
s.set_initial_value(ci,t0)

t=np.linspace(t0,100,50000)
x=np.zeros(len(t))
y=np.zeros(len(t))
z=np.zeros(len(t))

for i in range(len(t)):
    s.integrate(t[i])
    x[i], y[i], z[i] = s.y

fig = plt.figure(1)
fig.clf()

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('auto')

ax.plot(x, y, z, 'c')
plt.title("Atractor de Lorenz")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
