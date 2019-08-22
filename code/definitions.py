from core import *

Nsteps = 20
tFinal = 20.
dt = tFinal/Nsteps

# Define matrices
A = np.matrix([[0.,1.],[-0.5,-0.15]])
B = np.matrix([[0.],[0.2]])

Ad = sp_linalg.expm(A*dt)
Bd = np.matrix( np.dot(np.dot(sp_linalg.inv(A), (Ad-np.identity(2))), B))

# Control
Q = np.identity(2)*1.
R = np.identity(1)

#Simulation
dxBase = np.array([[-0.5],[-0.3]])*0.5

# Limit on u
uAbs = 0.065
xdotMax = 0.2
xdotMin = -0.2
xMax = 0.2
xMin = -0.5

stepFinish = int(Nsteps*2./3.)
tFinish = stepFinish*dt
xMaxFinish = 0.0175
xMinFinish = -0.0175