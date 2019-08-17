import numpy as np
import scipy as sp
from scipy import linalg as sp_linalg

from matplotlib import pyplot as plt

dt = 0.05

# Define matrices
A = np.matrix([[0.,1.],[-0.1,-0.5]])
B = np.matrix([[0.],[0.2]])

Ad = sp_linalg.expm(A*dt)
Bd = Ad*B

# Control
Q = np.identity(2)
R = np.identity(1)

Nsteps = 100

# Compute control and cost-matrices
# Safe all (naive implementation)
lqr_DynProg = object()
lqr_DynProg.t = np.array([dt*i for i in range(Nsteps)])
lqr_DynProg.P =  [Q.copy()]
lqr_DynProg.K = []

Pold = lqr_DynProg.P[-1]
for _ in range(Nsteps):
    # Compute new K
    Kold = -sp_linalg.inv(R+Bd*Pold*Bd)*(Bd.T*lqr_DynProg.P[-1]*Ad)
    lqr_DynProg.K.append( Kold )
    # Compute new P
    Pold = Q + Ad.T*Pold*Ad - Ad.T*Pold*Bd*sp_linalg.inv(R + Bd.T*Pold*Bd)*Bd.T*Pold*Ad
    lqr_DynProg.P.append( Pold )
# Reverse the order
lqr_DynProg.K = reversed(lqr_DynProg.K)
lqr_DynProg.P = reversed(lqr_DynProg.P)


# Get a reference trajectory
#xd = Bd.u -> u = pinv(Bd).xd
xref = lambda t: np.array([[np.sin( np.floor(t/dt)*dt)], [np.cos( np.floor(t/dt)*dt)]])
xdref = lambda t: np.array([[np.cos( np.floor(t/dt)*dt)], [-np.sin( np.floor(t/dt)*dt)]])
uref = lambda t: sp_linalg.lstsq(Bd, xdref(t))

# Simulate
x0 = xref(0.) + np.array([[0.5],[-0.2]])
X = np.zeros((2,Nsteps))
X[:,[0]] = x0
for i in range(1,Nsteps):
    X[:,[i]] = Ad*X[:,[i-1]] + Bd*(uref(lqr_DynProg.t[i-1]) + lqr_DynProg.K[i-1]*(X[:,[i-1]]-xref(lqr_DynProg.t[i-1])))
XREF = xref(lqr_DynProg.t)


ff,aa = plt.subplots(1,1)
aa.plot(XREF[0,:], XREF[1,:], '.b')
aa.plot(X[0,:], X[1,:], '.r')

plt.show()



