from core import *

dt = 0.05
omega = 2*np.pi*0.1

# Define matrices
A = np.matrix([[0.,1.],[-0.1,-0.5]])
B = np.matrix([[0.],[0.2]])

Ad = sp_linalg.expm(A*dt)
Bd = np.matrix( np.dot(np.dot(sp_linalg.inv(A), (Ad-np.identity(2))), B))

# Control
Q = np.identity(2)
R = np.identity(1)

Nsteps = 100

# Compute control and cost-matrices
# Safe all (naive implementation)
lqr_DynProg = variableStruct()
lqr_DynProg.t = np.array([dt*i for i in range(Nsteps)])
lqr_DynProg.P =  [Q.copy()]
lqr_DynProg.K = []

Pold = lqr_DynProg.P[-1]
for _ in range(Nsteps):
    # Compute new K
    Kold = -sp_linalg.inv(R+Bd.T*Pold*Bd)*(Bd.T*lqr_DynProg.P[-1]*Ad)
    lqr_DynProg.K.append( Kold )
    # Compute new P
    Pold = Q + Ad.T*Pold*Ad - Ad.T*Pold*Bd*sp_linalg.inv(R + Bd.T*Pold*Bd)*Bd.T*Pold*Ad
    lqr_DynProg.P.append( Pold )
# Reverse the order
lqr_DynProg.K.reverse()
lqr_DynProg.P.reverse()
lqr_DynProg.getK = lambda t: lqr_DynProg.K[np.flatnonzero(lqr_DynProg.t<=t)[0]]


# Get a reference trajectory
#xd = Bd.u -> u = pinv(Bd).xd
xref = lambda t: np.matrix([[np.sin( omega*np.floor(t/dt)*dt)], [omega*np.cos( omega*np.floor(t/dt)*dt)]])
xdref = lambda t: np.matrix([[omega*np.cos( omega*np.floor(t/dt)*dt)], [-omega**2*np.sin( omega*np.floor(t/dt)*dt)]])
# xd = A.x + B.u
# u = inv(B).(xd-A.x)
uref = lambda t: np.matrix(sp_linalg.lstsq(B, xdref(t)-A*xref(t))[0])

# Simulate
# Discrete
x0 = xref(0.) + np.array([[0.5],[-0.2]])*0.5
X = np.matrix(np.zeros((2,Nsteps)))
X[:,[0]] = x0
for i in range(1,Nsteps):
    X[:,[i]] = Ad*X[:,[i-1]] + Bd*(uref(lqr_DynProg.t[i-1]) + lqr_DynProg.K[i-1]*(X[:,[i-1]]-xref(lqr_DynProg.t[i-1])))
XREF = np.hstack([np.array(xref(at)) for at in lqr_DynProg.t])
X = np.array(X)
# Continuous
f = lambda x,t: np.array((A*x.reshape((2,1)) + B*(uref(t) + lqr_DynProg.getK(t)*(x.reshape((2,1)) - xref(t))))).reshape((-1,))
tCont = np.linspace(lqr_DynProg.t[0], lqr_DynProg.t[-1], 10*Nsteps)
Xcont = sp_int.odeint(f, np.array(x0).reshape((-1,)), tCont).T


ff,aa = plt.subplots(1,1)
aa.plot(Xcont[0,:], Xcont[1,:], '--b')
aa.plot(XREF[0,:], XREF[1,:], '.b')
aa.plot(X[0,:], X[1,:], '.r')

ff,aa = plt.subplots(2,1)
for i in range(2):
    aa[i].plot(tCont, Xcont[i,:], '-b')
    aa[i].plot(lqr_DynProg.t, XREF[i,:], '-g')
    aa[i].plot(lqr_DynProg.t, X[i,:], '-r')


plt.show()



