from core import *

from trajectories import getTraj1


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

# Compute control and cost-matrices
# Safe all (naive implementation)
lqr_DynProg = variableStruct()
lqr_DynProg.t = np.array([dt*i for i in range(Nsteps)])
lqr_DynProg.P =  [Q.copy()]
lqr_DynProg.K = []

P = lqr_DynProg.P[-1]
for _ in range(Nsteps):
    # Compute new K
    K = -sp_linalg.inv(R+Bd.T*P*Bd)*(Bd.T*P*Ad)
    lqr_DynProg.K.append( K )
    # Compute new P
    P = Q + Ad.T*P*Ad - Ad.T*P*Bd*sp_linalg.inv(R + Bd.T*P*Bd)*Bd.T*P*Ad
    lqr_DynProg.P.append( P )
# Reverse the order
lqr_DynProg.K.reverse()
lqr_DynProg.P.reverse()
lqr_DynProg.getK = lambda t: lqr_DynProg.K[np.flatnonzero(lqr_DynProg.t<=t)[0]]

traj = getTraj1(lqr_DynProg.t, A, B, omega=0.) # If omega is non-zero the time delay affects convergence

# Define cost =
def costFun(t,x,u=np.matrix(np.zeros((1,1)))):
    deltax = x-traj.xref_d(t)
    deltau = u-traj.uref_d(t)
    return np_multidot([deltax.T, Q, deltax]) + np_multidot([deltau.T, R, deltau])

# Simulate
# Discrete
x0 = traj.xref_c(0.) + np.array([[-0.5],[-0.3]])*0.5
X = np.matrix(np.zeros((2,Nsteps)))
X[:,[0]] = x0
Udelta = np.matrix(np.zeros((1, Nsteps)))
Utot = Udelta.copy()
J = 0
for i in range(1,Nsteps):
    #compute control
    Udelta[:, i - 1] = lqr_DynProg.K[i - 1] * (X[:, [i - 1]] - traj.xref_d(lqr_DynProg.t[i - 1]))
    Utot[:, i-1] = Udelta[:, i - 1] + traj.uref_d(lqr_DynProg.t[i-1])
    #update cost
    J += costFun((i-1)*dt, X[:,[i-1]], Utot[:, i-1])
    #update state
    X[:,[i]] = Ad*X[:,[i-1]] + Bd*(Utot[:, i-1])
# Terminal cost
J += costFun(Nsteps*dt, X[:, [-1]])
X = np.array(X)

ff,aa = plt.subplots(1,1)
aa.plot(X[0,:], X[1,:], '.k')

ff,aa = plt.subplots(4,1)
aa[0].set_title(f"Cost: {float(J):.3e}")
for i in range(2):
    aa[i].step(traj.trajDiscrete_.xref.t, np.array(traj.trajDiscrete_.xref.x[i,:]).squeeze(), where='post', color='b')
    aa[i].step(lqr_DynProg.t, X[i,:], '-k', where='post')
aa[-2].step(lqr_DynProg.t, np.array(Utot[0,:]).squeeze(), '-k', where='post')
aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')


# What if we limit?
uAbs = 0.065

Xsubopt = np.matrix(np.zeros((2,Nsteps)))
Xsubopt[:,[0]] = x0
Usuboptdelta = np.matrix(np.zeros((1, Nsteps)))
Usubopttot = Udelta.copy()
Jsubopt = 0
for i in range(1,Nsteps):
    #compute control
    Usuboptdelta[:, i - 1] = lqr_DynProg.K[i - 1] * (Xsubopt[:, [i - 1]] - traj.xref_d(lqr_DynProg.t[i - 1]))
    Usubopttot[:, i-1] = np.minimum(np.maximum(-uAbs, Usuboptdelta[:, i - 1] + traj.uref_d(lqr_DynProg.t[i-1])), uAbs)
    #update cost
    Jsubopt += costFun((i-1)*dt, Xsubopt[:,[i-1]], Usubopttot[:, i-1])
    #update state
    Xsubopt[:,[i]] = Ad*Xsubopt[:,[i-1]] + Bd*(Usubopttot[:, i-1])
# Terminal cost
Jsubopt += costFun(Nsteps*dt, Xsubopt[:, [-1]])
Xsubopt = np.array(Xsubopt)

plt.tick_params( axis='x', which='both', bottom=False, top=False,  labelbottom=False )

ffl,aal = plt.subplots(3,1, figsize=beamerFigSize)
#aa[0].set_title(f"Cost opt: {float(J):.3e}; Cost subopt: {float(Jsubopt):.3e}")
xxLabel = [r"$x$", r"$\dot{x}$"]
for i in range(2):
    aal[i].step(traj.trajDiscrete_.xref.t, np.array(traj.trajDiscrete_.xref.x[i,:]).squeeze(), where='post', color='b')
    aal[i].step(lqr_DynProg.t, X[i,:], '-k', where='post')
    aal[i].set_ylabel(xxLabel[i])
    aal[i].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
aal[-1].step(lqr_DynProg.t, np.array(Utot[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Usuboptdelta[0,:]).squeeze(), '-r', where='post')
aal[-1].set_ylabel('u')
aal[-1].set_xlabel('t')
aal[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()
if latexOut:
    ffl.savefig(fname = '../img/pgf/linear.pgf', format='pgf')


fflc,aalc = plt.subplots(3,1, figsize=beamerFigSize)
#aa[0].set_title(f"Cost opt: {float(J):.3e}; Cost subopt: {float(Jsubopt):.3e}")
for i in range(2):
    aalc[i].step(traj.trajDiscrete_.xref.t, np.array(traj.trajDiscrete_.xref.x[i,:]).squeeze(), where='post', color='b')
    aalc[i].step(lqr_DynProg.t, X[i,:], '-k', where='post')
    aalc[i].step(lqr_DynProg.t, Xsubopt[i,:], '-r', where='post')
    aalc[i].set_ylabel(xxLabel[i])
    aalc[i].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
aalc[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [uAbs, uAbs], '--g')
aalc[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [-uAbs, -uAbs], '--g')
aalc[-1].step(lqr_DynProg.t, np.array(Utot[0,:]).squeeze(), '-k', where='post')
aalc[-1].step(lqr_DynProg.t, np.array(Usubopttot[0,:]).squeeze(), '-r', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Usuboptdelta[0,:]).squeeze(), '-r', where='post')
aalc[-1].set_ylabel('u')
aalc[-1].set_xlabel('t')
aalc[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()
if latexOut:
    fflc.savefig(fname = '../img/pgf/linear_comparison.pgf', format='pgf')
else:
    plt.show()

aalc[1].text(6., .1, f"J:{float(J):.3e}", color='black')
aalc[1].text(14., .1, f"J:{float(Jsubopt):.3e}", color='red')
if latexOut:
    fflc.savefig(fname = '../img/pgf/linear_comparison_text.pgf', format='pgf')
else:
    plt.show()

