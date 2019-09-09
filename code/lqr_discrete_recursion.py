from os import path

from trajectories import getTraj1

from definitions import *

# Compute control and cost-matrices
# Safe all (naive implementation)

# In this case, the final cost and the intermediate cost matrices are
# identitical.

# Initialize
lqr_DynProg = variableStruct()
lqr_DynProg.t = np.array([dt*i for i in range(Nsteps)]) # The different time-steps
lqr_DynProg.P =  [Q.copy()] # current cost == final cost
lqr_DynProg.K = [] # Directly store the linear gain matrix

P = lqr_DynProg.P[-1]
for _ in range(Nsteps):
    # Compute new K based on last go-to-cost
    K = -sp_linalg.inv(R+Bd.T*P*Bd)*(Bd.T*P*Ad)
    lqr_DynProg.K.append( K )
    # Compute new P based on current optimal feedback gain
    P = Q + Ad.T*P*Ad - Ad.T*P*Bd*sp_linalg.inv(R + Bd.T*P*Bd)*Bd.T*P*Ad
    lqr_DynProg.P.append( P )

# Reverse the order
# The solution is computed starting at the final time-point -> reverse
lqr_DynProg.K.reverse()
lqr_DynProg.P.reverse()

traj = getTraj1(lqr_DynProg.t, A, B, omega=0.) # If omega is non-zero the time delay affects convergence

# Define cost
def costFun(t,x,u=np.matrix(np.zeros((1,1)))):
    deltax = x-traj.xref_d(t)
    deltau = u-traj.uref_d(t)
    return np_multidot([deltax.T, Q, deltax]) + np_multidot([deltau.T, R, deltau])

# Simulate
x0 = traj.xref_c(0.) + dxBase
X = np.matrix(np.zeros((2,Nsteps)))
X[:,[0]] = x0
Udelta = np.matrix(np.zeros((1, Nsteps)))
Utot = Udelta.copy()
J = 0
for i in range(1,Nsteps):
    #compute control using the linear control law
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


# What if we limit the maximal input
Xsubopt = np.matrix(np.zeros((2,Nsteps)))
Xsubopt[:,[0]] = x0
Usuboptdelta = np.matrix(np.zeros((1, Nsteps)))
Usubopttot = Udelta.copy()
Jsubopt = 0
for i in range(1,Nsteps):
    #compute control
    Usuboptdelta[:, i - 1] = lqr_DynProg.K[i - 1] * (Xsubopt[:, [i - 1]] - traj.xref_d(lqr_DynProg.t[i - 1]))
    Usubopttot[:, i-1] = np.minimum(np.maximum(-uAbs, Usuboptdelta[:, i - 1] + traj.uref_d(lqr_DynProg.t[i-1])), uAbs) # Impose upper and lower bound
    #update cost
    Jsubopt += costFun((i-1)*dt, Xsubopt[:,[i-1]], Usubopttot[:, i-1])
    #update state
    Xsubopt[:,[i]] = Ad*Xsubopt[:,[i-1]] + Bd*(Usubopttot[:, i-1])
# Terminal cost
Jsubopt += costFun(Nsteps*dt, Xsubopt[:, [-1]])
Xsubopt = np.array(Xsubopt)

plt.tick_params( axis='x', which='both', bottom=False, top=False,  labelbottom=False )

ffl,aal = plt.subplots(3,1, figsize=beamerFigSize)
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
    ffl.savefig(fname = path.join(imageFolder, 'linear.pgf'), format='pgf')


fflc,aalc = plt.subplots(3,1, figsize=beamerFigSize)
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
aalc[-1].set_ylabel('u')
aalc[-1].set_xlabel('t')
aalc[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()
if latexOut:
    fflc.savefig(fname = path.join(imageFolder, 'linear_comparison.pgf'), format='pgf')
else:
    plt.show()

aalc[1].text(6., .1, f"J:{float(J):.3e}", color='black')
aalc[1].text(14., .1, f"J:{float(Jsubopt):.3e}", color='red')
if latexOut:
    fflc.savefig(fname = path.join(imageFolder, 'linear_comparison_text.pgf'), format='pgf')
else:
    plt.show()



# Get the solution based on convex optimization
import cvxpy

 # List of all decision variables
 # One list entry is the control vector for one time-step
allUcvx = [cvxpy.Variable((1,1), f"u{i}") for i in range(Nsteps-1)]

# Get all states as a (affine) function of U
allXcvx = [dxBase]
for i in range(1,Nsteps):
    allXcvx.append( Ad@allXcvx[-1] + Bd@allUcvx[i-1] )

# Construct the cost
costcvx = 0.
# control cost
for i in range(Nsteps-1):
    costcvx += cvxpy.quad_form(allUcvx[i], R)#U[:,[i]].T@R@U[:,[i]]
# state cost
for ax in allXcvx:
    costcvx += cvxpy.quad_form(ax, Q)#ax.T@Q@ax

# Construct contraints
# Constraints are just a list of equality and inequality constraints
allCstrcvx = []
for i in range(Nsteps-1):
    allCstrcvx.append( -uAbs<=allUcvx[i][0,0] )
    allCstrcvx.append( -uAbs<=-allUcvx[i][0,0] )

# Solve
probcvx = cvxpy.Problem(cvxpy.Minimize(costcvx), allCstrcvx)
probcvx.solve()

assert probcvx.status == 'optimal' #Check if problem was actually solved. If constraints are infeasible, this flag will be set to "primal|dual infeasible"
probcvxoldvalue = probcvx.value

Xcvx = np.hstack( [allXcvx[0]] + [aXcvx.value for aXcvx in allXcvx[1:]] ) # Evaluate the solution for the states
Ucvx = np.hstack( [aU.value for aU in allUcvx] + [allUcvx[-1].value] ) #Repeat last control input for completeness

# Plot compared to constrained lqr
ffcvx,aacvx = plt.subplots(3,1, figsize=beamerFigSize)
#aa[0].set_title(f"Cost opt: {float(J):.3e}; Cost subopt: {float(Jsubopt):.3e}")
for i in range(2):
    aacvx[i].step(traj.trajDiscrete_.xref.t, np.array(traj.trajDiscrete_.xref.x[i,:]).squeeze(), where='post', color='b')
    aacvx[i].step(lqr_DynProg.t, Xsubopt[i,:], '-r', where='post')
    aacvx[i].step(lqr_DynProg.t, Xcvx[i, :], '-g', where='post')
    aacvx[i].set_ylabel(xxLabel[i])
    aacvx[i].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
aacvx[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [uAbs, uAbs], '--g')
aacvx[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [-uAbs, -uAbs], '--g')
aacvx[-1].step(lqr_DynProg.t, np.array(Usubopttot[0,:]).squeeze(), '-r', where='post')
aacvx[-1].step(lqr_DynProg.t, np.array(Ucvx[0,:]).squeeze(), '-g', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Usuboptdelta[0,:]).squeeze(), '-r', where='post')
aacvx[-1].set_ylabel('u')
aacvx[-1].set_xlabel('t')
aacvx[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()

aacvx[1].text(14., .1, f"J:{float(Jsubopt):.3e}", color='red')
aacvx[1].text(6., .1, f"J:{float(probcvx.value):.3e}", color='green')
if latexOut:
    ffcvx.savefig(fname = path.join(imageFolder, 'cvx_comparison_text.pgf'), format='pgf')
else:
    plt.show()

# Add state contraints in addition to the input constraints
for ax in allXcvx[1:]:
    allCstrcvx.append(xMin <= ax[0, 0])
    allCstrcvx.append(-xMax <= -ax[0, 0])
    allCstrcvx.append( xdotMin <= ax[1,0] )
    allCstrcvx.append( -xdotMax <= -ax[1,0] )

for ax in allXcvx[stepFinish:]:
    allCstrcvx.append(xMinFinish <= ax[0, 0])
    allCstrcvx.append(-xMaxFinish <= -ax[0, 0])

# Solve
probcvx = cvxpy.Problem(cvxpy.Minimize(costcvx), allCstrcvx)
probcvx.solve()

assert probcvx.status == 'optimal'

Xcvxc = np.hstack( [allXcvx[0]] + [aXcvx.value for aXcvx in allXcvx[1:]] )
Ucvxc = np.hstack( [aU.value for aU in allUcvx] + [allUcvx[-1].value] ) #Repeat last control input for completeness


# Plot compared to constrained lqr
ffcvxc,aacvxc = plt.subplots(3,1, figsize=beamerFigSize)
#aa[0].set_title(f"Cost opt: {float(J):.3e}; Cost subopt: {float(Jsubopt):.3e}")
for i in range(2):
    aacvxc[i].step(traj.trajDiscrete_.xref.t, np.array(traj.trajDiscrete_.xref.x[i,:]).squeeze(), where='post', color='b')
    if i == 0:
        aacvxc[i].plot([tFinish, lqr_DynProg.t[-1]], [xMaxFinish, xMaxFinish], '--g')
        aacvxc[i].plot([tFinish, lqr_DynProg.t[-1]], [xMinFinish, xMinFinish], '--g')

    #    aacvxc[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xMax, xMax], '--g')
    #    aacvxc[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xMin, -xMin], '--g')
    #if i == 1:
    #    aacvxc[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xdotMax, xdotMax], '--g')
    #    aacvxc[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xdotMin, -xdotMin], '--g')
    aacvxc[i].step(lqr_DynProg.t, Xcvx[i, :], '-g', where='post')
    aacvxc[i].step(lqr_DynProg.t, Xcvxc[i, :], '-m', where='post')
    aacvxc[i].set_ylabel(xxLabel[i])
    aacvxc[i].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
aacvxc[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [uAbs, uAbs], '--g')
aacvxc[-1].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [-uAbs, -uAbs], '--g')
aacvxc[-1].step(lqr_DynProg.t, np.array(Ucvx[0,:]).squeeze(), '-g', where='post')
aacvxc[-1].step(lqr_DynProg.t, np.array(Ucvxc[0,:]).squeeze(), '-m', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Usuboptdelta[0,:]).squeeze(), '-r', where='post')
aacvxc[-1].set_ylabel('u')
aacvxc[-1].set_xlabel('t')
aacvxc[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()

aacvxc[1].text(14., .1, f"J:{float(probcvx.value):.3e}", color='magenta')
aacvxc[1].text(6., .1, f"J:{float(probcvxoldvalue):.3e}", color='green')
if latexOut:
    ffcvxc.savefig(fname = path.join(imageFolder, 'cvx_stateCstr_comparison_text.pgf'), format='pgf')
else:
    plt.show()


# Plot compared to constrained lqr
ffcvxcdet,aacvxcdet = plt.subplots(3,1, figsize=beamerFigSize)
#aa[0].set_title(f"Cost opt: {float(J):.3e}; Cost subopt: {float(Jsubopt):.3e}")
for i in range(2):
    aacvxcdet[i].step(traj.trajDiscrete_.xref.t[stepFinish:], np.array(traj.trajDiscrete_.xref.x[i,stepFinish:]).squeeze(), where='post', color='b')
    if i == 0:
        aacvxcdet[i].plot([tFinish, lqr_DynProg.t[-1]], [xMaxFinish, xMaxFinish], '--g')
        aacvxcdet[i].plot([tFinish, lqr_DynProg.t[-1]], [xMinFinish, xMinFinish], '--g')

    #    aacvxcdet[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xMax, xMax], '--g')
    #    aacvxcdet[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xMin, -xMin], '--g')
    #if i == 1:
    #    aacvxcdet[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xdotMax, xdotMax], '--g')
    #    aacvxcdet[i].plot([lqr_DynProg.t[0], lqr_DynProg.t[-1]], [xdotMin, -xdotMin], '--g')
    aacvxcdet[i].step(lqr_DynProg.t[stepFinish:], Xcvx[i, stepFinish:], '-g', where='post')
    aacvxcdet[i].step(lqr_DynProg.t[stepFinish:], Xcvxc[i, stepFinish:], '-m', where='post')
    aacvxcdet[i].set_ylabel(xxLabel[i])
    aacvxcdet[i].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
aacvxcdet[-1].plot([tFinish, lqr_DynProg.t[-1]], [uAbs, uAbs], '--g')
aacvxcdet[-1].plot([tFinish, lqr_DynProg.t[-1]], [-uAbs, -uAbs], '--g')
aacvxcdet[-1].step(lqr_DynProg.t[stepFinish:], np.array(Ucvx[0,stepFinish:]).squeeze(), '-g', where='post')
aacvxcdet[-1].step(lqr_DynProg.t[stepFinish:], np.array(Ucvxc[0,stepFinish:]).squeeze(), '-m', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Udelta[0,:]).squeeze(), '-k', where='post')
#aa[-1].step(lqr_DynProg.t, np.array(Usuboptdelta[0,:]).squeeze(), '-r', where='post')
aacvxcdet[-1].set_ylabel('u')
aacvxcdet[-1].set_xlabel('t')
aacvxcdet[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

plt.tight_layout()

#aacvxcdet[1].text(14., .1, f"J:{float(probcvx.value):.3e}", color='magenta')
#aacvxcdet[1].text(6., .1, f"J:{float(probcvxoldvalue):.3e}", color='green')
if latexOut:
    ffcvxcdet.savefig(fname = path.join(imageFolder, 'cvx_stateCstr_det_comparison_text.pgf'), format='pgf')
else:
    plt.show()