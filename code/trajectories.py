from core import *

def getTraj1(tSteps:np.ndarray, A, B, omega = 2*np.pi*0.1):
    tSteps = tSteps.copy().reshape((-1,))

    # Get a reference trajectory
    # xd = Bd.u -> u = pinv(Bd).xd
    tSym = sy.symbols("t")
    omegaSym = sy.symbols("omega")

    xrefSym = sy.sin(omegaSym*tSym)
    xdrefSym = sy.diff(xrefSym, tSym)
    xddrefSym = sy.diff(xdrefSym, tSym)
    XrefSym = sy.Matrix([[xrefSym],[xdrefSym]])
    XdrefSym = sy.Matrix([[xdrefSym], [xddrefSym]])

    # Lambdify
    array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
    XrefNP = sy.lambdify([tSym, omegaSym], XrefSym, modules=array2mat)
    XdrefNP = sy.lambdify([tSym, omegaSym], XdrefSym, modules=array2mat)

    #Create the corresponding structure
    traj = variableStruct()

    # time continuous trajectories
    traj.xref_c = lambda t : np.matrix(XrefNP(t, omega)).reshape((2,np.array(t).size)) #Weird output of lambdify necessitates reshape
    traj.xdref_c = lambda t : np.matrix(XdrefNP(t, omega)).reshape((2,np.array(t).size))
    def uref(t):
        t = np.array(t).reshape((-1,))
        u = np.matrix(np.ones((1,t.size)))
        xr = traj.xref_c(t)
        xdr = traj.xdref_c(t)
        for i in range(t.size):
            u[:,i] = np.matrix(sp_linalg.lstsq(B, xdr[:,i] - A * xr[:,i])[0])
        return u
    traj.uref_c = lambda t: uref(t)

    # Discretized with zero order hold
    trajDiscrete = variableStruct()
    trajDiscrete.xref = leftNeighboor(tSteps, traj.xref_c(tSteps))
    trajDiscrete.xdref = leftNeighboor(tSteps, traj.xdref_c(tSteps))
    trajDiscrete.uref = leftNeighboor(tSteps, traj.uref_c(tSteps))
    # Forward
    traj.trajDiscrete_ = trajDiscrete
    traj.xref_d = lambda t: trajDiscrete.xref(t)
    traj.xdref_d = lambda t: trajDiscrete.xdref(t)
    traj.uref_d = lambda t: trajDiscrete.uref(t)

    return traj