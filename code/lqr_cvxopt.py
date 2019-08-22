from definitions import *

import cvxopt

# For the moment this file is restricteed to x=xd=u=0

# # variables are all x_i and u_i
# # dec_var = |... x_i, u_i ..., x_N !No u_N|_i
#
# # First construct the finite dimensional equality constraint set
# # x_i+1 = A.x_i + B.u_i
#
# Acstr = np.zeros((Nsteps*2, Nsteps*(2+1)-1))
# bcstr = cvxopt.matrix(np.zeros(Nsteps*2,))
#
# # Set initial value
# bcstr[:2] = dxBase[:,0]
# Acstr[:2,:2] = np.identity(2) #x_0 = dx
# for i in range(1,Nsteps):
#     ix0r = i*2
#     ix0c = i*3
#     im1x0c = (i-1)*3
#     #x_i = A.x_i-1 + B.u_i-1 -> x_i - A.x_i-1 - B.u_i-1 = 0
#     Acstr[ix0r:ix0r+2, ix0c:ix0c+2] = np.identity(2) # x_i
#     Acstr[ix0r:ix0r+2, im1x0c:im1x0c+2] = -Ad # - A.x_i-1
#     Acstr[ix0r:ix0r+2, im1x0c+2:im1x0c+3] = -Bd # - B.u_i-1



# variables are all x_i and u_i
# dec_var = |u_0, u_i ..., u_N-1|
# x_i = A.x_i-1 + B.u_i-1 = A^{i}.x_0 + sum_{j<i} A^{i-1-j}.B.u_j

#Construct inequality contraints on input
# G_u = np.vstack((np.identity(1*(Nsteps-1)), -np.identity(1*(Nsteps-1))))
# h_u = -uAbs*np.ones((2*(Nsteps-1),)) # u_m <= u <= uM -> u_m <= u; -uM <= -u
#
# # Construct all needed powers of Ad
# AdPowers = [np.identity(2)]
# for i in range(1,Nsteps):
#     AdPowers.append(np.dot(AdPowers[-1], Ad))
#
#
# #Construct the cost
# P = np.zeros((1*(Nsteps-1), 1*(Nsteps-1)))
# # R
# for i in range(Nsteps-1):
#     P[i*1:i*1+1] = R

import cvxpy

allU = [cvxpy.Variable((1,1), f"u{i}") for i in range(Nsteps-1)]

# Get all states as a function of U
allX = [dxBase]
for i in range(1,Nsteps):
    allX.append( Ad@allX[-1] + Bd@allU[i-1] )

# Construct the cost
cost = 0.
# control cost
for i in range(Nsteps-1):
    cost += cvxpy.quad_form(allU[i], R)#U[:,[i]].T@R@U[:,[i]]
# state cost
for ax in allX:
    cost += cvxpy.quad_form(ax, Q)#ax.T@Q@ax

# Construct contraints
allCstr = []
for i in range(Nsteps-1):
    allCstr.append( -uAbs<=allU[i][0,0] )
    allCstr.append( -uAbs<=-allU[i][0,0] )

prob = cvxpy.Problem(cvxpy.Minimize(cost), allCstr)
prob.solve()

assert prob.status == 'optimal'

#



print(Acstr)