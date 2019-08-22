from core import *

# Get a convex quadratic polynomial
P = np.array([[1., 0.4],[0.4, 2.]])*0.2

C = np.zeros((3,3))
C[0,1] = .5
C[1,0] = .33
C[0,2] = P[0,0]
C[2,0] = P[1,1]
C[1,1] = 2.*P[0,1]

x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)

z = np.polynomial.polynomial.polygrid2d(x,y,C)
z -= np.min(z)

zmin = np.min(z)
zmax = np.max(z)

zLevel = np.linspace(zmin**0.5, zmax**0.5, 21)**2
zColor = plt.get_cmap('viridis')(np.linspace(0.,1.,len(zLevel)))

ffcvx,aacvx = plt.subplots(1,1,figsize=beamerFigSizeSmall)
#aacvx.contour(x,y,z,cmap=plt.get_cmap('viridis'), levels=zLevel)
aacvx.contour(x,y,z,colors=zColor, levels=zLevel)

cvxCstr = np.array([[-1.5, 0., 1.5, 2., -1.5],[2., 0., 1., 2., 2.]])
aacvx.plot(cvxCstr[0,:], cvxCstr[1,:], '-k', linewidth=2.)
aacvx.set_xlabel(r"$x_0$")
aacvx.set_ylabel(r"$x_1$")
aacvx.tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)

ffncvx,aancvx = plt.subplots(1,1,figsize=beamerFigSizeSmall)
#aancvx.contour(x,y,z,cmap=plt.get_cmap('viridis'), levels=21)
aacvx.contour(x,y,z,colors=zColor, levels=zLevel)

ncvxCstr = np.array([[-1.5, -1.5, 0.25, 0.0, 2., 2., -1.5],[2, -0.2, 0.2, -1.5, -1.5, 2.25, 2.]]) + np.array([[-0.1],[0.5]])
aancvx.plot(ncvxCstr[0,:], ncvxCstr[1,:], '-r', linewidth=2.)
aancvx.set_xlabel(r"$x_0$")
aancvx.set_ylabel(r"$x_1$")
aancvx.tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)


if latexOut:
    ffcvx.savefig(fname = '../img/pgf/opt_cvxCstr.pgf', format='pgf')
    ffncvx.savefig(fname = '../img/pgf/opt_ncvxCstr.pgf', format='pgf')
else:
    plt.show()