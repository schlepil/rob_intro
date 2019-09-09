import matplotlib
if 1:
    latexOut = False
else:
    latexOut = True
if latexOut:
    matplotlib.use("pgf")
    pgf_with_custom_preamble = {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "text.latex.preamble": [r'\usepackage{amsmath}',r'\usepackage{amssymb}',r'\usepackage{amsfonts}'],
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size":12,
        "pgf.preamble": [
            "\\usepackage{units}",  # load additional packages
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            r"\setmathfont{xits-math.otf}",
            r"\setmainfont{DejaVu Serif}",  # serif font via preamble
        ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
else:
    matplotlib.use("Qt5Agg")

imageFolder = "../img/pgf"

beamerFigSize = (5*.8,3.5*.8)
beamerFigSizeSmall = (5*.55,3.5*.55)

import numpy as np
from numpy.linalg import multi_dot as np_multidot
import scipy as sp
from scipy import linalg as sp_linalg
from scipy import integrate as sp_int

import sympy as sy

from matplotlib import pyplot as plt

class variableStruct:
    def __init__(self, **kwargs):
        """
        A struct that can be easily accessed and extended
        :param kwargs:
        """
        self.__dict__.update(kwargs)
    
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()



class leftNeighboor():
    """
    Emulates scipy.interp1d("left") which does not exist
    """
    def __init__(self, t, x):
        self.t = np.array(t).squeeze()
        self.x = x.copy()
        if len(self.x.shape) == 1:
            # Here we need 2-D objects
            self.x.resize((1,self.t.size))
            self.out1d = True
        else:
            self.out1d = False


        assert (all(t[1::] - t[0:-1] > 0.))
        assert self.x.shape[1]
        self.dim = self.x.shape[0]

    def __call__(self, t):
        """
        get the left neighboor for each element in t
        :param t:
        :return:
        """
        t = np.array(t).squeeze()
        thisInd = np.maximum(np.searchsorted(self.t, t) - 1, 0)
        x = self.x[:, thisInd]

        # ensure that the returned object owns its data
        if not x.flags['OWNDATA']:
            x = x.copy()
        if self.out1d:
            x.resize((x.size,))

        return x
