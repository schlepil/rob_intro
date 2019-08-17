import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import scipy as sp
from scipy import linalg as sp_linalg
from scipy import integrate as sp_int

from matplotlib import pyplot as plt

class variableStruct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return self.__str__()