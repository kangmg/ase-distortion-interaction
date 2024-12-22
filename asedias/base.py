from ase.optimize import BFGS
import ase
import numpy as np

from typing import Union


class System:
    """
    asedias System class
    """
    def __init__(self, images:Union[list[ase.Atoms], ase.Trajectory], frag_indice:list, frag_charges:list[int], frag_spins:list[int]=None, frag_names:list[str]=None):
        self.images = images
        self.frag_charges = frag_charges
        self.frag_indice = frag_indice # start with 0
        
        self.n_frags = len(frag_charges)
        self.n_images = len(images)

        # validate parameters
        assert list(len(frag) for frag in frag_charges) == list(len(frag) for frag in frag_indice), 'shape mismatch'
        assert len(self.images[0]) == len(set(sum(frag_indice, []))), 'incomplete indice list'

        # fragment name
        if not self.frag_names:
            default_names = list(
                f"frag_{i+1}" for i in range(self.n_frags)
            )
            self.frag_names = default_names
        else: 
            assert len(frag_names) == self.n_frags, f"len(frag_names) must be same with n_frags"
            self.frag_names = frag_names 


        self.engine = None

    def iterable_system(self):
        for image in self.images:



    


class engine:
    """
    
    """
    def __init__(self, calc_wrapper, preoptimizer=None)
        self.calc_wrapper = calc_wrapper
        self.preoptimizer = preoptimizer


class ParameterManager:
    """Package parameter manager class
    
    
    Usage
    -----
    >>> from asedias import ParameterManager
    >>> 
    >>> # show all parameters
    >>> print(ParameterManager.show_parameters())
    >>> 
    >>> # change parameters
    >>> ParameterManager.param_name = 'new value'
    """
    unit = ''
    axis = 'irc'

    optimizer = BFGS # FIRE, LBFGS

    calc_fmax = 0.5
    calc_maxiter = 50

    preoptimizer_fmax = 0.3
    preoptimizer_maxiter = 200
    
  
    @classmethod
    def default_parameters(cls):
        """
        Show all default parameters
        """
        _params = {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
        try:
            from pprint import pformat
            return pformat(_params, indent=1)
        except ModuleNotFoundError:
            return _params


class aseDIAS:
    """
    
    """
    def __init__(self):
        pass