from ase.optimize import BFGS
import ase
import numpy as np
import os
import datetime
from asedias.utils import read_traj
import asedias
import warnings
from typing import Callable, Union


class System:
    """
    asedias System class
    """
    def __init__(self, frag_indice:Union[list, np.ndarray], frag_charges:list[int], frag_spins:list[int]=None, system_charge:int=None, system_spin:int=None, images:list[ase.Atoms]=None, filepath:str=None, frag_names:list[str]=None):
        """
        
        Parameters
        ----------

        """
        self.filename = None

        # images or filepath must be specified properly
        if not images and not filepath: ValueError('images or filepath must be specified')
        if images and filepath: ValueError('images and filepath cannot be specified at same time')
        if not images:
            self.filename = os.path.basename(filepath)
            images = read_traj(filepath)
            assert images, "images are empty"

        # convert to list if it is np.ndarray
        frag_indice = list(
            indice.tolist() if isinstance(indice, np.ndarray) else indice
            for indice in frag_indice
        )

        self.images = images
        self.frag_charges = frag_charges
        self.frag_spins = frag_spins
        self.frag_indice = frag_indice # start with 0
        
        self.n_frags = len(frag_charges)
        self.n_images = len(images)

        if not system_charge:
            system_charge = sum(frag_charges)
        self.system_charge = system_charge
        self.system_spin = system_spin
        # if not system_spin:
        #     warnings.warn("Spin is not specified. Spin estimated by number of electrons in the molecule.", category=UserWarning)
        #     total_e = sum(atom.number for atom in self.images[0]) - self.system_charge
        #     self.system_spin = 0 if total_e % 2 == 0 else 1

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
            assert len(frag_names) == self.n_frags, "len(frag_names) must be same with n_frags"
            self.frag_names = frag_names 


        self.Engine = None
    
    def iterator(self):
        """
        Generator to yield system image and fragment information
        """
        for image in self.images:
            yield {
                'molecule': image,
                'system_spin': self.system_spin,
                'system_charge': self.system_charge,
                'frag_indice': self.frag_indice,
                'frag_charges': self.frag_charges,
                'frag_spins': self.frag_spins,
                'frag_names': self.frag_names
                }




    


class Engine:
    """
    
    """
    def __init__(self, calc_wrapper, preoptimizer_wrapper=None)
        self.calc_wrapper = calc_wrapper
        self.preoptimizer_wrapper = preoptimizer_wrapper


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

    calc_fmax = 0.05
    calc_maxiter = 100

    preoptimizer_fmax = 0.03
    preoptimizer_maxiter = 300
    
    clear_logging = True

    # If true only interaction analysis is performed
    interaction_only = False
  
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
    def __init__(self, system:asedias.System, use_spin:bool=False, job_id:str=None):
        self.system = system

        if use_spin:
            # estimate system spin
            if self.system.system_spin:
                warnings.warn("frag_spins is not specified. Spin estimated by number of electrons in the fragments.", category=UserWarning)
                frag_atoms = list(
                    self.system.images[0][index] for index in self.system.frag_indice
                )
                _frag_spins = list()
                for frag, charge in zip(frag_atoms, self.system.frag_charges):
                    frag_e = sum(frag.number for atom in self.images[0]) - charge
                    frag_spin = 0 if frag_e % 2 == 0 else 1
                    _frag_spins.append(frag_spin)
                # reset system frag_spins
                self.system.frag_spins = _frag_spins

            # estimate frag spin
            if self.system.frag_spins:
                warnings.warn("system_spin is not specified. Spin estimated by number of electrons in the molecule.", category=UserWarning)
                total_e = sum(atom.number for atom in self.images[0]) - self.system_charge
                # reset system system_spin
                self.system.system_spin = 0 if total_e % 2 == 0 else 1
        
        if not job_id:
            pass
        pass

    def run(self):
        
        if isinstance(self.system.Engine, Callable):
            self.system.Engine = Engine(calc_wrapper=self.system.Engine)
