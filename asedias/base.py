import ase
import numpy as np
import os
import datetime
from asedias.utils import read_traj, animation
import asedias
import warnings
from typing import Callable, Union
from asedias.dias import trajDIAS


# TODO
# supports pbc, unit cell
class System:
    """
    asedias System class

    """
    def __init__(self, frag_indices:Union[list[list[int]], np.ndarray], frag_charges:list[int], 
                 frag_spins:list[int]=None, system_charge:int=None, system_spin:int=None, 
                 images:list[ase.Atoms]=None, filepath:str=None, frag_names:list[str]=None, 
                 constraints:Union[list[int], np.ndarray]=None
                 ):
        """
        
        Parameters
        ----------

        """
        self.filename = None
        self.Engine = None

        # images or filepath must be specified properly
        if not images and not filepath: ValueError('images or filepath must be specified')
        if images and filepath: ValueError('images and filepath cannot be specified at same time')
        if not images:
            self.filename = os.path.basename(filepath)
            images = read_traj(filepath)
            assert images, "images are empty"

        # convert to list if it is np.ndarray and sort by ascending
        frag_indices = list(
            sorted(indices.tolist()) if isinstance(indices, np.ndarray) else sorted(indices)
            for indices in frag_indices
        )

        self.images = images
        self.frag_charges = frag_charges
        self.frag_indices = frag_indices # start with 0

        self.n_frags = len(frag_charges)
        self.n_images = len(images)

        self.frag_spins = frag_spins if frag_spins else [[]] * self.n_frags
        self.system_charge = system_charge if system_charge else sum(frag_charges)
        self.system_spin = system_spin
        self.frag_names = frag_names if frag_names else list(f"frag_{i+1}" for i in range(self.n_frags))

        # validate parameters
        assert len(self.frag_names) == self.n_frags, "len(frag_names) must be same with n_frags"
        assert len(self.frag_spins) == self.n_frags, "len(frag_spins) must be same with n_frags"
        assert len(frag_charges) == len(frag_indices), 'shape mismatch'
        assert len(images[0]) == len(set(sum(frag_indices, []))), 'incomplete indices list'

        self.frag_constraints = [[]] * self.n_frags
        if constraints:
            # convert constraints to list if it is np.array and sort by ascending
            constraints = sorted(constraints.tolist()) if isinstance(constraints, np.ndarray) else sorted(constraints)
            # divide constraints
            # the index in the molecular system must be converted to the index in the fragment system
            # e.g. constraint index 3 in the fragment [2, 3, 5] should be converted to 1
            for constraint_idx in constraints:
                for frag_idx, frag_indice in enumerate(self.frag_indices):
                    if constraint_idx in frag_indice: 
                        self.frag_constraints[frag_idx].append(frag_indice.index(constraint_idx))
                        break
    
    def plot(self):
        pass

    def animation(self, colorby:str='fragment', covalent_radius_percent:float=108., **kwargs):
        """
        traj animation
        """
        animation(images=self.images, frag_indices=self.frag_indices, colorby=colorby,
                  covalent_radius_percent=covalent_radius_percent, **kwargs )


    def iterator(self):
        """
        Generator to yield system image and fragment information
        """
        for image in self.images:
            yield {
                'molecule': image,
                'system_spin': self.system_spin,
                'system_charge': self.system_charge,
                'frag_indices': self.frag_indices,
                'frag_charges': self.frag_charges,
                'frag_spins': self.frag_spins,
                'frag_names': self.frag_names,
                'frag_constraints': self.frag_constraints
                }




    


class Engine:
    """
    
    """
    def __init__(self, calc_wrapper, preoptimizer_wrapper=None):
        self.calc_wrapper = calc_wrapper
        self.preoptimizer_wrapper = preoptimizer_wrapper


# class ParameterManager:
#     """Package parameter manager class
    
    
#     Usage
#     -----
#     >>> from asedias import ParameterManager
#     >>> 
#     >>> # show all parameters
#     >>> print(ParameterManager.show_parameters())
#     >>> 
#     >>> # change parameters
#     >>> ParameterManager.param_name = 'new value'
#     """
#     unit = ''
#     axis = 'irc'

#     optimizer = BFGS # FIRE, LBFGS

#     calc_fmax = 0.05
#     calc_maxiter = 100

#     preoptimizer_fmax = 0.03
#     preoptimizer_maxiter = 300
    
#     clear_logging = True

#     # If true only interaction analysis is performed
#     interaction_only = False
  
#     @classmethod
#     def default_parameters(cls):
#         """
#         Show all default parameters
#         """
#         _params = {
#             key: value
#             for key, value in cls.__dict__.items()
#             if not key.startswith("__") and not callable(value)
#         }
#         try:
#             from pprint import pformat
#             return pformat(_params, indent=1)
#         except ModuleNotFoundError:
#             return _params


class aseDIAS:
    """
    
    """
    def __init__(self, system:System, use_spin:bool=False, job_id:str=None):
        self.system = system

        if use_spin:
            # estimate system spin
            if self.system.system_spin:
                warnings.warn("frag_spins is not specified. Spin estimated by number of electrons in the fragments.", category=UserWarning)
                frag_atoms = list(
                    self.system.images[0][index] for index in self.system.frag_indices
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
        
        self.tmp_contrainer = dict()

    def run(self):
        """
        
        """
        assert self.system.Engine, "Engine is not configured"
        if isinstance(self.system.Engine, Callable):
            self.system.Engine = Engine(calc_wrapper=self.system.Engine)
        
        _images = self.system.iterator()
        tmp = trajDIAS(
            images=_images,
            engine=self.system.Engine,
            trajDIASresult=self.tmp_contrainer,
            use_spin=False
        )