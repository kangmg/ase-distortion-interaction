from typing import Callable
from ase.calculators.calculator import Calculator
import ase
import warnings
from asedias.base import ParameterManager
from asedias.utils import is_ipython
import asedias

if is_ipython:
    from IPython.display import clear_output



def potential_energy(calc_wrapper:Callable[..., Calculator], atoms:ase.Atoms, charge:int, use_spin:bool, spin:int)->float:
    """
    Calculate the potential energy.

    Parameters
    ----------

    Returns
    -------
        - potential energy(float) : The potential energy of the molecule or fragment.
    """
    # prepare calculator
    _calc_kwargs = {'charge': charge}
    if use_spin: _calc_kwargs['spin'] = spin
    calc = calc_wrapper(**_calc_kwargs)
    
    # attach calculator and get potential energy
    atoms.calc = calc
    return atoms.get_potential_energy()



def optimize(calc_wrapper:Callable[..., Calculator], 
             preoptimizer_wrapper:Callable[..., Calculator], 
             atoms:ase.Atoms, 
             charge:int,
             use_spin:bool,
             spin:int,
             ParameterManager:asedias.ParameterManager=ParameterManager
             ):
    """
    Optimize the geometry.

    Parameters
    ----------
    """
    # parameters
    optimizer = ParameterManager.optimizer
    calc_fmax = ParameterManager.calc_fmax
    calc_maxiter = ParameterManager.calc_maxiter
    preopt_fmax = ParameterManager.preoptimizer_fmax
    preopt_maxiter = ParameterManager.preoptimizer_maxiter
    clear_logging = ParameterManager.clear_logging

    _calc_kwargs = {'charge': charge}
    if use_spin: _calc_kwargs['spin'] = spin

    # pre-optimizaion
    if preoptimizer_wrapper:
        preopt_calc = preoptimizer_wrapper(**_calc_kwargs)
        atoms.calc = preopt_calc
        preopt = optimizer(atoms=atoms)
        preopt_status = preopt.run(fmax=preopt_fmax, steps=preopt_maxiter)
        if not preopt_status: warnings.warn("Preoptimization not converged. Use a higher `calc_maxiter` or a reduced `calc_fmax` threshold.", category=UserWarning)

    # optimization
    calc = calc_wrapper(**_calc_kwargs)
    atoms.calc = calc
    opt = optimizer(atoms=atoms)
    opt_status = opt.run(fmax=calc_fmax, steps=calc_maxiter)
    if not opt_status: warnings.warn("Optimization not converged. Use a higher `opt_maxiter` or a reduced `preopt_fmax` threshold.", category=UserWarning)

    # clear optimization logging
    if is_ipython and clear_logging:
        clear_output(wait=True)





## 해결해야 함
'''
    # If spin is used but not specified, estimate the spin value based on the number of electrons
    if use_spin and not spin:
        total_e = sum(atom.number for atom in atoms) - charge
        spin = 0 if total_e % 2 == 0 else 1


  warnings.warn("Spin is not specified. Spin estimated by number of electrons in fragment.", category=UserWarning)
'''