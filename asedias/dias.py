from typing import Callable
from ase.calculators.calculator import Calculator
import ase
import warnings
from asedias.base import ParameterManager
from asedias.utils import is_ipython
import asedias

if is_ipython:
    from IPython.display import clear_output



def potential_energy(calc_wrapper:Callable[..., Calculator], atoms:ase.Atoms, charge:int, spin:int, use_spin:bool)->float:
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
             spin:int,
             use_spin:bool,
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




def DIAS(image:dict, Engine:asedias.Engine, use_spin:bool)->dict:
    """
    Parameters
    ----------

    Returns
    -------
      - DIASresult(dict) : A json containing single point DIAS results.
    """
    # black result dict
    DIAS_result = dict()
    DIAS_result['molecule'] = dict()

    _molecule = image['molecule']

    # total energy of molecule
    total_energy = potential_energy(
       calc_wrapper=Engine.calc_wrapper, 
       atoms=_molecule, 
       charge=image['system_charge'], 
       spin=image['system_spin'],
       use_spin=use_spin
    )
    DIAS_result["molecule"]["total"] = total_energy

    # distortion energy
    total_distortion = 0
    frag_container = zip(image['frag_indice'], image['frag_charges'], image['frag_spins'], image['frag_names'])
    for frag_index, frag_charge, frag_spin, frag_name in frag_container:
        frag_atoms = _molecule[frag_index]
        # preopt energy
        pre_optimized_Energy = potential_energy(
           calc_wrapper=Engine.calc_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge, 
           spin=frag_spin,
           use_spin=use_spin
           )
        print(f"\nOptimizer : optimizing {frag_name}")
        optimize(
           calc_wrapper=Engine.calc_wrapper, 
           preoptimizer_wrapper=Engine.preoptimizer_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge, 
           spin=frag_spin,
           use_spin=use_spin
           )
        optimized_Energy = potential_energy(
           calc_wrapper=Engine.calc_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge,
           spin=frag_spin,
           use_spin=use_spin
           )
        fragDistortion = pre_optimized_Energy - optimized_Energy # eV unit
        DIAS_result[frag_name] = {"distortion" : fragDistortion}
        total_distortion += fragDistortion
    DIAS_result["molecule"]["distortion"] = total_distortion

    # interaction energy
    total_interaction = total_energy - total_distortion
    DIAS_result["molecule"]["interaction"] = total_interaction


    clear_logging = ParameterManager.clear_logging

    # clear logging
    if is_ipython() and clear_logging:
        clear_output(wait=True)

    return DIAS_result



# def trajDIAS(
#     calc_wrapper:Callable[..., Calculator], 
#     trajFile:str, 
#     fragments_params: list|dict, 
#     preoptimizer_wrapper:Callable[..., Calculator]=None,
#     resultSavePath:str="./result.json",
#     save_kws:dict={}, 
#     **calc_kwargs
#     ):
#   """
#   Description
#   -----------
#   This function performs DIAS calculations for each IRC point in a trajectory file.

#   Parameters
#   ----------


#     Returns
#     -------
#       - trajDIASresult(dict) : A dictionary containing DIAS results for each frame in the trajectory.
#   """
#   if calc_kwargs.get('use_spin'):
#     warnings.warn('The current spin estimation is based on the total number of electrons.', UserWarning)
    
#   trajDIASresult = dict()
#   trajNum = len(read_traj(trajFile))
#   for IRC_idx, xyzString in enumerate(read_traj(trajFile, returnString=True)):
#     progress_bar(trajNum, IRC_idx)
#     trajDIASresult[IRC_idx] = DIAS(
#       xyzString=xyzString, 
#       fragments_params=fragments_params, 
#       calc_wrapper=calc_wrapper, 
#       preoptimizer_wrapper=preoptimizer_wrapper,
#       clear_log=True, 
#       **calc_kwargs
#       )
#   json_dump(
#     trajDIASresult=trajDIASresult, 
#     trajFile=trajFile, 
#     resultSavePath=resultSavePath, 
#     **save_kws
#     )
#   print("ase_dias CALCULATION TERMINATED NORMALLY")
#   return trajDIASresult





## 해결해야 함
'''
    # If spin is used but not specified, estimate the spin value based on the number of electrons
    if use_spin and not spin:
        total_e = sum(atom.number for atom in atoms) - charge
        spin = 0 if total_e % 2 == 0 else 1


  warnings.warn("Spin is not specified. Spin estimated by number of electrons in fragment.", category=UserWarning)
'''