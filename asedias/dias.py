from typing import Callable
from ase.calculators.calculator import Calculator
import ase
import warnings
from asedias.base import ParameterManager
from asedias.utils import is_ipython, progress_bar
import asedias
from collections.abc import Iterable


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
        if not preopt_status: warnings.warn("Preoptimization not converged. Use a higher `preopt_maxiter` or a reduced `preopt_fmax` threshold.", category=UserWarning)

    # optimization
    calc = calc_wrapper(**_calc_kwargs)
    atoms.calc = calc
    opt = optimizer(atoms=atoms)
    opt_success = opt.run(fmax=calc_fmax, steps=calc_maxiter)
    if not opt_success: warnings.warn("Optimization not converged. Use a higher `calc_maxiter` or a reduced `calc_fmax` threshold.", category=UserWarning)

    return opt_success


def IAS(image:dict, engine:asedias.Engine, use_spin:bool)->dict:
    """
    Single Point interaction analysis(IAS) calculation

    Parameters
    ----------

    Returns
    -------
      - DIASresult(dict) : A json containing single point DIAS results.
    """


def DIAS(image:dict, engine:asedias.Engine, use_spin:bool)->dict:
    """
    Single Point distortion interaction analysis(DIAS) calculation

    Parameters
    ----------

    Returns
    -------
      - DIASresult(dict) : A json containing single point DIAS results.
    """
    # black result dict
    DIASresult = dict()
    DIASresult['molecule'] = dict()

    _molecule = image['molecule']

    # total energy of molecule
    total_energy = potential_energy(
       calc_wrapper=engine.calc_wrapper, 
       atoms=_molecule, 
       charge=image['system_charge'], 
       spin=image['system_spin'],
       use_spin=use_spin
    )
    DIASresult["molecule"]["total"] = total_energy

    # distortion energy
    total_distortion = 0
    frag_container = zip(image['frag_indice'], image['frag_charges'], image['frag_spins'], image['frag_names'])
    for frag_index, frag_charge, frag_spin, frag_name in frag_container:
        frag_atoms = _molecule[frag_index]
        # preopt energy
        pre_optimized_Energy = potential_energy(
           calc_wrapper=engine.calc_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge, 
           spin=frag_spin,
           use_spin=use_spin
           )
        print(f"\nOptimizer : optimizing {frag_name}")
        opt_success = optimize(
           calc_wrapper=engine.calc_wrapper, 
           preoptimizer_wrapper=engine.preoptimizer_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge, 
           spin=frag_spin,
           use_spin=use_spin
           )
        DIASresult['opt_success'] = opt_success
        optimized_Energy = potential_energy(
           calc_wrapper=engine.calc_wrapper, 
           atoms=frag_atoms, 
           charge=frag_charge,
           spin=frag_spin,
           use_spin=use_spin
           )
        fragDistortion = pre_optimized_Energy - optimized_Energy # eV unit
        DIASresult[frag_name] = {"distortion" : fragDistortion}
        total_distortion += fragDistortion
    DIASresult["molecule"]["distortion"] = total_distortion

    # interaction energy
    total_interaction = total_energy - total_distortion
    DIASresult["molecule"]["interaction"] = total_interaction

    # clear logging
    if is_ipython() and ParameterManager.clear_logging:
        clear_output(wait=True)

    return DIASresult



def trajDIAS(
    images:Iterable[dict], 
    engine:asedias.Engine, 
    use_spin:bool,
    trajDIASresult:dict
    ):
    """
    This function performs DIAS calculations for each IRC point in a trajectory file.

    Parameters
    ----------


    Returns
    -------
      - trajDIASresult(dict) : A dictionary containing DIAS results for each frame in the trajectory.
    """
    NumImages = sum(1 for _ in images)
    success_list = list()
    # iterate DIAS calculation for each image
    for image_idx, image in enumerate(images):
        progress_bar(NumImages, image_idx)
        # pass if already normally calculated
        if trajDIASresult.get(image_idx) and isinstance(trajDIASresult, dict):
            if trajDIASresult[image_idx].get('opt_success'):
                continue
        DIASresult = DIAS(
            image=image,
            engine=engine
            use_spin=use_spin
            )
        success_list.append(DIASresult['opt_success'])
        trajDIASresult[image_idx] = DIASresult
    
    if not all(success_list): warnings.warn("asedias CALCULATION ABNORMALLY TERMINATED", category=UserWarning)
    else: print("asedias CALCULATION NORMALLY TERMINATED")

    return trajDIASresult


# def DIASparser(
#     resultDictionary:dict, 
#     fragType:str, 
#     energyType:str=None, 
#     relative_idx:None|str|int=None, 
#     unit:str="kcal/mol"
#     ):
#   """
#   Description
#   -----------
#   Parses DIAS results from a dictionary based on the specified fragment type, energy type, relative index, and unit.

#   Parameters
#   ----------
#     - resultDictionary (dict) : The dictionary containing DIAS results.
#     - fragType (str) : The type of fragment or molecule to parse (`molecule` or `fragment names`).
#     - energyType (str) : The type of energy to parse ("total", "interaction", or "distortion").
#     - relative_idx (None | str | int) : The index of the reference energy value for computing relative values.
#     - unit (str, optional) : The unit for the energy values ("kcal/mol", "kJ/mol", "Hartree", or "eV"). Default is "kcal/mol".

#   Returns
#   -------
#     - energies(tuple) : A tuple of parsed energy values.
#   """
#   unit = unit.upper()
#   eV2unitFactor = {
#       "KJ/MOL"    : 96.485332,
#       "HARTREE"    : 0.0367493,
#       "KCAL/MOL"  : 23.060548,
#       "EV"        : 1
#   # Ref : http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table-detail.html
#   }
#   match fragType:
#     # > molecule
#     case "molecule":
#       molecule_energies = tuple(map(
#         lambda energy_dictionary : energy_dictionary["molecule"], 
#         list(resultDictionary.values())
#         ))
#       # > molecule > total | distortion | interaction
#       match energyType:
#         case "total":
#           total = tuple(map(
#             lambda molecule_energy : molecule_energy["total"] * eV2unitFactor[unit], 
#             molecule_energies
#             ))
#           return relative_values(
#             total, 
#             relative_index=relative_idx
#             ) if relative_idx is not None else total
#         case "interaction":
#           molecule_interaction = tuple(map(
#             lambda molecule_energy : molecule_energy["interaction"] * eV2unitFactor[unit], 
#             molecule_energies
#             ))
#           return relative_values(
#             molecule_interaction, 
#             relative_index=relative_idx
#             ) if relative_idx is not None else molecule_interaction
#         case "distortion" :
#           molecule_distortion = tuple(map(
#             lambda molecule_energy : molecule_energy["distortion"] * eV2unitFactor[unit], 
#             molecule_energies
#             ))
#           return relative_values(
#             molecule_distortion, 
#             relative_index=relative_idx
#             ) if relative_idx is not None else molecule_distortion
#     # returns IRC index
#     case "irc":
#       return tuple(resultDictionary.keys())
#     # > fragment > distortion
#     case fragments:
#       frag_distortion = tuple(map(
#         lambda distortion_dictionary : distortion_dictionary[fragType]["distortion"] * eV2unitFactor[unit], 
#         list(resultDictionary.values())
#         ))
#       return relative_values(
#         energy_series=frag_distortion, 
#         relative_index=relative_idx
#         ) if relative_idx is not None else frag_distortion


# # new geometric_parameter
# def geometric_parameter(trajFile:str, geo_param:dict)->tuple:
#   """
#   Description
#   -----------
#   This function computes geometric parameters, such as distance, angle, or dihedral,
#   for each IRC point in a trajectory file based on given geo_param.

#   Parameters
#   ----------
#   - trajFile (str) : The trajectory file path or trajectory format string to read.
#   - geo_param (dict) : The geometric parameters to compute.
#     - It should have the following structure: {axis_type: [list | tuple | str]}.
#     - Supported axis_types for the values are "distance", "angle", and "dihedral".

#   Returns
#   -------
#     - geometry paramters(tuple) : A tuple of geometric parameter values for each IRC point in the trajectory.

#   Example
#   -------
#   >>> geo_param = { "distance" : "1 2" }
#   >>> geometric_parameter(trajFile, geo_param)
#   (3.0, 3.1, 3.2, ...)
#   """ 
#   axis_type, axis_values = tuple(geo_param.items())[0]  # indices in axis_values start with 1

#   try:  # tuple or list is processed to start indexing from 0
#     axis_values = tuple(map(lambda value: value - 1, axis_values))
#   except TypeError:  # string types, such as '1 2 3 4', are to be processed in each match-case block
#     pass

#   match axis_type:
#     case "distance":
#       # distance | string   ex) geo_param = { "distance" : "1 2" }
#       if isinstance(axis_values, str):
#         indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
#         num_indices = len(indices)
#         if num_indices == 2:
#           return tuple(map(lambda Atoms: Atoms.get_distance(*indices), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 2 but got {num_indices}, {axis_values}")
#       # distance | [ list | tuple ]   ex) geo_param = { "distance" : [1,2] }
#       elif isinstance(axis_values, (list, tuple)):
#         if len(axis_values) == 2:
#           return tuple(map(lambda Atoms: Atoms.get_distance(*axis_values), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 2 but got {len(axis_values)}, {axis_values}")
#       else:
#         raise ValueError("Invalid geo_param value")

#     case "angle":
#       # angle | string   ex) geo_param = { "angle" : "1 2 3" }
#       if isinstance(axis_values, str):
#         indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
#         num_indices = len(indices)
#         if num_indices == 3:
#           return tuple(map(lambda Atoms: Atoms.get_angle(*indices), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 3 but got {num_indices}, {axis_values}")
#       # angle | [ list | tuple ]   ex) geo_param = { "angle" : [1,2,3] }
#       elif isinstance(axis_values, (list, tuple)):
#         if len(axis_values) == 3:
#           return tuple(map(lambda Atoms: Atoms.get_angle(*axis_values), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 3 but got {len(axis_values)}, {axis_values}")
#       else:
#         raise ValueError("Invalid geo_param value")

#     case "dihedral":
#       # dihedral | string   ex) geo_param = { "dihedral" : "1 2 3 4" }
#       if isinstance(axis_values, str):
#         indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
#         num_indices = len(indices)
#         if num_indices == 4:
#           return tuple(map(lambda Atoms: Atoms.get_dihedral(*indices), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 4 but got {num_indices}, {axis_values}")
#       # dihedral | [ list | tuple ]   ex) geo_param = { "dihedral" : [1,2,3,4] }
#       elif isinstance(axis_values, (list, tuple)):
#         if len(axis_values) == 4:
#           return tuple(map(lambda Atoms: Atoms.get_dihedral(*axis_values), read_traj(trajFile)))
#         else:
#           raise ValueError(f"geo_param expected 4 but got {len(axis_values)}, {axis_values}")
#       else:
#         raise ValueError("Invalid geo_param value")

#     case _:
#       raise ValueError(f"Invalid param type : {axis_type} | Supported types : 'distance', 'angle', 'dihedral'")


# def xaxis_formatter(trajFile:str, geo_param:dict):
#   """
#   Description
#   -----------
#   Format the x-axis label based on the geometric parameter specified in geo_param.

#   Parameters
#   ----------
#     - trajFile (str): The trajectory file path or trajectory format string to read.
#     - geo_param (dict): The geometric parameter for formatting the x-axis label.
#       - It should contain only one geometric parameter.
#       - The key should represent the type of geometric parameter, such as "angle", "distance", or "dihedral".
#       - The value should be either a string representing indices (e.g., "1 2"), or a list/tuple of indices.

#   Returns
#   -------
#     - x_axis_type, xlabel (tuple): A tuple containing the type of geometric parameter and the formatted x-axis label.

#   Note
#   ----
#   Suppose the XYZ coordinates are on an angstrom scale.
#   """
#   # check sigle geo_param
#   if len(geo_param) != 1:
#     raise ValueError(f"geo_param expected 1 geometric parameter, got {len(geo_param)} parameters")

#   paramType, param = tuple(geo_param.items())[0] # indices in param starts with 1

#   if isinstance(param, str):
#     param = param.strip()
#     param_tuple =  tuple(string.strip() for string in re.split("\s+", param)) # indice
#   elif isinstance(param, (list|tuple)):
#     param_tuple = tuple(str(integer) for integer in param) # indice
#   else:
#     raise ValueError("geo_param value error")
  
#   symbols = tuple(get_symbol(trajFile, idx) for idx in param_tuple) # symbols
#   symbol_index = "-".join(list(map(lambda arg : f"{arg[0]}({arg[1]})" ,zip(symbols, param_tuple))))

#   ParamType = paramType[0].upper() + paramType[1:] # e.g. distance --> Distance
      
#   axis_unit = {"angle": "Degree", "distance": "Angstrom", "dihedral": "Degree"} # suppose the XYZ coordinates are on an angstrom scale
#   x_axis_format = f'{ParamType} {symbol_index} / {axis_unit[paramType]}'
#   return paramType, x_axis_format
