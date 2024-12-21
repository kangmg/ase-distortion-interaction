# import ase.calculators
# import ase.calculators.calculator

from typing import Callable
from ase.calculators.calculator import Calculator
import torch
from ase import Atoms
from asedias.aimnet2ase import AIMNet2Calculator
import ase
from ase.optimize import BFGS
import os
from IPython.display import clear_output
import re
import json
import ase.io
from io import StringIO
from os.path import isfile, dirname, basename
import matplotlib.pyplot as plt
from asedias.utils import is_ipython, progress_bar, husl_palette, markers_, json_dump, draw_xyz
from importlib.util import find_spec
import subprocess
import warnings

models = lambda : print("""
- Supported models

      Abbr.     |      Full Name           |  Recommended
    ________________________________________________________
      b0        | aimnet2_b973c_0.jpt      |
      b1        | aimnet2_b973c_1.jpt      |
      b2        | aimnet2_b973c_2.jpt      |
      b3        | aimnet2_b973c_3.jpt      | 
      b973c     | aimnet2_b973c_ens.jpt    |   * 
      w0        | aimnet2_wb97m-d3_0.jpt   |
      w1        | aimnet2_wb97m-d3_1.jpt   |
      w2        | aimnet2_wb97m-d3_2.jpt   |
      w3        | aimnet2_wb97m-d3_3.jpt   |
      wb97m-d3  | aimnet2_wb97m-d3_ens.jpt |   * (default)
    ________________________________________________________

[Note] 
- Both `Abbr.' and `Full Name` are supported.
- The ensemble models( `b973c`,  `wb97m-d3`) are included in the ase_dias package. However, other models will be automatically downloaded when they are called.
""")


abbr = {"b0"        :  "aimnet2_b973c_0.jpt",
        "b1"        :  "aimnet2_b973c_1.jpt",
        "b2"        :  "aimnet2_b973c_2.jpt",
        "b3"        :  "aimnet2_b973c_3.jpt",
        "b973c"     :  "aimnet2_b973c_ens.jpt",
        "w0"        :  "aimnet2_wb97m-d3_0.jpt",
        "w1"        :  "aimnet2_wb97m-d3_1.jpt",
        "w2"        :  "aimnet2_wb97m-d3_2.jpt",
        "w3"        :  "aimnet2_wb97m-d3_3.jpt",
        "wb97m-d3"  :  "aimnet2_wb97m-d3_ens.jpt"}

def get_symbol(trajFile:str, atomic_idx:int|str)->str:
  """
  Description
  -----------
  get atomic symbol from atomic idx

  Parameters
  ----------
    - trajFile (str) : The trajectory file path or trajectory format string to read.  
    - atomic_idx(int) : atomic index starts with 1

  Returns
  -------
    - atomic symbol(str)
  """
  python_idx = int(atomic_idx) - 1
  atoms = read_traj(trajFile)[0]
  symbol = atoms.get_chemical_symbols()[python_idx]
  return symbol


def model_checker(modelPath:str)->None:
  """
  Description
  -----------
  Downloads the model if it doesn't exist at the specified path.

  Parameters
  ----------
    - modelPath (str): The path to check for the model.
  """
  # import wget
  if not isfile(modelPath):
    print("\033[31mModel not found. Downloading the model.\033[0m\n")
    try:
      import wget
    except (ModuleNotFoundError, ImportError):
      print("\033[31mInstalling the dependency package, wget.\033[0m\n")
      subprocess.run(["pip", "install", "wget"])
      import wget

    # model download
    name = basename(modelPath)
    github_url = f"https://github.com/kangmg/ase-distortion-interaction/raw/main/models/{name}" # TO BE CHANGE
    wget.download(github_url, modelPath)
    print("\033[34m\nModel successfully downloaded.\033[0m\n")
  else:
    pass


def load_aimnet2(model:str="wb97m-d3"): # TO BE CHANGE (docs string)
  """
  Description
  -----------

  Parameters
  ----------
    - model(str) : Name aimnet2 model. Default is wb97m-d3. \n
  Check available models via `ase_dias.models()`

  Returns
  -------
    - torch model (torch.jit.ScriptModule)
  """
  # model Directory
  packageDir = find_spec("ase_dias").submodule_search_locations[0]
  modelsDir = os.path.join(packageDir, "models", "aimnet2")
  # load model 
  if abbr.get(model):
    modelPath = os.path.join(modelsDir, abbr[model])
    model_checker(modelPath)
    return torch.jit.load(modelPath)
  elif model in abbr.values():
    modelPath = os.path.join(modelsDir, model)
    model_checker(modelPath)
    return torch.jit.load(modelPath)
  else:
    raise ValueError(f"'{model}' <-- is not a supported model. Check available model names via `ase_dias.models()`")


def potential_energy(calc_wrapper:Callable[..., Calculator], fragment:ase.Atoms, charge:int, **calc_kwargs)->float:
  """
  Description
  -----------
  Calculate the potential energy of a molecular fragment.

  Parameters
  ----------


  Returns
  -------
    - potential energy(float) : The potential energy of the molecular fragment.
  """
  use_spin = calc_kwargs.get('use_spin', False)

  if use_spin:
    total_e = sum(atom.number for atom in fragment) - charge
    spin = 0 if total_e % 2 == 0 else 1

  _calc_kwargs = {
    **calc_kwargs,
    'charge':charge
    }
  if use_spin: _calc_kwargs['spin'] = spin
  _calc_kwargs.pop('use_spin', None)
  
  calc = calc_wrapper(**_calc_kwargs)
  fragment.calc = calc
  return fragment.get_potential_energy()


def optimize(calc_wrapper:Callable[..., Calculator], fragment:ase.Atoms, charge:int, preoptimizer_wrapper:Callable[..., Calculator]=None, fmax:float=0.05, steps=500, clear_log=False, **calc_kwargs)->str:
  """
  Description
  -----------
  Optimize the geometry of a molecular fragment using the AIMNet2 model.

  Parameters
  ----------

  Returns
  -------
    - log(str) : optimization logging
  """
  use_spin = calc_kwargs.get('use_spin', False)

  if use_spin:
    total_e = sum(atom.number for atom in fragment) - charge
    spin = 0 if total_e % 2 == 0 else 1

  _calc_kwargs = {
    **calc_kwargs,
    'charge':charge
    }
  if use_spin: _calc_kwargs['spin'] = spin
  _calc_kwargs.pop('use_spin', None)

  # preoptimizaion
  if preoptimizer_wrapper:
    preopt_calc = preoptimizer_wrapper(**_calc_kwargs)
    fragment.calc = preopt_calc
    preopt = BFGS(atoms=fragment)
    preopt.run(fmax=fmax, steps=steps)

  calc = calc_wrapper(**_calc_kwargs)
  fragment.calc = calc

  optimize = BFGS(atoms=fragment)
  optimize.run(fmax=fmax, steps=steps)

  if is_ipython and clear_log:
    clear_output(wait=True)
  #return fragment

def read_traj(trajFile:str, returnString=False)->tuple[ase.Atoms|str]:
  """
  Description
  -----------
  Read trajectory data from a file or a string in XYZ format. If a file path is provided, it reads the content of the file. If a string is provided, it assumes the string contains trajectory data in XYZ format.

  Parameters
  ----------
    - trajFile (str) : The trajectory file path or trajectory format string to read.
    - returnString (bool) : Whether to return XYZ format strings. If false, returns ase.Atoms objects. Default is False.

  Returns
  -------
    - ase.Atoms objects(tuple)  <- if returnString == Fasle
    - xyz format strings(tuple) <- if returnString == True    
  """
  if isfile(trajFile):
    with open(trajFile, "r") as file:
      traj = file.read()
  else:
    traj = trajFile

  # Split the trajectory file into multiple XYZ format strings
  pattern = re.compile("(\s?\d+\n.*\n(\s*[a-zA-Z]{1,2}(\s+-?\d+.\d+){3,3}\n?)+)")
  matched = pattern.findall(traj)

  xyzStringTuple = tuple(map(lambda groups : groups[0], matched))
  if returnString:
    return xyzStringTuple
  else:
    aseAtomsTuple = tuple(map(lambda xyzString : ase.io.read(StringIO(xyzString), format="xyz"), xyzStringTuple))
    return aseAtomsTuple
  

def relative_values(energy_series:list|tuple, relative_index:str|int="min")->tuple:
  """
  Description
  -----------
  Compute relative energy values based on a reference point.

  Parameters
  ----------
    - energy_series (list | tuple) : A list or tuple of absolute energy values.
    - relative_index (str | int, optional) : The reference point for computing relative energy values. IRC index starts with 0. Default is "min".
        If "min", the minimum value in the `energy_series` is used as the reference point.
        If an integer, the value at the specified index in the `energy_series` is used as the reference point.
          - `0` : First IRC point
          - `-1` : Last IRC point

    Returns
    -------
      - Relative energy(tuple) : A tuple of relative energy values.
  """
  if relative_index == "min":
    globalMin = min(energy_series)
    relative_energy_series = tuple(map(lambda absoluteValue : absoluteValue - globalMin, energy_series))
  elif type(relative_index) == int:
    relativeValue = energy_series[relative_index]
    relative_energy_series = tuple(map(lambda absoluteValue : absoluteValue - relativeValue, energy_series))
  else:
    raise ValueError(f"Invalid relative_index value : {relative_index}")
  return relative_energy_series


def fragments_params_processing(fragments_params: list | dict) -> tuple[list,list]:
  """
  Description
  -----------
  Process fragment parameters.
  
  Parameters
  ----------
    - fragments_params[list|dict] : If provided as a dictionary, keys are fragment names and values are fragment data. If provided as a list, it contains fragment data and default names are assigned.

  Returns
  -------
    - tuple( fragment_names, fragments ) : A tuple containing `fragment_names` and `fragments`.

  Examples
  --------
    dict --> with custom fragment names \n
    list --> with default fragment names
    >>> fragments_params_processing({
    ...     "Br-": (-1, [2]),
    ...     "CH3+": (+1, [1, 3, 4, 5]),
    ...     "Cl-": (-1, [6])
    ... })
    (['Br-', 'CH3+', 'Cl-'], [(-1, [2]), (+1, [1, 3, 4, 5]), (-1, [6])])
    >>> fragments_params_processing([
    ...     (-1, [2]),
    ...     (+1, [1, 3, 4, 5]),
    ...     (-1, [6])
    ... ])
    (['fragment_1', 'fragment_2', 'fragment_3'], [(-1, [2]), (+1, [1, 3, 4, 5]), (-1, [6])])
  """
  if isinstance(fragments_params, dict):
    fragment_names = list(fragments_params.keys())
    fragments = list(fragments_params.values())
  elif isinstance(fragments_params, list):
    fragment_names = list(map(lambda idx: f"fragment_{idx}", range(len(fragments_params))))
    fragments = fragments_params
  else:
    raise ValueError("Invalid fragments_params format : list or dict")

  # pre-defined variable name check
  uni = set(fragment_names) & set(["total", "distortion", "interaction"])
  if bool(uni):
    raise ValueError(f"{str(uni)} values are pre-defined names, E_total E_interaction E_distortion")

  return fragment_names, fragments


def fragmentBuilder(xyzString:str, indice:list)->ase.Atoms:
  """
  Description
  -----------
  Build a fragment(ase.Atoms) from an XYZ format string based on atomic indices.

  Parameters
  ----------
    - xyzString (str) : The XYZ format string representing the molecular system.
    - indice (list) : A list of atomic indices to include in the fragment. Atomic index starts with 1.

  Returns
  -------
    - ase.Atoms: An ASE Atoms object representing the fragment.
  """
  pattern = re.compile("([a-zA-Z]{1,2})((\s+-?\d+.\d+){3,3})")
  atoms = pattern.findall(xyzString)
  symbols, positions = list(), list()
  for idx in indice:
    atom = atoms[idx-1]
    symbols.append(atom[0])
    position_tuple = tuple(map(float, re.split("\s+", atom[1].strip())))
    positions.append(position_tuple)
  fragment = ase.Atoms(symbols=symbols, positions=positions)
  return fragment



def DIAS(
    xyzString:str, 
    fragments_params: list | dict, 
    calc_wrapper:Callable[..., Calculator], 
    preoptimizer_wrapper:Callable[..., Calculator]=None,
    clear_log=True, 
    **calc_kwargs
    )->dict:
  """
  Description
  -----------
  Calculate DIAS (Distortion Interaction and Atom Specific energies) for a single point in given XYZ trajectory.

  Parameters
  ----------

  Returns
  -------
    - DIASresult(dict) : A json containing single point DIAS results.
  """
  fragment_names, fragments = fragments_params_processing(fragments_params)
  
  # blank dict
  DIAS_result = dict()
  DIAS_result["molecule"] = dict()
  
  molecule = ase.io.read(StringIO(xyzString), format="xyz")
  net_charge = sum(tuple(map(lambda frag : frag[0], fragments)))

  # molecule total energy
  total_energy = potential_energy(
    calc_wrapper=calc_wrapper, 
    fragment=molecule, 
    charge=net_charge, 
    **calc_kwargs
    )
  DIAS_result["molecule"]["total"] = total_energy

  # distortion energy
  total_distortion = 0
  for name, fragment in zip(fragment_names, fragments):
    frag_chrg, frag_indices = fragment
    frag:ase.Atoms = fragmentBuilder(xyzString, frag_indices)
    pre_optimized_Energy = potential_energy(calc_wrapper=calc_wrapper, fragment=frag, charge=frag_chrg, **calc_kwargs)
    print(f"\nOptimizer : optimizing {name}")
    optimize(calc_wrapper=calc_wrapper, preoptimizer_wrapper=preoptimizer_wrapper, fragment=frag, charge=frag_chrg, clear_log=False, **calc_kwargs)
    optimized_Energy = potential_energy(calc_wrapper=calc_wrapper, fragment=frag, charge=frag_chrg, **calc_kwargs)

    fragDistortion = pre_optimized_Energy - optimized_Energy # eV unit
    DIAS_result[name] = {"distortion" : fragDistortion}
    total_distortion += fragDistortion
  DIAS_result["molecule"]["distortion"] = total_distortion

  # total interaction energy
  total_interaction = total_energy - total_distortion
  DIAS_result["molecule"]["interaction"] = total_interaction

  if is_ipython() and clear_log:
    clear_output(wait=True)

  return DIAS_result


def DIASparser(
    resultDictionary:dict, 
    fragType:str, 
    energyType:str=None, 
    relative_idx:None|str|int=None, 
    unit:str="kcal/mol"
    ):
  """
  Description
  -----------
  Parses DIAS results from a dictionary based on the specified fragment type, energy type, relative index, and unit.

  Parameters
  ----------
    - resultDictionary (dict) : The dictionary containing DIAS results.
    - fragType (str) : The type of fragment or molecule to parse (`molecule` or `fragment names`).
    - energyType (str) : The type of energy to parse ("total", "interaction", or "distortion").
    - relative_idx (None | str | int) : The index of the reference energy value for computing relative values.
    - unit (str, optional) : The unit for the energy values ("kcal/mol", "kJ/mol", "Hartree", or "eV"). Default is "kcal/mol".

  Returns
  -------
    - energies(tuple) : A tuple of parsed energy values.
  """
  unit = unit.upper()
  eV2unitFactor = {
      "KJ/MOL"    : 96.485332,
      "HARTREE"    : 0.0367493,
      "KCAL/MOL"  : 23.060548,
      "EV"        : 1
  # Ref : http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table-detail.html
  }
  match fragType:
    # > molecule
    case "molecule":
      molecule_energies = tuple(map(
        lambda energy_dictionary : energy_dictionary["molecule"], 
        list(resultDictionary.values())
        ))
      # > molecule > total | distortion | interaction
      match energyType:
        case "total":
          total = tuple(map(
            lambda molecule_energy : molecule_energy["total"] * eV2unitFactor[unit], 
            molecule_energies
            ))
          return relative_values(
            total, 
            relative_index=relative_idx
            ) if relative_idx is not None else total
        case "interaction":
          molecule_interaction = tuple(map(
            lambda molecule_energy : molecule_energy["interaction"] * eV2unitFactor[unit], 
            molecule_energies
            ))
          return relative_values(
            molecule_interaction, 
            relative_index=relative_idx
            ) if relative_idx is not None else molecule_interaction
        case "distortion" :
          molecule_distortion = tuple(map(
            lambda molecule_energy : molecule_energy["distortion"] * eV2unitFactor[unit], 
            molecule_energies
            ))
          return relative_values(
            molecule_distortion, 
            relative_index=relative_idx
            ) if relative_idx is not None else molecule_distortion
    # returns IRC index
    case "irc":
      return tuple(resultDictionary.keys())
    # > fragment > distortion
    case fragments:
      frag_distortion = tuple(map(
        lambda distortion_dictionary : distortion_dictionary[fragType]["distortion"] * eV2unitFactor[unit], 
        list(resultDictionary.values())
        ))
      return relative_values(
        energy_series=frag_distortion, 
        relative_index=relative_idx
        ) if relative_idx is not None else frag_distortion


def trajDIAS(
    calc_wrapper:Callable[..., Calculator], 
    trajFile:str, 
    fragments_params: list|dict, 
    preoptimizer_wrapper:Callable[..., Calculator]=None,
    resultSavePath:str="./result.json",
    save_kws:dict={}, 
    **calc_kwargs
    ):
  """
  Description
  -----------
  This function performs DIAS calculations for each IRC point in a trajectory file.

  Parameters
  ----------


    Returns
    -------
      - trajDIASresult(dict) : A dictionary containing DIAS results for each frame in the trajectory.
  """
  if calc_kwargs.get('use_spin'):
    warnings.warn('The current spin estimation is based on the total number of electrons.', UserWarning)
    
  trajDIASresult = dict()
  trajNum = len(read_traj(trajFile))
  for IRC_idx, xyzString in enumerate(read_traj(trajFile, returnString=True)):
    progress_bar(trajNum, IRC_idx)
    trajDIASresult[IRC_idx] = DIAS(
      xyzString=xyzString, 
      fragments_params=fragments_params, 
      calc_wrapper=calc_wrapper, 
      preoptimizer_wrapper=preoptimizer_wrapper,
      clear_log=True, 
      **calc_kwargs
      )
  json_dump(
    trajDIASresult=trajDIASresult, 
    trajFile=trajFile, 
    resultSavePath=resultSavePath, 
    **save_kws
    )
  print("ase_dias CALCULATION TERMINATED NORMALLY")
  return trajDIASresult


# new geometric_parameter
def geometric_parameter(trajFile:str, geo_param:dict)->tuple:
  """
  Description
  -----------
  This function computes geometric parameters, such as distance, angle, or dihedral,
  for each IRC point in a trajectory file based on given geo_param.

  Parameters
  ----------
  - trajFile (str) : The trajectory file path or trajectory format string to read.
  - geo_param (dict) : The geometric parameters to compute.
    - It should have the following structure: {axis_type: [list | tuple | str]}.
    - Supported axis_types for the values are "distance", "angle", and "dihedral".

  Returns
  -------
    - geometry paramters(tuple) : A tuple of geometric parameter values for each IRC point in the trajectory.

  Example
  -------
  >>> geo_param = { "distance" : "1 2" }
  >>> geometric_parameter(trajFile, geo_param)
  (3.0, 3.1, 3.2, ...)
  """ 
  axis_type, axis_values = tuple(geo_param.items())[0]  # indices in axis_values start with 1

  try:  # tuple or list is processed to start indexing from 0
    axis_values = tuple(map(lambda value: value - 1, axis_values))
  except TypeError:  # string types, such as '1 2 3 4', are to be processed in each match-case block
    pass

  match axis_type:
    case "distance":
      # distance | string   ex) geo_param = { "distance" : "1 2" }
      if isinstance(axis_values, str):
        indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
        num_indices = len(indices)
        if num_indices == 2:
          return tuple(map(lambda Atoms: Atoms.get_distance(*indices), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 2 but got {num_indices}, {axis_values}")
      # distance | [ list | tuple ]   ex) geo_param = { "distance" : [1,2] }
      elif isinstance(axis_values, (list, tuple)):
        if len(axis_values) == 2:
          return tuple(map(lambda Atoms: Atoms.get_distance(*axis_values), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 2 but got {len(axis_values)}, {axis_values}")
      else:
        raise ValueError("Invalid geo_param value")

    case "angle":
      # angle | string   ex) geo_param = { "angle" : "1 2 3" }
      if isinstance(axis_values, str):
        indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
        num_indices = len(indices)
        if num_indices == 3:
          return tuple(map(lambda Atoms: Atoms.get_angle(*indices), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 3 but got {num_indices}, {axis_values}")
      # angle | [ list | tuple ]   ex) geo_param = { "angle" : [1,2,3] }
      elif isinstance(axis_values, (list, tuple)):
        if len(axis_values) == 3:
          return tuple(map(lambda Atoms: Atoms.get_angle(*axis_values), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 3 but got {len(axis_values)}, {axis_values}")
      else:
        raise ValueError("Invalid geo_param value")

    case "dihedral":
      # dihedral | string   ex) geo_param = { "dihedral" : "1 2 3 4" }
      if isinstance(axis_values, str):
        indices = list(map(lambda value: int(value) - 1, re.split("\s+", axis_values)))
        num_indices = len(indices)
        if num_indices == 4:
          return tuple(map(lambda Atoms: Atoms.get_dihedral(*indices), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 4 but got {num_indices}, {axis_values}")
      # dihedral | [ list | tuple ]   ex) geo_param = { "dihedral" : [1,2,3,4] }
      elif isinstance(axis_values, (list, tuple)):
        if len(axis_values) == 4:
          return tuple(map(lambda Atoms: Atoms.get_dihedral(*axis_values), read_traj(trajFile)))
        else:
          raise ValueError(f"geo_param expected 4 but got {len(axis_values)}, {axis_values}")
      else:
        raise ValueError("Invalid geo_param value")

    case _:
      raise ValueError(f"Invalid param type : {axis_type} | Supported types : 'distance', 'angle', 'dihedral'")


def xaxis_formatter(trajFile:str, geo_param:dict):
  """
  Description
  -----------
  Format the x-axis label based on the geometric parameter specified in geo_param.

  Parameters
  ----------
    - trajFile (str): The trajectory file path or trajectory format string to read.
    - geo_param (dict): The geometric parameter for formatting the x-axis label.
      - It should contain only one geometric parameter.
      - The key should represent the type of geometric parameter, such as "angle", "distance", or "dihedral".
      - The value should be either a string representing indices (e.g., "1 2"), or a list/tuple of indices.

  Returns
  -------
    - x_axis_type, xlabel (tuple): A tuple containing the type of geometric parameter and the formatted x-axis label.

  Note
  ----
  Suppose the XYZ coordinates are on an angstrom scale.
  """
  # check sigle geo_param
  if len(geo_param) != 1:
    raise ValueError(f"geo_param expected 1 geometric parameter, got {len(geo_param)} parameters")

  paramType, param = tuple(geo_param.items())[0] # indices in param starts with 1

  if isinstance(param, str):
    param = param.strip()
    param_tuple =  tuple(string.strip() for string in re.split("\s+", param)) # indice
  elif isinstance(param, (list|tuple)):
    param_tuple = tuple(str(integer) for integer in param) # indice
  else:
    raise ValueError("geo_param value error")
  
  symbols = tuple(get_symbol(trajFile, idx) for idx in param_tuple) # symbols
  symbol_index = "-".join(list(map(lambda arg : f"{arg[0]}({arg[1]})" ,zip(symbols, param_tuple))))

  ParamType = paramType[0].upper() + paramType[1:] # e.g. distance --> Distance
      
  axis_unit = {"angle": "Degree", "distance": "Angstrom", "dihedral": "Degree"} # suppose the XYZ coordinates are on an angstrom scale
  x_axis_format = f'{ParamType} {symbol_index} / {axis_unit[paramType]}'
  return paramType, x_axis_format


def dias_run(
    calc_wrapper:Callable[..., Calculator], 
    trajFile:str, fragments_params: list | dict, 
    preoptimizer_wrapper:Callable[..., Calculator]=None,
    mode:str="calculation", 
    DIAresultPath:str="./result.json", 
    axis_type:str="irc", 
    geo_param:dict | None =None, 
    unit:str="kcal/mol", 
    include_fragments=False, 
    relative_idx:None|str|int=0, 
    plot_highlights:dict={"marker": False, "linestyle": True}, 
    resultSavePath:str="./result.json", 
    horizontal_line:bool=True, 
    save_kws:dict={},
    **calc_kwargs
    )->None:
  """
  Description
  -----------
  Performs Activation Strain Model (ASM) & Distortion/Interaction (D/I) analysis on an XYZ trajectory using the AIMNet2 calculator.

  Parameters
  ----------

  Note
  ----
  Important parameters

  1. axis_type, geo_param --> your plot axis type
  2. mode, DIAresultPath  --> plot without calculation

  Example
  -------
  >>> fragments_params = {
  ...     "Br-": (-1, [2]),
  ...     "CH3+": (+1, [1, 3, 4, 5]),
  ...     "Cl-": (-1, [6])
  ... }
  >>> dias_run("trajectory.xyz", fragments_params)
  """
  # unit check
  available_units = ["EV", "KCAL/MOL", "KJ/MOL", "HARTREE"]
  if unit.upper() not in available_units:
    raise ValueError(f"Unit '{unit}' is not supported. Please use one of the following case-insensitive units: \n{available_units}")
      
  # key parameter check
  if not fragments_params:
    raise ValueError(f"'fragments_params' expected str type but got {fragments_params}")
  elif not trajFile:
    raise ValueError(f"'trajFile' expected str type but got {trajFile}")

  # check mode
  if mode == "calculation":
    # geo_param check
    if (axis_type != "irc") & (geo_param == None):
      raise ValueError("geo_param expected '[ dict | list ]' when axis_type specified ( != irc ), got None")
    elif (axis_type == "irc") & (geo_param != None):
      warning_msg = f"Message : got geo_param but axis_type is not specified(default value, 'irc')\nSo internally set axis_type = f'{tuple(geo_param.values())[0]}'"
      print(f"\033[91m{warning_msg}\033[0m")
    # run DIAS
    DIASresult = trajDIAS(
      calc_wrapper=calc_wrapper, 
      preoptimizer_wrapper=preoptimizer_wrapper,
      trajFile=trajFile, 
      fragments_params=fragments_params, 
      resultSavePath=resultSavePath, 
      save_kws=save_kws,
      **calc_kwargs
      )
  
  elif mode == "plot":
    if DIAresultPath == None:
      raise ValueError(f"DIAresultPath is not specified, got {DIAresultPath} When mode='plot', DIAresultPath must be provided.")
    elif not os.path.isfile(DIAresultPath):
      raise FileNotFoundError(f"DIAresultPath not found, got {DIAresultPath}. \n\t - Windows paths may be interpreted as escape sequences in Python, potentially leading to errors. Using a raw string is recommended.")
    with open(DIAresultPath, "r") as file:
      DIASresult = json.load(file)["result"]

  # plot style option, marker & linestype
  mk_bool = plot_highlights.get("marker")
  ls_bool = plot_highlights.get("linestyle")

  # get geometric axis and xlabel
  match axis_type:
    case "irc":
      geo_param_axis_ = DIASparser(DIASresult, fragType="irc")
      xlabel_ = "IRC points"
    case "distance" | "angle" | "dihedral":
      paramType, x_axis_format = xaxis_formatter(trajFile, geo_param)
      if paramType == axis_type:
        geo_param_axis_ = geometric_parameter(trajFile, geo_param)
        xlabel_ = x_axis_format
      else:
        raise ValueError(f"Incosistancy between geo_param / axis_type, geo_param got {paramType} but axis_type got {axis_type}")
  
  # ylabel
  ylabel_ = f"{'Rel. ' if relative_idx is not None else ''}Energy / {unit}"

  # show horizontal reference line
  if horizontal_line:
    plt.plot(geo_param_axis_, [0]*len(geo_param_axis_), linewidth=.5, color="black")

  # plot basic components | total, distortion, interaction of super-molecule
  default_energy_componet = ["total", "distortion", "interaction"] # energies to plot
  colors_list = ["black", "tab:blue", "tab:red"]
  marker_list = ["+", "x", "4"]
  linestyle_list = ["-", "-.", "--"]
  for idx, EnergyType in enumerate(default_energy_componet):
    tmp_energy = DIASparser(
      resultDictionary=DIASresult, 
      fragType="molecule", 
      energyType = EnergyType, 
      relative_idx=relative_idx, 
      unit=unit
      ) 
    plt.plot(
      geo_param_axis_, 
      tmp_energy, 
      label=f"E_{EnergyType[:3]}",
      color=colors_list[idx], 
      marker=marker_list[idx] if mk_bool else None, 
      linestyle=linestyle_list[idx] if ls_bool else None
      )
  
  # if True, plot fragment distortion energies
  if include_fragments == True:
    line_style_= ":"
    fragment_names, _ = fragments_params_processing(fragments_params)
    n_frags = len(fragment_names)
    clr_list = husl_palette(n_frags)
    mkr_list = markers_(n_frags) 
    for idx, fragment in enumerate(fragment_names):
      frag_dist_energy =  DIASparser(
        resultDictionary=DIASresult, 
        fragType=fragment, 
        energyType = "distortion", 
        relative_idx=relative_idx, 
        unit=unit
        )
      plt.plot(
        geo_param_axis_, 
        frag_dist_energy, 
        label=f"E_dis({fragment})", 
        color=clr_list[idx], 
        marker=mkr_list[idx] if mk_bool else None, 
        linestyle=line_style_ if ls_bool else None
        ) 
  
  plt.legend()
  plt.xlabel(xlabel_)
  plt.ylabel(ylabel_)
