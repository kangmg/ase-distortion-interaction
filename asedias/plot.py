import ase
import warnings
from typing import Union
import matplotlib.pyplot as plt
from asedias.utils import DIASparser


def xaxis_formatter(images:list[ase.Atoms], indices:list=None)->str:
    """
    returns x-axis text & x-axis parameters
    """
    _first_image = images[0]
    if not indices:
        # IRC
        xaxis = 'IRC Points'
        params = tuple(i for i in range(len(_first_image)))
    else:
        axis_unit = {"Angle": "Degree", "Distance": "Å", "Dihedral Angle": "Degree"}
        axis_head = {"Angle": "∠", "Distance": "r", "Dihedral Angle": "∠"}
        param_type, params = geometric_analysis(images=images, indices=indices)
        symbols = _first_image.get_chemical_symbols()
        selected_symbols = list(symbols[idx] for idx in indices)
        # symbols_text = "-".join(list(f"{arg[0]}({arg[1]})" for arg in zip(selected_symbols, indices))) # e.g. H0-O1-H2
        symbols_text = "-".join(selected_symbols) # e.g. H-O-H
        xaxis = f'{axis_head[param_type]} {symbols_text} / {axis_unit[param_type]}'

    return xaxis, params


def geometric_analysis(images:list[ase.Atoms], indices:list[int])->tuple:
    """
    get list of internal coordinates
    
    Parameters
    ----------
    """
    def _is_monotonic(lst:list):
        """check list is monotonic list"""
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)) or \
               all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))
    
    # check invalid index
    _index_diff = set(indices) - set(i for i in range(len(images[0])))
    assert _index_diff == set(), f"Invalid indices, {_index_diff}"

    # distance
    if len(indices) == 2:
        params = tuple(map(lambda atoms: atoms.get_distance(*indices), images))
        param_type = "Distance"
    # angle
    elif len(indices) == 3:
        params = tuple(map(lambda atoms: atoms.get_angle(*indices), images))
        param_type = "Angle"
    # dihedral angle
    elif len(indices) == 4:
        params = tuple(map(lambda atoms: atoms.get_dihedral(*indices), images))
        param_type = "Dihedral Angle"
    else:
        raise ValueError(f"Invalid number of indices. Expected 2/3/4, got {len(indices)}")

    # check monotonicity 
    if not _is_monotonic(params):
        warnings.warn("The x-axis values are not monotonic. This may lead to unexpected behavior in the plot.", UserWarning)

    return param_type, params


def husl_palette(pal_len:int)->list:
    """
    seaborn husl palette without seaborn 

    Parameters
    ----------
      - pal_len(int) : n_colors in husl palette ( 2 =< pal_len =< 9 )

    Returns
    -------
      - palette(list) : RGB list
    """
    if pal_len == 2:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.21044753832183283, 0.6773105080456748, 0.6433941168468681)]
    elif pal_len == 3:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
                (0.23299120924703914, 0.639586552066035, 0.9260706093977744)]
    elif pal_len == 4:
        return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                 (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
                 (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
                 (0.6423044349219739, 0.5497680051256467, 0.9582651433656727)]
    elif pal_len == 5:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.6804189127793346, 0.6151497514677574, 0.19405452111445337),
                (0.20125317221201128, 0.6907920815379025, 0.47966761189275336),
                (0.2197995660828324, 0.6625157876850336, 0.7732093159317209),
                (0.8004936186423958, 0.47703363533737203, 0.9579547196007522)]
    elif pal_len == 6:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.7350228985632719, 0.5952719904750953, 0.1944419133847522),
                (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
                (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
                (0.23299120924703914, 0.639586552066035, 0.9260706093977744),
                (0.9082572436765556, 0.40195790729656516, 0.9576909250290225)]
    elif pal_len == 7:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.7757319041862729, 0.5784925270759935, 0.19475566538551875),
                (0.5105309046900421, 0.6614299289084904, 0.1930849118538962),
                (0.20433460114757862, 0.6863857739476534, 0.5407103379425205),
                (0.21662978923073606, 0.6676586160122123, 0.7318695594345369),
                (0.5049017849530067, 0.5909119231215284, 0.9584657252128558),
                (0.9587050080494409, 0.3662259565791742, 0.9231469575614251)]
    elif pal_len == 8:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.8087954113106306, 0.5634700050056693, 0.19502642696727285),
                (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
                (0.19783576093349015, 0.6955516966063037, 0.3995301037444499),
                (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
                (0.22335772267769388, 0.6565792317435265, 0.8171355503265633),
                (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
                (0.9603888539940703, 0.3814317878772117, 0.8683117650835491)]
    elif pal_len == 9:
        return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
                (0.8369430560927636, 0.5495828952802333, 0.1952683223448124),
                (0.6430915736746491, 0.6271955086583126, 0.19381135329796756),
                (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
                (0.20582072623426667, 0.6842209016721069, 0.5675558225732941),
                (0.2151139535594307, 0.6700707833028816, 0.7112365203426209),
                (0.23299120924703914, 0.639586552066035, 0.9260706093977744),
                (0.731751635642941, 0.5128186367840487, 0.9581005178234921),
                (0.9614880299080136, 0.3909885385134758, 0.8298287106954371)]
    else:
        raise ValueError(f"{pal_len} is not in [2, 9], 2 =< pal_len =< 9")


def get_markers(lens:int)->list:
    """
    matplotlib markers

    Parameters
    ----------
      - len(lens) : number of plots ( 2 =< lens =< 9 )

    Returns
    -------
      - palette(list) : RGB list
    """
    if lens == 2:
        return ["v", "^"]
    elif lens == 3:
        return ["2", "3", "4"]
    elif lens == 4:
        return [">", "<", "^", "v"]
    elif lens == 5:
        return [">", "<", "^", "s", "D"]
    elif lens == 6:
        return ["1", "3", "4", "v", "<", ">"]
    elif lens == 7:
        return ["1", "3", "4", "v", "<", ">", "^"]
    elif lens == 8:
        return ["1", "2", "3", "4", "v", "<", ">", "^"]
    elif lens == 9:
        return [">", "<", "^", "s", "D", "1", "2", "3", "4"]
    else:
        raise ValueError(f"{lens} is not in [2, 9]")


def plot_dias(resultDict:dict, axis_type:str, include_fragments:bool=False, 
              yaxis_unit:str='eV', relative_idx:Union[str, int]=0, **kwargs):
    """
    Plot dias result
    """
    # plot style option, marker & linestype
    mk_bool = kwargs.get("marker", False)
    ls_bool = kwargs.get("linestyle", True)
    hline = kwargs.get('hline', True)
    

    if axis_type.upper() == 'IRC':
        xaxis_formatter()
        

#
#  # get geometric axis and xlabel
#  match axis_type:
#    case "irc":
#      geo_param_axis_ = DIASparser(DIASresult, fragType="irc")
#      xlabel_ = "IRC points"
#    case "distance" | "angle" | "dihedral":
#      paramType, x_axis_format = xaxis_formatter(trajFile, geo_param)
#      if paramType == axis_type:
#        geo_param_axis_ = geometric_parameter(trajFile, geo_param)
#        xlabel_ = x_axis_format
#      else:
#        raise ValueError(f"Incosistancy between geo_param / axis_type, geo_param got {paramType} but axis_type got {axis_type}")
#  
#  # ylabel
#  ylabel_ = f"{'Rel. ' if relative_idx is not None else ''}Energy / {unit}"
#
#  # show horizontal reference line
#  if horizontal_line:
#    plt.plot(geo_param_axis_, [0]*len(geo_param_axis_), linewidth=.5, color="black")
#
#  # plot basic components | total, distortion, interaction of super-molecule
#  default_energy_componet = ["total", "distortion", "interaction"] # energies to plot
#  colors_list = ["black", "tab:blue", "tab:red"]
#  marker_list = ["+", "x", "4"]
#  linestyle_list = ["-", "-.", "--"]
#  for idx, EnergyType in enumerate(default_energy_componet):
#    tmp_energy = DIASparser(
#      resultDictionary=DIASresult, 
#      fragType="molecule", 
#      energyType = EnergyType, 
#      relative_idx=relative_idx, 
#      unit=unit
#      ) 
#    plt.plot(
#      geo_param_axis_, 
#      tmp_energy, 
#      label=f"E_{EnergyType[:3]}",
#      color=colors_list[idx], 
#      marker=marker_list[idx] if mk_bool else None, 
#      linestyle=linestyle_list[idx] if ls_bool else None
#      )
#  
#  # if True, plot fragment distortion energies
#  if include_fragments == True:
#    line_style_= ":"
#    fragment_names, _ = fragments_params_processing(fragments_params)
#    n_frags = len(fragment_names)
#    clr_list = husl_palette(n_frags)
#    mkr_list = markers_(n_frags) 
#    for idx, fragment in enumerate(fragment_names):
#      frag_dist_energy =  DIASparser(
#        resultDictionary=DIASresult, 
#        fragType=fragment, 
#        energyType = "distortion", 
#        relative_idx=relative_idx, 
#        unit=unit
#        )
#      plt.plot(
#        geo_param_axis_, 
#        frag_dist_energy, 
#        label=f"E_dis({fragment})", 
#        color=clr_list[idx], 
#        marker=mkr_list[idx] if mk_bool else None, 
#        linestyle=line_style_ if ls_bool else None
#        ) 
#  
#  plt.legend()
#  plt.xlabel(xlabel_)
#  plt.ylabel(ylabel_)