import ase
import warnings
from typing import Union
import ase.atom
import matplotlib.pyplot as plt
from asedias.utils import DIASparser


def geometric_analysis(images:list[ase.Atoms], geometric_indices:list[int])->tuple:
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
    _index_diff = set(geometric_indices) - set(i for i in range(len(images[0])))
    assert _index_diff == set(), f"Invalid geometric_indices, {_index_diff}"

    # distance
    if len(geometric_indices) == 2:
        xaxis = tuple(map(lambda atoms: atoms.get_distance(*geometric_indices), images))
        xaxis_type = "Distance"
    # angle
    elif len(geometric_indices) == 3:
        xaxis = tuple(map(lambda atoms: atoms.get_angle(*geometric_indices), images))
        xaxis_type = "Angle"
    # dihedral angle
    elif len(geometric_indices) == 4:
        xaxis = tuple(map(lambda atoms: atoms.get_dihedral(*geometric_indices), images))
        xaxis_type = "Dihedral Angle"
    else:
        raise ValueError(f"Invalid number of geometric_indices. Expected 2/3/4, got {len(geometric_indices)}")

    # check monotonicity 
    if not _is_monotonic(xaxis):
        warnings.warn("The x-axis values are not monotonic. This may lead to unexpected behavior in the plot.", UserWarning)

    return xaxis_type, xaxis


def xaxis_formatter(images:list[ase.Atoms], geometric_indices:list=None)->str:
    """
    returns x-axis text & x-axis parameters
    """
    _first_image = images[0]
    if not geometric_indices:
        # IRC
        xaxis_string = 'IRC Points'
        xaxis = tuple(str(i) for i in range(len(images)))
    else:
        xaxis_unit = {"Angle": "Degree", "Distance": "Å", "Dihedral Angle": "Degree"}
        xaxis_head = {"Angle": "∠", "Distance": "r", "Dihedral Angle": "∠"}
        xaxis_type, xaxis = geometric_analysis(images=images, geometric_indices=geometric_indices)
        symbols = _first_image.get_chemical_symbols()
        selected_symbols = list(symbols[idx] for idx in geometric_indices)
        # symbols_text = "-".join(list(f"{arg[0]}({arg[1]})" for arg in zip(selected_symbols, geometric_indices))) # e.g. H0-O1-H2
        symbols_text = "-".join(selected_symbols) # e.g. H-O-H
        xaxis_string = f'{xaxis_head[xaxis_type]} {symbols_text} / {xaxis_unit[xaxis_type]}'

    return xaxis_string, xaxis


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


def plot_dias(images:list[ase.Atoms], resultDict:dict, include_fragments:bool=False, yaxis_unit:str='eV', 
              relative_idx:Union[str, int]=0, geometric_indices:list[int]=None, save_name:str=None, **kwargs):
    """
    Plot dias result
    """
    # plot style option, marker & linestype
    mk_bool = kwargs.get("marker", False)
    ls_bool = kwargs.get("linestyle", True)
    hline = kwargs.get('hline', True)

    if yaxis_unit.upper() == 'EV':
        _unit = 'eV'
    elif yaxis_unit.upper() == 'KCAL/MOL':
        _unit = 'kcal/mol'
    elif yaxis_unit.upper() == 'KJ/MOL':
        _unit = 'kJ/mol'
    elif yaxis_unit.upper() == 'HARTREE':
        _unit = 'Hartree'
    else:
        raise ValueError("Unsupported yaxis_unit") 
    
    xaxis_string, xaxis = xaxis_formatter(images=images, geometric_indices=geometric_indices)
    yaxis_string = f"{'Rel. ' if relative_idx is not None else ''}Energy / {_unit}"

    # holizontal line
    if hline:
        plt.plot(xaxis, [0]*len(xaxis), linewidth=.5, color='black')
    
    # basic dias results
    basic_energy_componet = ["total", "distortion", "interaction"] # energies to plot
    colors_list = ["black", "tab:blue", "tab:red"]
    marker_list = ["+", "x", "4"]
    linestyle_list = ["-", "-.", "--"]
    for idx, energy_type in enumerate(basic_energy_componet):
        energy_series = DIASparser(
            resultDict=resultDict, 
            frag_type="molecule", 
            energy_type=energy_type, 
            relative_idx=relative_idx, 
            unit_conversion=_unit
            )
        plt.plot(
            xaxis, 
            energy_series, 
            label=f"E_{energy_type[:3]}",
            color=colors_list[idx], 
            marker=marker_list[idx] if mk_bool else None, 
            linestyle=linestyle_list[idx] if ls_bool else None
            )

    if include_fragments:
        frag_names = set(resultDict['0'].keys()) - {'molecule', 'success'}
        frag_names = sorted(list(frag_names))

        line_style_= ":"
        n_frags = len(frag_names)
        clr_list = husl_palette(n_frags)
        mkr_list = get_markers(n_frags) 

        for idx, fragment in enumerate(frag_names):
            frag_dist_energy =  DIASparser(
                resultDict=resultDict, 
                frag_type=fragment, 
                energy_type = "distortion", 
                relative_idx=relative_idx,
                unit_conversion=_unit
                )
            plt.plot(
                xaxis, 
                frag_dist_energy, 
                label=f"E_dis({fragment})", 
                color=clr_list[idx], 
                marker=mkr_list[idx] if mk_bool else None, 
                linestyle=line_style_ if ls_bool else None
                ) 
    
    plt.legend()
    plt.xlabel(xaxis_string)
    plt.ylabel(yaxis_string)

    if save_name: plt.savefig(f"{save_name}.png")
    plt.show()