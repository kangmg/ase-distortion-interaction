from datetime import datetime
import json
from os.path import basename, isfile
import re
from io import StringIO
import plotly.colors as pc
import plotly.graph_objects as go
from itertools import cycle
import numpy as np
from asedias.data import atomic_number2element_symbol, covalent_radii
import ase


def animation(images:list[ase.Atoms], frag_indices:list=None, colorby:str="fragment",
                   cmap:str=None, covalent_radius_percent:float=108., **kwargs):
    """
    Visualization of molecular structures in 3D using Plotly.

    Parameters
    ----------


    Additional Keyword Arguments
    ----------------------------
    - alpha_atoms : float
        Opacity of atoms. Default: 0.55.
    - alpha_bonds : float
        Opacity of bonds. Default: 0.55.
    - atom_scaler : float
        Scaling factor for atom visualization. Default: 20.
    - bond_scaler : float
        Scaling factor for bond visualization. Default: 10,000.
    - legend : bool
        Show legend in the visualization. Default: False.

    Returns
    -------
    None. Displays a 3D interactive visualization.
    """
    def _get_pallete_colors(cmap: str | list, n: int):
        """Generate a list of n colors from a Plotly colormap or a custom color list."""
        if not cmap:
            cmap = 'Plotly3'
        try:
            pc.get_colorscale(cmap)
        except Exception:
            print("\033[31m[WARNING]\033[0m", f"'{cmap}' is not a valid Plotly colormap. Defaulting to 'Plotly3'.")
            cmap = 'Plotly3'

        if isinstance(cmap, str):
            colors = pc.get_colorscale(cmap)
            return list(pc.sample_colorscale(colors, [ratio for ratio in np.linspace(0, 1, n + 1)[1:]], colortype='rgb'))
        elif isinstance(cmap, list):
            cyclic_iterator = cycle(cmap)
            return [next(cyclic_iterator) for _ in range(n)]

    def _get_frag_color(frag_indices:list):
        """
        
        """
        colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F5A623", "#8E44AD", 
            "#2ECC71", "#F39C12", "#1F618D", "#F1948A", "#7D3C98"
            ]
        frag_colors = colors[:len(frag_indices)]
        return frag_colors

    # Default visualization parameters
    alpha_atoms = kwargs.get("alpha_atoms", 1)
    alpha_bonds = kwargs.get("alpha_bonds", 0.7)
    atom_scaler = kwargs.get("atom_scaler", 300)
    bond_scaler = kwargs.get("bond_scaler", 8200000)
    legend = kwargs.get("legend", True)

    # Calculate global ranges for all molecular structures
    all_positions = np.vstack([atoms.positions for atoms in images])
    range_array = np.vstack([[np.min(all_positions[:, i]) for i in range(3)], [np.max(all_positions[:, i]) for i in range(3)]])
    x_range, y_range, z_range = range_array[:, 0], range_array[:, 1], range_array[:, 2]
    padding = 0.1
    x_range = [x_range[0] - padding, x_range[1] + padding]
    y_range = [y_range[0] - padding, y_range[1] + padding]
    z_range = [z_range[0] - padding, z_range[1] + padding]

    frames = []

    if colorby == "molecule":
        num_of_atoms = max(len(atoms) for atoms in images)
        num_of_molecules = len(images)
        palette = _get_pallete_colors(cmap, num_of_molecules)

        for mol_idx, atoms in enumerate(images):
            frame_data = []
            color = palette[mol_idx]

            # Add atoms
            atom_size = max(np.log10(atom_scaler / num_of_atoms) * 5, 2)
            frame_data.append(go.Scatter3d(
                x=atoms.positions[:, 0],
                y=atoms.positions[:, 1],
                z=atoms.positions[:, 2],
                mode='markers',
                opacity=alpha_atoms,
                marker=dict(size=atom_size, color=color),
                name=f'Molecule {mol_idx + 1} atoms'
            ))

            # Add bonds (simple covalent bond estimation based on radii)
            # bond thickness
            bond_thickness = np.maximum(np.log10(bond_scaler / num_of_atoms) * 2, 1)
            bond_x, bond_y, bond_z = [], [], []
            for i, pos1 in enumerate(atoms.positions):
                for j, pos2 in enumerate(atoms.positions[i + 1:], start=i+1):
                    dist = np.linalg.norm(pos1 - pos2)
                    radius_sum = covalent_radii[atomic_number2element_symbol[atoms.numbers[i]]] + covalent_radii[atomic_number2element_symbol[atoms.numbers[j]]]
                    if dist <= radius_sum * covalent_radius_percent / 100:
                        bond_x.extend([pos1[0], pos2[0], None])
                        bond_y.extend([pos1[1], pos2[1], None])
                        bond_z.extend([pos1[2], pos2[2], None])
            frame_data.append(go.Scatter3d(
                x=bond_x, y=bond_y, z=bond_z,
                mode='lines',
                opacity=alpha_bonds,
                line=dict(width=bond_thickness, color=color),
                name=f'Molecule {mol_idx + 1} bonds'
            ))
            frames.append(go.Frame(data=frame_data, name=str(mol_idx)))

    elif colorby == "fragment":
        num_of_atoms = max(len(atoms) for atoms in images)
        num_of_molecules = len(images)
        frag_colors = _get_frag_color(frag_indices=frag_indices)
        atom_size = max(np.log10(atom_scaler / num_of_atoms) * 5, 2)
        for mol_idx, atoms in enumerate(images):
            frame_data = []
            for frag_idx, (atomic_indices, frag_color) in enumerate(zip(frag_indices, frag_colors)):
                frag_atoms = atoms[atomic_indices]
                
                # Add atoms
                frame_data.append(go.Scatter3d(
                    x=frag_atoms.positions[:, 0],
                    y=frag_atoms.positions[:, 1],
                    z=frag_atoms.positions[:, 2],
                    mode='markers',
                    opacity=alpha_atoms,
                    marker=dict(size=atom_size, color=frag_color),
                    name=f'Fragment {frag_idx + 1}',
                    legendgroup=f'Fragment {frag_idx + 1}',
                    showlegend=True if legend else False,
                    hoverinfo='text',
                    hovertext=f'Fragment {frag_idx + 1}'
                ))


                # Add bonds
                bond_x, bond_y, bond_z = [], [], []
                # bond thickness
                bond_thickness = np.maximum(np.log10(bond_scaler / num_of_atoms) * 2, 1)
                for i, pos1 in enumerate(frag_atoms.positions):
                    for j, pos2 in enumerate(frag_atoms.positions[i + 1:], start=i+1):
                        dist = np.linalg.norm(pos1 - pos2)
                        radius_sum = covalent_radii[atomic_number2element_symbol[frag_atoms.numbers[i]]] + covalent_radii[atomic_number2element_symbol[frag_atoms.numbers[j]]]
                        if dist <= radius_sum * covalent_radius_percent / 100:
                            bond_x.extend([pos1[0], pos2[0], None])
                            bond_y.extend([pos1[1], pos2[1], None])
                            bond_z.extend([pos1[2], pos2[2], None])
                frame_data.append(go.Scatter3d(
                    x=bond_x, y=bond_y, z=bond_z,
                    mode='lines',
                    opacity=alpha_bonds,
                    line=dict(width=bond_thickness, color=frag_color),
                    name=f'Fragment {frag_idx + 1}',
                    showlegend=False,
                    legendgroup=f'Fragment {frag_idx + 1}',
                    hoverinfo='text',
                    hovertext=f'Fragment {frag_idx + 1}'
                ))
            frames.append(go.Frame(data=frame_data, name=str(mol_idx)))

    # Create the figure
    fig = go.Figure()
    fig.add_traces(frames[0].data)

    # Update the layout and add animation controls
    fig.frames = frames
    fig.update_layout(
        width=800, height=600,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 10, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate',
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate',
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}], 
                 'label': str(k), 
                 'method': 'animate'} for k, f in enumerate(fig.frames)
            ]
        }],
        scene=dict(
            xaxis=dict(range=x_range, visible=False),
            yaxis=dict(range=y_range, visible=False),
            zaxis=dict(range=z_range, visible=False),
            aspectmode='data',
            camera_projection=dict(type='orthographic'),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        showlegend=True if legend else False
    )

    fig.show()


def read_traj(trajFile:str, returnString=False)->list[ase.Atoms|str]: 
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
    - ase.Atoms objects(list)  <- if returnString == Fasle
    - xyz format strings(list) <- if returnString == True    
  """
  if isfile(trajFile):
    with open(trajFile, "r") as file:
      traj = file.read()
  else:
    traj = trajFile

  # Split the trajectory file into multiple XYZ format strings
  pattern = re.compile(r"(\s?\d+\n.*\n(\s*[a-zA-Z]{1,2}(\s+-?\d+.\d+){3,3}\n?)+)")
  matched = pattern.findall(traj)

  xyzStringTuple = list(map(lambda groups : groups[0], matched))
  if returnString:
    return xyzStringTuple
  else:
    aseAtomsTuple = list(map(lambda xyzString : ase.io.read(StringIO(xyzString), format="xyz"), xyzStringTuple))
    return aseAtomsTuple
  

def progress_bar(total, current)->None:
  """
  Description
  -----------
    Display a progress bar indicating the completion status of a task.

  Parameters
  ----------
    - total (int): The total number of units representing the task's completion.
    - current (int): The current progress, representing the number of units completed.

  Example
  -------
  >>> progress_bar(100, 50)
  Processing . . .
    50.0%  |===========>                |  50 / 100
  """
  percent = round(current/total * 100, 2)
  num_progress_bar = round(int(percent) // 5)
  num_redidual_bar =  20 - num_progress_bar
  progress_bar_string = "\033[34mProcessing . . .  \n  {}%  |{}>{}|  {} / {}\033[0m".format(percent, num_progress_bar * "=", num_redidual_bar * " ", current, total)
  print(progress_bar_string)


def is_ipython():
  try:
    __IPYTHON__
    return True
  except NameError:
    return False


def husl_palette(pal_len:int)->list:
  """
  Description
  -----------
  seaborn husl palette without seaborn 

  Parameters
  ----------
    - pal_len(int) : n_colors in husl palette ( 2 =< pal_len =< 9 )

  Returns
  -------
    - palette(list) : RGB list
  """
  match pal_len:
    case 2:
      return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
              (0.21044753832183283, 0.6773105080456748, 0.6433941168468681)]
    case 3:
      return [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
              (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
              (0.23299120924703914, 0.639586552066035, 0.9260706093977744)]
    case 4:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
               (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
               (0.6423044349219739, 0.5497680051256467, 0.9582651433656727)]
    case 5:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.6804189127793346, 0.6151497514677574, 0.19405452111445337),
               (0.20125317221201128, 0.6907920815379025, 0.47966761189275336),
               (0.2197995660828324, 0.6625157876850336, 0.7732093159317209),
               (0.8004936186423958, 0.47703363533737203, 0.9579547196007522)]
    case 6:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.7350228985632719, 0.5952719904750953, 0.1944419133847522),
               (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
               (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
               (0.23299120924703914, 0.639586552066035, 0.9260706093977744),
               (0.9082572436765556, 0.40195790729656516, 0.9576909250290225)]
    case 7:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.7757319041862729, 0.5784925270759935, 0.19475566538551875),
               (0.5105309046900421, 0.6614299289084904, 0.1930849118538962),
               (0.20433460114757862, 0.6863857739476534, 0.5407103379425205),
               (0.21662978923073606, 0.6676586160122123, 0.7318695594345369),
               (0.5049017849530067, 0.5909119231215284, 0.9584657252128558),
               (0.9587050080494409, 0.3662259565791742, 0.9231469575614251)]
    case 8:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.8087954113106306, 0.5634700050056693, 0.19502642696727285),
               (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
               (0.19783576093349015, 0.6955516966063037, 0.3995301037444499),
               (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
               (0.22335772267769388, 0.6565792317435265, 0.8171355503265633),
               (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
               (0.9603888539940703, 0.3814317878772117, 0.8683117650835491)]
    case 9:
      return  [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
               (0.8369430560927636, 0.5495828952802333, 0.1952683223448124),
               (0.6430915736746491, 0.6271955086583126, 0.19381135329796756),
               (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
               (0.20582072623426667, 0.6842209016721069, 0.5675558225732941),
               (0.2151139535594307, 0.6700707833028816, 0.7112365203426209),
               (0.23299120924703914, 0.639586552066035, 0.9260706093977744),
               (0.731751635642941, 0.5128186367840487, 0.9581005178234921),
               (0.9614880299080136, 0.3909885385134758, 0.8298287106954371)]
    case _:
      raise ValueError(f"{pal_len} is not in [2, 9], 2 =< pal_len =< 9")


def markers_(lens:int)->list:
  """
  Description
  -----------
  matplotlib markers

  Parameters
  ----------
    - len(lens) : number of plots ( 2 =< lens =< 9 )

  Returns
  -------
    - palette(list) : RGB list
  """
  match lens:
    case 2:
      return ["v", "^"]
    case 3:
      return ["2", "3", "4"]
    case 4:
      return [">", "<", "^", "v"]
    case 5:
      return [">", "<", "^", "s", "D"]
    case 6:
      return ["1", "3", "4", "v", "<", ">"]
    case 7:
      return ["1", "3", "4", "v", "<", ">", "^"]
    case 8:
      return ["1", "2", "3", "4", "v", "<", ">", "^"]
    case 9:
      return [">", "<", "^", "s", "D", "1", "2", "3", "4"]
    case _:
      raise ValueError(f"{lens} is not in [2, 9], 2 =< lens =< 9")


# def json_dump(trajDIASresult:dict, trajFile:str, resultSavePath:str="./result.json", title=None, note=None)->None:
#   """
#   Description
#   -----------
#   Dump the DIAS results into a JSON file.

#   Parameters
#   ----------
#     - trajDIASresult (dict): The DIAS results to be dumped.
#     - trajFile (str): The path to the trajectory file.
#     - resultSavePath (str, optional): The path to save the JSON file. Default is "./result.json".
#     - title (str, optional): Title for the JSON file. Default is None.
#     - note (str, optional): Additional note for the JSON file. Default is None.
#   """
#   with open(resultSavePath, "w") as file:
#     json.dump({
#       "title"           : title if title else "",
#       "note"            : note if note else "",
#       "trajectory_file" : basename(trajFile) if isfile(trajFile) else "",
#       "submission_date" : str(datetime.now().strftime("%Y-%m-%d %H:%M")),
#       "result"          : trajDIASresult
#         }, file, indent=4, ensure_ascii=False)
    


