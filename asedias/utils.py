from datetime import datetime
import json
from os.path import basename, isfile, exists
import re
from io import StringIO
import plotly.colors as pc
import plotly.graph_objects as go
from itertools import cycle
import numpy as np
from asedias.data import atomic_number2element_symbol, covalent_radii, atomic_symbols2hex
import ase
from scipy.spatial.distance import pdist, squareform
import warnings 
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union
from ase.units import mol, kcal,  kJ, Hartree, eV
import pandas as pd


def is_ipython():
  try:
    __IPYTHON__
    return True
  except NameError:
    return False
  
if is_ipython:
    from IPython.display import display



def get_adjacency_matrix(atoms:ase.Atoms, covalent_radius_percent:float=108.)->np.ndarray:
    """
    Get adjacency matrix from ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object representing the molecule.

    covalent_radius_percent : float, optional, default=108.0
        The percentage of the standard covalent radii to use for determining bond distances.

    Returns
    -------
    np.ndarray
        Adjacency matrix representing bond connections between atoms.
    """
    def _covalent_radii(element: str, percent: float):
        """resize covalent radius"""
        radius = covalent_radii[element]
        radius *= (percent / 100)
        return radius
    # Get atomic coordinates and symbols
    coordinates = atoms.positions  # (N, 3)
    symbols = atoms.get_chemical_symbols()  # (N, )

    # Calculate interatomic distance (L2 norm) matrix
    L2_matrix = squareform(pdist(coordinates, 'euclidean'))  # (N, N)

    # Calculate sum of atomic radii matrix
    radii_vector = np.array([_covalent_radii(symbol, covalent_radius_percent) for symbol in symbols])  # (N, )
    radii_sum_matrix = np.add.outer(radii_vector, radii_vector)  # (N, N)

    # Calculate adjacency (bond) matrix
    adjacency_matrix = np.array(L2_matrix <= radii_sum_matrix).astype(int)  # (N, N)
    np.fill_diagonal(adjacency_matrix, 0)  # Diagonal means self-bonding is not allowed

    return adjacency_matrix


def atoms2rdkit_mol(atoms:ase.Atoms, covalent_radius_percent:float=108.)->Chem.Mol:
    """
    convert ase.Atoms to 2D rdkit.Chem.Mol
    """
    # get adjacency matrix
    adjacency_matrix = get_adjacency_matrix(
        atoms=atoms, 
        covalent_radius_percent=covalent_radius_percent
        )

    # define rdkit mol
    mol = Chem.RWMol()
    for atom in atoms:
        rdkit_atom = Chem.Atom(int(atom.number))
        mol.AddAtom(rdkit_atom)
    
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            if adjacency_matrix[i, j] == 1:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
    
    # build rdkit mol
    mol = mol.GetMol()
    
    # compute xy coordinate
    AllChem.Compute2DCoords(mol)
    
    return mol


def autofrag(atoms_list: list, covalent_radius_percent: float = 108.):
    """
    Parameters
    ----------
    atoms_list : list of ase.Atoms
        A list of ase.Atoms objects, each representing a molecule.

    covalent_radius_percent : float, optional, default=108.0
        The percentage of the standard covalent radii to use for determining bond distances.

    Returns
    -------

    """
    def find_fragments(adjacency_matrix):
        """Find connected fragments using adjacency matrix."""
        n_atoms = len(adjacency_matrix)
        visited = np.zeros(n_atoms, dtype=bool)
        fragments = []

        def dfs(atom_idx, fragment):
            """Depth-first search to find all atoms connected to the given atom_idx."""
            visited[atom_idx] = True
            fragment.append(atom_idx)
            for neighbor_idx in range(n_atoms):
                if adjacency_matrix[atom_idx, neighbor_idx] == 1 and not visited[neighbor_idx]:
                    dfs(neighbor_idx, fragment)

        for atom_idx in range(n_atoms):
            if not visited[atom_idx]:
                fragment = []
                dfs(atom_idx, fragment)
                fragments.append(sorted(fragment))

        return fragments

    fragmentations = []

    for atoms in atoms_list:
        # get adjacency matrix
        adjacency_matrix = get_adjacency_matrix(atoms=atoms, covalent_radius_percent=covalent_radius_percent)
        np.fill_diagonal(adjacency_matrix, 0)  # Diagonal means self-bonding is not allowed

        # Track connected fragments
        fragments = find_fragments(adjacency_matrix)

        # Store results
        if len(fragments) > 1 and fragments not in fragmentations:
            fragmentations.append(fragments)
            
    # empty list : no fragmentation detected
    if not fragmentations:
        warnings.warn("No fragmentations found.", category=UserWarning)

    return fragmentations


def animation(images:list[ase.Atoms], frag_indices:list=None, colorby:str="fragment",
                   cmap:str=None, covalent_radius_percent:float=108., **kwargs):
    """
    Visualization of molecular structures in 3D using Plotly.

    Parameters
    ----------


    Additional Keyword Arguments
    ----------------------------
    - alpha_atoms : float
        Opacity of atoms. Default: 0.55
    - alpha_bonds : float
        Opacity of bonds. Default: 0.55
    - atom_scaler : float
        Scaling factor for atom visualization. Default: 20
    - bond_scaler : float
        Scaling factor for bond visualization. Default: 10,000
    - legend : bool
        Show legend in the visualization. Default: True
    - scale_box : bool
        Show scale box in the visualization. Default: True
    - unit_bar : bool
        Show unit bar in the visualization. Default: True
    - camera_projection : str
        ['orthographic', 'perspective'] Defaut: 'orthographic'
    - template : str
        ['plotly', 'plotly_dark', etc. ] Defaut: 'plotly_dark'

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

    def _plot_scale_box(box_half_length:float, unit_bar:bool=True):
        """
        plot scale box
        """
        hl = box_half_length # box half-length

        # Box vertices
        x_coords = [-hl, hl, hl, -hl, -hl, hl, hl, -hl]
        y_coords = [-hl, -hl, hl, hl, -hl, -hl, hl, hl]
        z_coords = [-hl, -hl, -hl, -hl, hl, hl, hl, hl]

        # Define edges by vertex indices
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]

        x_box_lines = []
        y_box_lines = []
        z_box_lines = []

        for edge in edges:
            for vertex in edge:
                x_box_lines.append(x_coords[vertex])
                y_box_lines.append(y_coords[vertex])
                z_box_lines.append(z_coords[vertex])
            x_box_lines.append(None)
            y_box_lines.append(None)
            z_box_lines.append(None)

        box_plots = [
            # scaling bar
            go.Scatter3d(
                x=[hl, hl],
                y=[-hl, -hl+1],
                z=[-hl, -hl],
                mode='lines+text',
                line=dict(color='red', width=13),
                text=['1 Å'],
                textposition='top center',
                textfont=dict(size=12, color='red'),
                showlegend=False,
                name=f'Bar',
                legendgroup=f'Box',
                hoverinfo='text',
                hovertext=f'Bar'
            ) if unit_bar else go.Scatter3d(
                x=[hl, hl],
                y=[-hl, +hl],
                z=[-hl, -hl],
                mode='lines+text',
                line=dict(color='red', width=13),
                text=[f'{np.round(hl*2, 1)} Å'],
                textposition='top center',
                textfont=dict(size=12, color='red'),
                showlegend=False,
                name=f'Bar',
                legendgroup=f'Box',
                hoverinfo='text',
                hovertext=f'Bar'            
                ),
            # box lines
            go.Scatter3d(
                x=x_box_lines,
                y=y_box_lines,
                z=z_box_lines,
                mode='lines',
                line=dict(color='grey', width=4),
                showlegend=True,
                name=f'Box',
                legendgroup=f'Box',
                hoverinfo='text',
                hovertext=f'Box'
            )
            ]
        return box_plots

    # Default visualization parameters
    alpha_atoms = kwargs.get("alpha_atoms", 1)
    alpha_bonds = kwargs.get("alpha_bonds", 0.7)
    atom_scaler = kwargs.get("atom_scaler", 300)
    bond_scaler = kwargs.get("bond_scaler", 8200000)
    scale_box = kwargs.get("scale_box", True)
    legend = kwargs.get("legend", True)
    unit_bar = kwargs.get("unit_bar", True)
    camera_projection = kwargs.get("camera_projection", "orthographic")
    template = kwargs.get("template", "plotly_dark")

    # Make sure each position is centered
    for atoms in images:
        atoms.center()

    # Calculate global ranges for all molecular structures
    all_positions = np.vstack([atoms.positions for atoms in images])
    range_array = np.vstack([[np.min(all_positions[:, i]) for i in range(3)], [np.max(all_positions[:, i]) for i in range(3)]])
    x_range, y_range, z_range = range_array[:, 0], range_array[:, 1], range_array[:, 2]
    padding = 0.1
    if scale_box:
        half_box_length = np.max(np.abs(range_array)) * 1.3
        max_range = [- half_box_length - padding, half_box_length + padding]
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

            # Add bonds
            bond_thickness = np.maximum(np.log10(bond_scaler / num_of_atoms) * 2, 1) # bond thickness
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
            
            # add scale_box plots if scale_box == True
            if scale_box:
                frame_data.extend(
                    _plot_scale_box(box_half_length=half_box_length, unit_bar=unit_bar)
                    )

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
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=f'Fragment {frag_idx + 1}'
                ))

                # Add bonds
                bond_x, bond_y, bond_z = [], [], []
                bond_thickness = np.maximum(np.log10(bond_scaler / num_of_atoms) * 2, 1) # bond thickness
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

            # add scale_box plots if scale_box == True
            if scale_box:
                frame_data.extend(
                    _plot_scale_box(box_half_length=half_box_length, unit_bar=unit_bar)
                    )

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
            xaxis=dict(range=max_range if scale_box else x_range, visible=False),
            yaxis=dict(range=max_range if scale_box else y_range, visible=False),
            zaxis=dict(range=max_range if scale_box else z_range, visible=False),
            aspectmode='manual',
            camera_projection=dict(type=camera_projection), # orthograpic perspective
            aspectratio=dict(x=1, y=1, z=1),
        ),
        showlegend=True if legend else False,
        template=template,
    )
    
    fig.show()


def fragment_selector(mol:Chem.Mol, **kwargs):
    """
    Plot RDKit 2D mol using Plotly.
    """
    # parsing kwargs
    plot_bgcolor = kwargs.get('plot_bgcolor', 'white')

    # get 2D coordinates
    coords = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    x = coords[:, 0]
    y = coords[:, 1]
    
    # trace container 
    data = []

    # add bonds (line indicates interatomic connectivity not a bond order)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x_line = [x[i], x[j]]
        y_line = [y[i], y[j]]
        data.append(go.Scatter(
            x=x_line, 
            y=y_line, 
            mode='lines',
            opacity=0.7,
            line=dict(
                width=3, 
                color='#3f3f3f'
                )
            )
         )

    # add atoms
    data.append(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(
            size=23, 
            color='white', 
            #opacity=0.7
            ),
        textfont=dict(
            size=20,
            color='black',
        ),
        text=symbols,
        textposition="middle center",
        hoverinfo="none"
        )
    )     

    fig = go.FigureWidget(data=data)

    # plot settings
    fig.update_layout(
        width=800, height=600,
        showlegend=False,
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        template='plotly_white',
        plot_bgcolor=plot_bgcolor,
        dragmode='lasso'
    )
    # callback function
    def selected_indices(trace, points, selector):
        if points.point_inds:
            selected_points = points.point_inds
            print(selected_points)
        else:
          print('No atoms are selected')

    fig.data[-1].on_selection(selected_indices)
          
    display(fig)
    print()


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


def json_dump(trajDIASresult:dict, runtime:float, job_name:str, metadata:Union[str, dict])->None:
  """
  Dump the DIAS results into a JSON file.

  Parameters
  ----------
    
  """
  _savepath = f'./{job_name}.json'
  with open(_savepath, "w") as file:
    json.dump({
      "METADATA": metadata,
      "DATE": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "RUNTIME(min)": runtime,
      "RESULT": trajDIASresult
      }, 
      fp=file, 
      indent=4, 
      ensure_ascii=False
      )
    

def relative_values(energy_series:Union[list, tuple], relative_index:Union[str, int]="min")->tuple:
    """
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
    _energy_series = np.array(energy_series)
    if relative_index == "min":
        _energy_series -= np.min(_energy_series)
    elif isinstance(relative_index, int):
        _energy_series -= _energy_series[relative_index]
    else:
        raise ValueError(f"Invalid relative_index value : {relative_index}")
    return _energy_series.tolist()


def DIASparser(resultDict:Union[dict, str], frag_type:str, 
               energy_type:str, relative_idx:Union[str,int]=None, 
               unit_conversion:Union[float, str]='eV'):
    """
    Parses DIAS results from a dictionary based on the specified fragment type, energy type, relative index, and unit.

    Parameters
    ----------
    - resultDict (dict|str) : The dictionary containing DIAS results.
    - frag_type (str) : The type of fragment or molecule to parse (`molecule` or `fragment names`).
    - energy_type (str) : The type of energy to parse ("total", "interaction", or "distortion").
    - relative_idx (str|int) : The index of the reference energy value for computing relative values.
    - unit_conversion (str|float) : Default unit is eV.
        - ["KJ/MOL", "HARTREE", "KCAL/MOL", "EV"]
    Returns
    -------
    - energies(tuple) : A tuple of parsed energy values.
    """
    eV2unitFactors = {
       "KJ/MOL"  :  1/kJ * mol, 
       "HARTREE" :  1/Hartree, 
       "KCAL/MOL":  1/kcal * mol, 
       "EV"      :  eV
       }
    
    # convert to str to float
    if isinstance(unit_conversion, str):
        UNIT_CONVERSION = unit_conversion.upper() 
        assert UNIT_CONVERSION in eV2unitFactors.keys(), 'Invalid unit'
        unit_conversion = eV2unitFactors[UNIT_CONVERSION]

    # read json format resultDict file
    if isinstance(resultDict, str):
        assert exists(resultDict), "The resultDict file does not exist."
        with open(resultDict, 'r') as file:
            resultDict = json.load(file)

    # get RESULT data from the json
    if resultDict.get('RESULT'):
        result = resultDict['RESULT']
    elif resultDict.get('0'):
        result = resultDict
    else:
        raise ValueError('asedias cannot parse the result from the json file')
  
    # get frag_names list
    frag_names = set(result['0'].keys()) 
    frag_names -= {'molecule', 'success'}

    # validate parameter values
    assert frag_type in {'molecule', *frag_names}, f"Invalid name, `{frag_type}`"
    assert energy_type in ({'distortion'} if frag_type in frag_names else {'distortion', 'interaction', 'total'}), 'Invalid energy_type' 

    energySeries = np.array([pointResult[frag_type][energy_type] for pointResult in result.values()]) * unit_conversion
    
    # get relative values
    if relative_idx != None:
        return relative_values(energy_series=energySeries, relative_index=relative_idx)
    
    return energySeries


def convert_df(resultDict:dict)->pd.DataFrame:
    """
    Convert asedias json-format result dict to pandas dataframe.
    """
    def _get_value(resultDict:dict, keys:list):
        if len(keys) == 1:
            return resultDict[keys[0]]
        else:
            return _get_value(resultDict[keys[0]], keys[1:])

    data_container = dict()

    for irc_idx in resultDict.keys():
        for frag_key in resultDict[irc_idx].keys():
            if isinstance(resultDict[irc_idx][frag_key], dict):
                for energy_key in resultDict[irc_idx][frag_key].keys():
                    if not data_container.get(f"{frag_key}_{energy_key}"):
                        data_container[f"{frag_key}_{energy_key}"] = list()
                    data_container[f"{frag_key}_{energy_key}"].append(_get_value(resultDict=resultDict, keys=[irc_idx, frag_key, energy_key]))
            else:
                if not data_container.get(f"{frag_key}"):
                    data_container[f"{frag_key}"] = list()
                data_container[f"{frag_key}"].append(_get_value(resultDict=resultDict, keys=[irc_idx, frag_key]))
    
    return pd.DataFrame(data_container)
