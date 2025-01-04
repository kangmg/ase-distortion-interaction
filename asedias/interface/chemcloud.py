import warnings
from ase.units import Hartree, Bohr
from io import StringIO
import ase
import ase.io
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

try:
    from chemcloud import CCClient
    import chemcloud
    import qcio
    from qcio import ProgramInput, ProgramOutput, Structure, DualProgramInput, SinglePointResults, OptimizationResults
except ModuleNotFoundError:
    warnings.warn("Chemcloud is not available", category=UserWarning)

class BatchCalculator:
    pass

class CCBatchCalculation(BatchCalculator):
    """
    Dummy calculator for Chemcloud
    """
    implemented_properties = ["energy", "forces", "optimization"]

    def __init__(self, client:chemcloud.CCClient, atoms_list:list[ase.Atoms], spin_list:int, 
                 charge_list:int, basis:str, method:str, batch_size:int=10):
        
        assert len(spin_list) == len(charge_list) == len(atoms_list), 'Size mismatch. (`spin_list`, `charge_list` and `atoms_list`)'
        assert len(atoms_list) <= batch_size, "Num. of images are larger than batch_size"
        self.atoms_list = atoms_list
        self.client = client
        self.charge_list = charge_list
        self.spin_list = spin_list
        self.basis = basis
        self.method = method
        self.results = {}
        self.outputs = None
        self.optimized = False


    def atoms2xyz(self, atoms:ase.Atoms)->str:
        """convert ase.Atoms to xyz-string"""
        output = StringIO()
        ase.io.write(output, atoms, format='xyz')

        xyz_string = output.getvalue()
        return xyz_string
    
    
    def structure2atoms(self, structure:qcio.Structure, atoms:ase.Atoms=None)->ase.Atoms:
        """Update atomic positions or convert to ase.Atoms"""
        if atoms:
            assert atoms.get_atomic_numbers().tolist() == structure.atomc_numbers, 'atomic numbers are different'
            atoms.positions = structure.geometry_angstrom
        else:
            atoms = ase.Atoms(
                numbers=structure.atomic_numbers,
                positions=structure.geometry_angstrom
            )
        return atoms


    def _input_builder(self, atoms:ase.Atoms, charge:int, 
                       spin:int, calctype:str, method:str, 
                       basis:str, **calc_kwargs)->ProgramInput:
        """Prepare qcio input"""
        _xyz_string = self.atoms2xyz(atoms=atoms)
        _structure = Structure.from_xyz(
            xyz_str=_xyz_string, 
            charge=charge, 
            multiplicity=(2 * spin + 1)
            )
        
        if calctype in ['gradient', 'energy']:
            return ProgramInput(
                calctype=calctype,
                structure=_structure,
                model = {"method": method, "basis": basis},
                keywords=calc_kwargs
                )
        elif calctype == 'optimization':
            return DualProgramInput(
                calctype="optimization",
                structure=_structure,
                subprogram_args={"model": {"method": method, "basis": basis}},
                keywords=calc_kwargs,
                subprogram="terachem"
                )
        else:
            raise NotImplementedError(f'`{calctype}` is not supported yet')
    


    def parse_potential_energy_list(self, outputList:list[ProgramOutput]):
        """
        Returns energy list parsed from batch calculation results.

        Terachem out unit : Hartree
        Converted unit    : eV
        """
        def _get_energy_result(output:ProgramOutput):
            """Returns unit converted energy"""
            assert output.success, 'Calculation faild'
            if isinstance(output.results, SinglePointResults):
                energy = output.results.energy # in Hartree
            elif isinstance(output.results, OptimizationResults):
                energy = output.results.energies[-1] # in Hartree
            else:
                raise TypeError("The `output` should be either `qcio.ProgramOutput` or `qcio.OptimizationOutput`")
            # convert unit
            energy *= Hartree # in eV
            return energy
        
        return list(_get_energy_result(output) for output in self.outputs)

        
    def parse_forces_list(self, outputList:list[ProgramOutput]):
        """
        Returns forces list parsed from batch calculation results.
        
        Terachem out unit : Ha/Bohr
        Coverted unit     : eV/A
        """
        def _get_force_result(output:ProgramOutput):
            """Returns unit converted forces"""
            assert output.success, 'Calculation faild'
            if isinstance(output.results, SinglePointResults):
                gradient = output.results.gradient # in Hartree/Bohr
            elif isinstance(output.results, OptimizationResults):
                gradient = output.results.trajectory[-1].results.gradient # in Hartree/Bohr
            else:
                raise TypeError("The `output` should be either `qcio.ProgramOutput` or `qcio.OptimizationOutput`")
            # convert unit
            assert gradient.any() if isinstance(gradient, np.ndarray) else gradient, "output does not include the gradient data"
            gradient *= (Hartree / Bohr) # in eV/A
            return -gradient
        try:
            return list(_get_force_result(output) for output in self.outputs)
        except AssertionError:
            raise AttributeError('output has attributes to gradient')


    def call_batch_api(self, calctype, **calc_kwargs):
        """submit batch job on cloud"""
        systems = zip(self.atoms_list, self.charge_list, self.spin_list)
        _input_list = list(
            self._input_builder(
            atoms=atoms, 
            charge=charge, 
            spin=spin,
            calctype=calctype,
            method=self.method, 
            basis=self.basis, 
            **calc_kwargs
            ) for (atoms, charge, spin) in systems
        )

        # Perform the calculation using the ChemCloud client
        future_results = self.client.compute(
            program="terachem",
            inp_obj=_input_list,
            collect_stdout=False,
            collect_files=True
            )
        
        outputs = future_results.get()
        self.outputs = outputs

        success_list = list(output.success for output in outputs)
        if (not all(success_list)) or (len(success_list) == 0):
            raise ValueError('Calculation Faild')
    
    
    def call_batch_optimization_api(self, fmax:float=None, **calc_kwargs):
        """batch optimization
        
        Parameters
        ----------
        fmax (float):
            ase optimizer param (eV/A)
        """
        if fmax:
            convergence_gmax = fmax * (Bohr / Hartree) # eV/A to Ha/Bohr
            calc_kwargs = {
                **calc_kwargs,
                'convergence_gmax': convergence_gmax
            }

        systems = zip(self.atoms_list, self.charge_list, self.spin_list)
        _input_list = list(
            self._input_builder(
            atoms=atoms, 
            charge=charge, 
            spin=spin,
            calctype='optimization',
            method=self.method, 
            basis=self.basis, 
            **calc_kwargs
            ) for (atoms, charge, spin) in systems
        )

        # Perform the calculation using the ChemCloud client
        future_results = self.client.compute(
            program="geometric",
            inp_obj=_input_list,
            #collect_stdout=False,
            collect_files=True
            )
        
        outputs = future_results.get()
        
        success_list = list(output.success for output in outputs)
        if (not all(success_list)) or (len(success_list) == 0):
            raise ValueError('Calculation Faild')

        self.outputs = outputs
        self.optimized = True
        # clear previous calculation results
        self.results = {}

        # update optimized geometry 
        structures_list = list(output.results.final_structure for output in self.outputs)
        self.atoms_list = list(
            self.structure2atoms(
                structure=structure, atoms=atoms
                ) for (structure, atoms) in zip(structures_list, self.atoms_list)
            )
        

    def get_potential_energy_list(self):
        """get pre-optimized potential energy list"""
        if self.results.get('energy'):
            return self.results['energy']
        
        if self.outputs:
            self.results['energy'] = self.parse_potential_energy_list(self.outputs)
            return self.results['energy']
        
        # call api
        self.call_batch_api(calctype='energy')
        self.results['energy'] = self.parse_potential_energy_list(self.outputs)
        return self.results['energy']
        

    def get_forces_list(self):
        """get forces list"""
        if self.results.get('forces'):
            return self.results['forces']
        
        if self.outputs:
            try:
                self.results['forces'] = self.parse_forces_list(self.outputs)
                return self.results['forces']
            except AttributeError:
                pass

        # call api 
        self.call_batch_api(calctype='gradient')
        self.results['forces'] = self.parse_forces_list(self.outputs)    
        return self.results['forces']


    def get_optimized_energy_list(self, fmax:float=0.05):
        """get optimized potential energy list"""
        self.call_batch_optimization_api(fmax=fmax)
        self.results['energy'] = self.parse_potential_energy_list(self.outputs)
        return self.results['energy']



class CCCalculator(Calculator):
    """
    ASE Calculator for API interfacing with Chemcloud
    """
    implemented_properties = ["energy", "forces", "charges"]

    def __init__(self, client:chemcloud.CCClient, spin:int, 
                 charge:int, basis:str, method:str, calc_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.method = method
        self.calc_kwargs = calc_kwargs
        self.output = None


    def atoms2xyz(self, atoms:ase.Atoms)->str:
        """convert ase.Atoms to xyz-string"""
        output = StringIO()
        ase.io.write(output, atoms, format='xyz')

        xyz_string = output.getvalue()
        return xyz_string
    

    def _input_builder(self, atoms:ase.Atoms, charge:int, 
                       spin:int, calctype:str, method:str, 
                       basis:str, **calc_kwargs)->ProgramInput:
        """Prepare qcio input"""
        _xyz_string = self.atoms2xyz(atoms=atoms)
        _structure = Structure.from_xyz(
            xyz_str=_xyz_string, 
            charge=charge, 
            multiplicity=(2 * spin + 1)
            )
        return ProgramInput(
            calctype=calctype,
            structure=_structure,
            model = {"method": method, "basis": basis},
            keywords=calc_kwargs
            )


    def get_force_result(self, output:ProgramOutput):
        """Returns force parsed from the terachem output
        """
        assert output.success, 'Calculation faild'
        gradient = output.results.gradient # in Hartree/Bohr
        gradient *= (Hartree / Bohr) # in eV/A
        return -gradient


    def get_energy_result(self, output:ProgramOutput):
        """Returns energy parsed from the terachem output
        """
        assert output.success, 'Calculation faild'
        energy = output.results.energy # in Hartree
        energy *= Hartree
        return energy


    def get_charges_result(self, output:ProgramOutput):
        """parsing charge population result
        """
        charges = list(line.split('\t')[-1] for line in output.results.files['scr.geometry/charge_mull.xls'].strip().split('\n'))
        return np.array(charges).astype(float)


    def call_api(self, calctype):
        """submit job on cloud"""
        _input = self._input_builder(
            atoms=self.atoms, 
            charge=self.charge, 
            spin=self.spin,
            calctype=calctype,
            method=self.method, 
            basis=self.basis, 
            **self.calc_kwargs
            )

        # Perform the calculation using the ChemCloud client
        future_result = self.client.compute(program="terachem",
                                    inp_obj=_input,
                                    collect_stdout=False,
                                    collect_files=True
                                    )
        output = future_result.get()

        self.output = output

        if not output.success:
            raise ValueError('Calculation Faild')
            print(output.traceback)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure to be calculated.
        
        """
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        if system_changes:
            self.results.clear()

        # get result
        if ('energy' in properties) and ('energy' not in self.results):
            output = self.call_api(calctype='energy')

        if ('forces' in properties) and ('forces' not in self.results):
            output = self.call_api(calctype='gradient')
            self.results["forces"] = self.get_force_result(output=self.output)

        self.results["energy"] = self.get_energy_result(output=self.output)
        self.results["charges"] = self.get_charges_result(output=self.output)