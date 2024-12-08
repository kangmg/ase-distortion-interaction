## Recommended ASE Calculators (tested)
- semi-emphirical methods
  - xTB (TBLite Calculator)
  - MOPAC (built-in Calculator)
- ab initio
  - pyscf (pyscf4ase : GPU accelerated Calculator)
- ML-potential
  - XequiNet (delta learning model is better for this task)
  - AIMNet2

![image](https://github.com/user-attachments/assets/ba31838f-ab6b-4386-9330-4e675d7e60af)


## Basic Usage

> Sample reaction path files
  
```python
from ase_dias import load_data

# download file
load_data("wittig.xyz", save_file=True)
load_data("DA.xyz", save_file=True)
load_data("sn2.xyz", save_file=True)
```

> Basic Usage / ML-Potential (e.g. AIMNet2)
```python
from ase_dias import dias_run, AIMNet2Calculator, load_aimnet2

# fragments_params
fp_n3 = {
  "Br-"   : (-1, [2]),
  "CH3+"  : (+1, [1,3,4,5]),
  "Cl-"   : (-1, [6])
}

model = load_aimnet2()
aimnet2_calc = lambda **kwargs : AIMNet2Calculator(model=model, **kwargs)

dias_run(
  calc_wrapper=aimnet2_calc,
  trajFile="sn2.xyz",
  fragments_params=fp_n3,
  )
```

> Basic Usage / Semi-emphirical method (e.g. MOPAC)
```python
from ase_dias import dias_run
from ase.calculators.mopac import MOPAC

# fragments_params
fp_DA = {
    "diene"        : (0, [7,8,11,12,14,20,21,22,23,24,25,26]),
    "dienophile"   : (0, [1,2,3,4,5,6,9,10,13,15,16,17,18,19])
    }

mopac_calc = lambda **kwargs : MOPAC(method='PM7', **kwargs)

dias_run(
  calc_wrapper=mopac_calc,
  trajFile="DA.xyz",
  fragments_params=fp_DA,
  use_spin=False
  )
```

> Advanced Usage / ab initio method with preoptimizer (AIMNet2)
```python
from ase_dias import dias_run, AIMNet2Calculator, load_aimnet2
from pyscf4ase.dft import PySCFCalculator

model = load_aimnet2()

def preoptimizer(**kwargs):
  """AIMNet2 preoptimizer
  """
  charge = kwargs.get("charge", None)
  return AIMNet2Calculator(model=model, charge=charge)

def dft_calc(**kwargs):
  """Reference DFT calculator
  """
  spin = kwargs.get("spin", None)
  charge = kwargs.get("charge", None)
  _parameters = {
      "spin" : spin,
      "charge" : charge,
      "xc"  : 'wb97m-d3bj',
      "basis" : 'def2-tzvpp',
      "device" : 'gpu',
      "verbose" : 0
  }
  return PySCFCalculator(**_parameters)

# fragments_params
fp_DA = {
    "diene"        : (0, [7,8,11,12,14,20,21,22,23,24,25,26]),
    "dienophile"   : (0, [1,2,3,4,5,6,9,10,13,15,16,17,18,19])
    }

# parameters
include_fragments=True                # show fragmets distortion energy
relative_idx=0                        # relative energy reference
unit="eV"                             # energy unit -- HArTrEe harTREE both possible

dias_run(
    calc_wrapper=dft_calc,
    preoptimizer_wrapper=preoptimizer,
    trajFile="DA_sampled.xyz",
    fragments_params=fp_DA,
    resultSavePath='DA.json',
    include_fragments=include_fragments,
    relative_idx=relative_idx,
    unit=unit,
    use_spin=True,
    fmax=0.1
    )
```


<br/>

## How to Install
> ***pip***
- 
  ```shell
  pip install git+https://github.com/kangmg/ase-distortion-interaction.git
  ```

> ***git clone***
- terminal
  ```shell
  ### terminal ###
  git clone https://github.com/kangmg/ase-distortion-interaction.git
  cd ase-distortion-interaction
  pip install .
  ```

