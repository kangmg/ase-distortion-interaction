from ase.optimize import BFGS
import ase

class System:
    """
    asedias System class

    """
    def __init__(system:list[ase.Atoms], frag_charges:list, frag_indice:list=None, ):
        pass


class ParameterManager:
    """Package parameter manager class
    
    
    Usage
    -----
    >>> from asedias import ParameterManager
    >>> 
    >>> # show all parameters
    >>> print(ParameterManager.show_parameters())
    >>> 
    >>> # change parameters
    >>> ParameterManager.param_name = 'new value'
    """

    optimizer = BFGS # FIRE, LBFGS

  
    @classmethod
    def default_parameters(cls):
        """
        Show all default parameters
        """
        _params = {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
        try:
            from pprint import pformat
            return pformat(_params, indent=1)
        except ModuleNotFoundError:
            return _params

class aseDIAS:
    """
    
    """
    def __init__():
        pass