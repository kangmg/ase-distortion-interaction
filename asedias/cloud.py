import warnings

try:
    import chemcloud
    import qcio
except ModuleNotFoundError:
    warnings.warn("Chemcloud is not available", category=UserWarning)

try:
    import rowan
except ModuleNotFoundError:
    warnings.warn("Rowan is not available", category=UserWarning)


class ChemCloudCalculator:
    """
    Dummy calculator for integrate with Chemcloud
    """
    pass

class RowanCalculator:
    """
    Dummy calculator for integrate with Rowan
    """
    pass