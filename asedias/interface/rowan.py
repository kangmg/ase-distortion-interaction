import warnings

try:
    import rowan
except ModuleNotFoundError:
    warnings.warn("Rowan is not available", category=UserWarning)


class RowanCalculator:
    """
    Dummy calculator for integrate with Rowan
    """
    pass