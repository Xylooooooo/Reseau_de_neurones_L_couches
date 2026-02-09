import numpy as np

def initialisation_parametres_classique(dimension_couches):
    """
    Arguments:
    dimension_couches -- dimension des couches
    
    returns:
    parametres -- dictionnaire des paramètres initialisés
    """

    np.random.seed(1)
    L = len(dimension_couches)
    parametres = {}

    for couche in range (1, L):
        parametres[f"W{couche}"] = np.random.randn(dimension_couches[couche], dimension_couches[couche - 1]) * 0.01
        parametres[f"b{couche}"] = np.zeros((dimension_couches[couche], 1))
    
    return parametres


def initialisation_parametres_he(dimension_couches):
    """
    Arguments:
    dimension_couches -- dimension des couches
    
    returns:
    parametres -- dictionnaire des paramètres initialisés
    """

    np.random.seed(1)
    L = len(dimension_couches)
    parametres = {}

    for couche in range (1, L):
        parametres[f"W{couche}"] = np.random.randn(dimension_couches[couche], dimension_couches[couche - 1]) * np.sqrt(2 / dimension_couches[couche - 1])
        parametres[f"b{couche}"] = np.zeros((dimension_couches[couche], 1))
    
    return parametres