import numpy as np
from fonction_activation import sigmoid, relu

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

def linear_forward_propagation(A_precedent, W, b, fct_activation) :
    """
    Argument :
    A_precedent -- activations de la couche précédente (ou les entrées du réseau)
    W -- poids de la couche courante
    b -- biais de la couche courante
    fct_activation -- fonction d'activation à utiliser ("sigmoid" ou "relu")

    Returns :
    A -- activations de la couche courante
    cache -- dictionnaire contenant "A_precedent", "W", "b" et "Z" pour la rétropropagation
    """

    Z = np.dot(W, A_precedent) + b

    if fct_activation.lower() == "sigmoid" :
        A = sigmoid(Z)
    elif fct_activation.lower() == "relu" :
        A = relu(Z)

    cache = {
        "A_precedent": A_precedent,
        "W": W,
        "b": b,
        "Z": Z,
    }

    return A, cache

def forward_propagation(X, parametres) :
    """
    Argument :
    X -- données d'entrées
    parametres -- dictionnaire comportant les parametres init / pre train

    Returns :
    Y_pred -- activations de la couche outputde sortie
    caches -- dictionnaire contenant "A_precedent", "W", "b" et "Z" pour la rétropropagation
    """
    caches = []
    L = len(parametres) // 2
    A = X

    for couche in range (1, L) :
        A_precedent = A
        A, cache = linear_forward_propagation(A_precedent, parametres[f"W{couche}"], parametres[f"b{couche}"], "relu")
        caches.append(cache)

    Y_pred, cache = linear_forward_propagation(A, parametres[f"W{L}"], parametres[f"b{L}"], "sigmoid")
    caches.append(cache)

    return Y_pred, caches