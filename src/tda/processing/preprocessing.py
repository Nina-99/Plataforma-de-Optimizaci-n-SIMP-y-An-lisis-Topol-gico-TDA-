"""Utilidades de preprocesamiento para diagramas de persistencia en el pipeline TDA.

Proporciona funciones para filtrar intervalos de persistencia cortos, normalizar
coordenadas de nacimiento/muerte y calcular histogramas de vida de persistencia.
"""

from typing import List
import numpy as np


def filter_persistence_diagram(dgm: List[List[float]], threshold: float) -> List[List[float]]:
    """Elimina intervalos de persistencia más cortos que el umbral.

    Args:
        dgm (List[List[float]]): Diagrama de persistencia de forma (n_pairs, 2).
        threshold (float): Longitud mínima de persistencia para retener un punto.

    Returns:
        List[List[float]]: Diagrama después de eliminar intervalos de vida corta.

    Examples:
        >>> dgm = [[[0.1, 1.2], [0.5, 0.6]]]
        >>> filter_persistence_diagram(dgm, 0.5)
        [[[0.1, 1.2]]]
    """
    filtered = []
    for dim_dgm in dgm:
        kept = []
        for birth, death in dim_dgm:
            persistence = death - birth
            if persistence >= threshold:
                kept.append([birth, death])
        filtered.append(kept)
    return filtered


def normalize_diagram(dgm: List[List[float]], diameter: float) -> List[List[float]]:
    """Normaliza un diagrama de persistencia escalando coordenadas de nacimiento/muerte.

    Args:
        dgm (List[List[float]]): Diagrama de persistencia de forma (n_pairs, 2).
        diameter (float): Diámetro de la nube de puntos (distancia máxima entre pares).

    Returns:
        List[List[float]]: Diagrama escalado.

    Examples:
        >>> dgm = [[[0.5, 1.5]]]
        >>> normalize_diagram(dgm, 2.0)
        [[[0.25, 0.75]]]
    """
    normalized = []
    for dim_dgm in dgm:
        norm_dim = []
        for birth, death in dim_dgm:
            # Evita división por cero
            factor = 1.0 / max(diameter, 1e-8)
            norm_birth = birth * factor
            norm_death = death * factor
            norm_dim.append([norm_birth, norm_death])
        normalized.append(norm_dim)
    return normalized


def get_persistence_histogram(dgm: List[List[float]], bins: int = 10) -> np.ndarray:
    """Calcula un histograma de las vidas de persistencia.

    Args:
        dgm (List[List[float]]): Diagrama de persistencia (n_pairs, 2).
        bins (int): Número de intervalos (bins) para el histograma. Por defecto 10.

    Returns:
        numpy.ndarray: Arreglo 1D que representa la distribución de las vidas.

    Examples:
        >>> import numpy as np
        >>> dgm = np.array([[1.2, 0.2], [0.8, 0.4]])
        >>> get_persistence_histogram(dgm, bins=3)
        array([1., 0., 1.])
    """
    # Aplanar todas las vidas
    lifetimes = []
    for death, birth in dgm:  # Nota: persim a veces devuelve [death, birth]
        if len(death.shape) > 1:  # Manejar caso de matriz
            for b, d in death:
                lifetimes.append(d - b)
        else:
            lifetimes.append(death - birth)

    if not lifetimes:
        return np.zeros(bins)

    # Determinar rango del histograma basado en la vida máxima
    max_life = max(lifetimes)
    if max_life == 0:
        max_life = 1.0

    hist, _ = np.histogram(lifetimes, bins=bins, range=(0, max_life))
    return hist