"""Módulo de funciones topológicas básicas.

Este módulo proporciona implementaciones fundamentales para calcular distancias entre diagramas de persistencia y extraer números de Betti de estructuras topológicas. Las funciones implementadas utilizan las bibliotecas persim y numpy para cálculos topológicos eficientes.

Las funciones principales incluyen:
- Cálculo de distancia de Wasserstein entre diagramas
- Cálculo de distancia de Bottleneck entre diagramas
- Extracción de números de Betti (invariantes topológicos)
"""

import numpy as np
from typing import Tuple

try:
    from persim import wasserstein_distance as persim_wasserstein
    from persim import bottleneck_distance as persim_bottleneck
except ImportError:
    persim_wasserstein = None
    persim_bottleneck = None


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Wasserstein entre dos diagramas de persistencia.

    Esta función mide la similitud topológica entre dos estructuras
    mediante la distancia de Wasserstein, que cuantifica el esfuerzo
    requerido para transformar un diagrama en otro.

    Args:
        dgm1 (np.ndarray): Primer diagrama de persistencia con forma (n, 2)
            donde cada fila representa [nacimiento, muerte].
        dgm2 (np.ndarray): Segundo diagrama de persistencia con forma (m, 2)
            donde cada fila representa [nacimiento, muerte].

    Returns:
        float: La distancia de Wasserstein entre los dos diagramas.

    Raises:
        ImportError: Si la biblioteca persim no está disponible.
        ValueError: Si los arrays de entrada no tienen forma (n, 2) o (m, 2).

    Examples:
        >>> import numpy as np
        >>> dgm1 = np.array([[0.0, 1.0], [1.2, 2.0]])
        >>> dgm2 = np.array([[0.0, 1.1], [1.0, 1.8]])
        >>> # Suponiendo que persim está instalado:
        >>> # distancia = wasserstein_distance(dgm1, dgm2)
    """
    if persim_wasserstein is None:
        raise ImportError("persim library is required for wasserstein_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_wasserstein(dgm1, dgm2))


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Bottleneck entre dos diagramas de persistencia.

    Esta función calcula la distancia de Bottleneck, una métrica más
    conservadora que Wassserstein que mide la diferencia máxima entre
    puntos correspondientes de los diagramas.

    Args:
        dgm1 (np.ndarray): Primer diagrama de persistencia con forma (n, 2)
            donde cada fila representa [nacimiento, muerte].
        dgm2 (np.ndarray): Segundo diagrama de persistencia con forma (m, 2)
            donde cada fila representa [nacimiento, muerte].

    Returns:
        float: La distancia de Bottleneck entre los dos diagramas.

    Raises:
        ImportError: Si la biblioteca persim no está disponible.
        ValueError: Si los arrays de entrada no tienen forma (n, 2) o (m, 2).

    Examples:
        >>> import numpy as np
        >>> dgm1 = np.array([[0.0, 1.0], [1.2, 2.0]])
        >>> dgm2 = np.array([[0.0, 1.1], [1.0, 1.8]])
        >>> # Suponiendo que persim está instalado:
        >>> # distancia = bottleneck_distance(dgm1, dgm2)
    """
    if persim_bottleneck is None:
        raise ImportError("persim library is required for bottleneck_distance")
    if dgm1.ndim != 2 or dgm1.shape[1] != 2:
        raise ValueError("dgm1 must be of shape (n, 2)")
    if dgm2.ndim != 2 or dgm2.shape[1] != 2:
        raise ValueError("dgm2 must be of shape (m, 2)")
    return float(persim_bottleneck(dgm1, dgm2))


def betti_numbers(persistence_diagram: np.ndarray) -> Tuple[int, int]:
    """Extrae los números de Betti (β₀, β₁) de un diagrama de persistencia.

    Los números de Betti son invariantes topológicos que representan:
    - β₀: Número de componentes conexas (H₀)
    - β₁: Número de agujeros de dimensión 1 (H₁)

    Args:
        persistence_diagram (np.ndarray): Diagrama de persistencia con forma (n, 3)
            donde cada fila es [nacimiento, muerte, dimensión]. La dimensión 0
            corresponde a H₀ (componentes conexas), la dimensión 1 a H₁ (agujeros).

    Returns:
        Tuple[int, int]: Una tupla (beta_0, beta_1) donde:
            beta_0: Número de componentes conexas.
            beta_1: Número de agujeros de dimensión 1.

    Raises:
        ValueError: Si el array de entrada no tiene forma (n, 3) o contiene
            dimensiones inválidas (deben ser solo 0 o 1).

    Examples:
        >>> import numpy as np
        >>> dgm = np.array([
        ...     [0.0, 1.0, 0.0],
        ...     [0.0, np.inf, 0.0],
        ...     [0.5, 1.2, 1.0]
        ... ])
        >>> betti_numbers(dgm)
        (2, 1)
    """
    if persistence_diagram.ndim != 2 or persistence_diagram.shape[1] != 3:
        raise ValueError("persistence_diagram must be of shape (n, 3) with [birth, death, dimension]")
    if not np.all(np.isin(persistence_diagram[:, 2], [0, 1])):
        raise ValueError("Dimension column must contain only 0 (H_0) or 1 (H_1)")

    h0 = persistence_diagram[persistence_diagram[:, 2] == 0]
    h1 = persistence_diagram[persistence_diagram[:, 2] == 1]

    # beta_0: número de componentes conexas (muerte finita o infinita)
    beta_0 = int(h0.shape[0])
    # beta_1: número de agujeros de dimensión 1
    beta_1 = int(h1.shape[0])

    return (beta_0, beta_1)