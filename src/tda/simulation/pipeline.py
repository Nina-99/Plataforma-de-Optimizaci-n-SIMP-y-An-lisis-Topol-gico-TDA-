"""Pipeline principal de experimentos TDA con preprocesamiento de diagramas.

Este módulo orquesta la generación de nubes de puntos sintéticas, cálculo
de diagramas de persistencia, preprocesamiento, y exportación de resultados
para validar hipótesis topológicas sobre robustez ante ruido.
"""

import sys
import os
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist

# Add project root to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check for required libraries
try:
    import ripser
    import persim
except ImportError as e:
    raise ImportError("Missing required libraries: ripser or persim. Please install them to run the TDA pipeline.")

# Import metrics and preprocessing modules
from tda.core.topology import betti_numbers
from tda.processing.preprocessing import filter_persistence_diagram, normalize_diagram

# Create data directory if needed
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============== Helper functions ==============

def generate_cloud(shape: str, n_points: int) -> np.ndarray:
    """Genera una nube de puntos 3D sintética para la forma especificada.

    Args:
        shape (str): Tipo de forma a generar. Debe ser 'sphere', 'torus' o 'cube'.
        n_points (int): Número de puntos a generar.

    Returns:
        numpy.ndarray: Nube de puntos 3D de forma (n_points, 3).

    Raises:
        ValueError: Si se proporciona una forma no reconocida.
    """
    np.random.seed(None)  # sembrado por quien llama
    if shape == "sphere":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.arccos(np.random.uniform(-1, 1, n_points))
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    elif shape == "torus":
        R, r = 2.0, 1.0
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        return np.column_stack([x, y, z])
    elif shape == "cube":
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            face = np.random.randint(0, 6)
            uv = np.random.uniform(0, 1, 2)
            if face == 0:
                points[i] = [0, uv[0], uv[1]]
            elif face == 1:
                points[i] = [1, uv[0], uv[1]]
            elif face == 2:
                points[i] = [uv[0], 0, uv[1]]
            elif face == 3:
                points[i] = [uv[0], 1, uv[1]]
            elif face == 4:
                points[i] = [uv[0], uv[1], 0]
            else:
                points[i] = [uv[0], uv[1], 1]
        return points
    else:
        raise ValueError(f"Unknown shape: {shape}. Choose from 'sphere', 'torus', 'cube'.")


def compute_diameter(points: np.ndarray) -> float:
    """Calcula la distancia máxima entre pares en una nube de puntos.

    Args:
        points (numpy.ndarray): Coordenadas de la nube de puntos.

    Returns:
        float: Distancia máxima entre pares (diámetro).
    """
    return float(pdist(points).max())


def add_gaussian_noise(points: np.ndarray, noise_std: float) -> np.ndarray:
    """Agrega ruido gaussiano escalado por noise_std (fracción del diámetro).

    Args:
        points (numpy.ndarray): Coordenadas originales de la nube de puntos.
        noise_std (float): Desviación estándar del ruido como fracción del diámetro.

    Returns:
        numpy.ndarray: Coordenadas de la nube de puntos con ruido.
    """
    diam = compute_diameter(points)
    noise = np.random.normal(loc=0.0, scale=noise_std * diam, size=points.shape)
    return points + noise


def compute_persistence_diagram(points: np.ndarray, maxdim: int = 1) -> list:
    """Calcula el diagrama de persistencia usando ripser, filtrando intervalos inválidos.

    Args:
        points (numpy.ndarray): Coordenadas de la nube de puntos.
        maxdim (int): Dimensión máxima de homología a calcular. Por defecto 1.

    Returns:
        list: Lista filtrada de diagramas de persistencia para cada dimensión.
    """
    result = ripser.ripser(points, maxdim=maxdim)
    dgms = result['dgms']

    filtered_dgms = []
    for dim_dgm in dgms:
        valid_intervals = []
        for birth, death in dim_dgm:
            if np.isfinite(birth) and np.isfinite(death):
                adjusted_birth = max(birth, 1e-8)
                adjusted_death = max(death, adjusted_birth + 1e-8)
                valid_intervals.append([adjusted_birth, adjusted_death])
        filtered_dgms.append(valid_intervals)

    return filtered_dgms


def compute_wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Wasserstein entre dos diagramas.

    Args:
        dgm1 (numpy.ndarray): Primer diagrama de persistencia.
        dgm2 (numpy.ndarray): Segundo diagrama de persistencia.

    Returns:
        float: Distancia de Wasserstein.
    """
    return float(persim.wasserstein(dgm1, dgm2))


def compute_bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Calcula la distancia de Bottleneck entre dos diagramas.

    Args:
        dgm1 (numpy.ndarray): Primer diagrama de persistencia.
        dgm2 (numpy.ndarray): Segundo diagrama de persistencia.

    Returns:
        float: Distancia de Bottleneck.
    """
    return float(persim.bottleneck(dgm1, dgm2))


# ============== Experiment ==============

def run_tda_experiment(shape: str, noise_levels: list[float] = [0.10, 0.15, 0.20], 
                       n_rep: int = 100, n_points: int = 200, seed: int = 42) -> dict:
    """Ejecuta el pipeline TDA con preprocesamiento.

    Procesa el pipeline:
    * Filtrar diagramas de persistencia por umbral
    * Normalizar por diámetro

    Args:
        shape (str): Forma a generar ('sphere', 'torus', 'cube').
        noise_levels (list[float]): Lista de desviaciones estándar de ruido. Por defecto [0.10, 0.15, 0.20].
        n_rep (int): Número de repeticiones del experimento. Por defecto 100.
        n_points (int): Número de puntos por nube. Por defecto 200.
        seed (int): Semilla para reproducibilidad. Por defecto 42.

    Returns:
        dict: Estadísticas del experimento para cada nivel de ruido.
    """
    np.random.seed(seed)
    results = {}
    threshold = 1.5  # persistence_threshold consistent with SIMP

    for noise in noise_levels:
        betti0_list = []
        betti1_list = []
        wasserstein_list = []
        bottleneck_list = []

        for rep in range(n_rep):
            # Nube de puntos limpia
            clean_pts = generate_cloud(shape, n_points)
            clean_dgms = compute_persistence_diagram(clean_pts)

            # Preprocesar diagrama limpio
            clean_filtered = filter_persistence_diagram(clean_dgms, threshold)
            clean_diameter = compute_diameter(clean_pts)
            clean_normalized = normalize_diagram(clean_filtered, clean_diameter)
            clean_arr = np.vstack(clean_normalized) if any(clean_normalized) else np.empty((0, 2))

            # Nube de puntos con ruido
            noisy_pts = add_gaussian_noise(clean_pts, noise)
            noisy_dgms = compute_persistence_diagram(noisy_pts)
            noisy_filtered = filter_persistence_diagram(noisy_dgms, threshold)
            noisy_diameter = compute_diameter(noisy_pts)
            noisy_normalized = normalize_diagram(noisy_filtered, noisy_diameter)
            noisy_arr = np.vstack(noisy_normalized) if any(noisy_normalized) else np.empty((0, 2))

            # Números de Betti
            betti0_noisy, betti1_noisy = betti_numbers(np.hstack([noisy_arr, np.zeros((noisy_arr.shape[0], 1))])) if noisy_arr.shape[0] > 0 else (0, 0)
            betti0_list.append(betti0_noisy)
            betti1_list.append(betti1_noisy)

            # Distancias
            if clean_arr.shape[0] > 0 and noisy_arr.shape[0] > 0:
                try:
                    w_dist = compute_wasserstein_distance(clean_arr, noisy_arr)
                    bn_dist = compute_bottleneck_distance(clean_arr, noisy_arr)
                except Exception:
                    w_dist = 0.0
                    bn_dist = 0.0
            else:
                w_dist = 0.0
                bn_dist = 0.0

            wasserstein_list.append(w_dist)
            bottleneck_list.append(bn_dist)

        # Estadísticas
        results[noise] = {
            'betti0_mean': float(np.mean(betti0_list)),
            'betti0_std': float(np.std(betti0_list)),
            'betti1_mean': float(np.mean(betti1_list)),
            'betti1_std': float(np.std(betti1_list)),
            'wasserstein_mean': float(np.mean(wasserstein_list)),
            'wasserstein_std': float(np.std(wasserstein_list)),
            'bottleneck_mean': float(np.mean(bottleneck_list)),
            'bottleneck_std': float(np.std(bottleneck_list)),
        }

    return results


# ============== Output ==============

def _save_results_to_csv(results: dict, shape: str):
    """Guarda los resultados del experimento en un archivo CSV.

    Args:
        results (dict): Diccionario que mapea niveles de ruido a estadísticas.
        shape (str): Nombre de la forma.

    Returns:
        None
    """
    import csv
    rows = []
    for noise_level, stats in results.items():
        rows.append({
            'noise_level': noise_level,
            'betti0_mean': stats['betti0_mean'],
            'betti0_std': stats['betti0_std'],
            'betti1_mean': stats['betti1_mean'],
            'betti1_std': stats['betti1_std'],
            'wasserstein_mean': stats['wasserstein_mean'],
            'wasserstein_std': stats['wasserstein_std'],
            'bottleneck_mean': stats['bottleneck_mean'],
            'bottleneck_std': stats['bottleneck_std'],
        })
    csv_path = DATA_DIR / f"tda_{shape}_results.csv"
    fieldnames = ['noise_level', 'betti0_mean', 'betti0_std', 'betti1_mean', 'betti1_std',
                  'wasserstein_mean', 'wasserstein_std', 'bottleneck_mean', 'bottleneck_std']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {csv_path}")


def main():
    """Función principal de ejecución del script de experimentos del pipeline TDA.

    Returns:
        None
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run TDA pipeline experiments on synthetic point clouds.")
    parser.add_argument('--shape', type=str, required=True, choices=['sphere', 'torus', 'cube'],
                        help='Shape to generate point cloud from.')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.10, 0.15, 0.20],
                        help='List of noise levels (as fraction of diameter).')
    parser.add_argument('--n_rep', type=int, default=100,
                        help='Number of repetitions per noise level.')
    parser.add_argument('--n_points', type=int, default=200,
                        help='Number of points per cloud.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    print(f"Running TDA experiment on {args.shape} with {args.n_points} points, {args.n_rep} reps, noise levels {args.noise_levels}")
    results = run_tda_experiment(
        shape=args.shape,
        noise_levels=args.noise_levels,
        n_rep=args.n_rep,
        n_points=args.n_points,
        seed=args.seed
    )
    for noise, stats in results.items():
        print(f"Noise {noise:.2f}: beta0={stats['betti0_mean']:.2f}±{stats['betti0_std']:.2f}, "
              f"beta1={stats['betti1_mean']:.2f}±{stats['betti1_std']:.2f}, "
              f"W={stats['wasserstein_mean']:.4f}±{stats['wasserstein_std']:.4f}, "
              f"B={stats['bottleneck_mean']:.4f}±{stats['bottleneck_std']:.4f}")
    _save_results_to_csv(results, args.shape)


if __name__ == "__main__":
    main()