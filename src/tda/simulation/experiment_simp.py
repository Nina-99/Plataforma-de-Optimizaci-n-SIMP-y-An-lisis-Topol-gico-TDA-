"""Experimento sin cabeza para validar la hipótesis H.E.2 de optimización SIMP.

Este script ejecuta la optimización de topología SIMP en una viga voladizo
2D, validando que se pueda alcanzar una reducción de compliance ≥ 40% mientras
se mantiene el número de Betti-1 ≤ 2.
"""

import os
import pandas as pd
from tda.optimization.simp_optimizer import SimpTda2DOptimizer


def run_headless_experiment():
    """Ejecuta el experimento de optimización SIMP sin cabeza para validar la Hipótesis H.E.2.

    Inicializa el optimizador, registra el progreso de compliance, realiza
    la optimización de topología, calcula números de Betti-1 y exporta las
    métricas finales a un archivo CSV.

    Returns:
        None
    """
    print("--- Iniciando validación de la Hipótesis Específica H.E.2 (SIMP) ---")
    print("Configuración: Viga voladizo, Malla 60x30, Volumen=0.5, Penalización=3.0\\n")

    # Instanciación del optimizador con los parámetros definidos en la tesis
    opt = SimpTda2DOptimizer(nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=1.5)

    # Callback simple para monitorear el progreso en consola
    def log_progress(loop, xPhys, c, reduccion, max_iter):
        """Función callback para registrar el progreso de la optimización.

        Args:
            loop (int): Iteración actual.
            xPhys (numpy.ndarray): Distribución de densidad física.
            c (float): Valor actual de compliance.
            reduccion (float): Porcentaje de reducción de compliance.
            max_iter (int): Presupuesto máximo de iteraciones.

        Returns:
            None
        """
        if loop % 10 == 0:
            print(f"Iteración {loop}: Compliance = {c:.4f} | Reducción = {reduccion:.2f}%")

    # Ejecución de la optimización
    x_final, dgms, b1, c_final, red = opt.run_optimization(callback=log_progress)

    print("\\n--- RESULTADOS FINALES ---")
    print(f"Reducción total de compliance: {red:.2f}%")
    print(f"Número de Betti_1 (topología resultante): {b1}")

    # Verificación técnica de la hipótesis
    hipotesis_cumplida = red >= 40.0 and b1 <= 2
    print(f"¿Cumple con los umbrales de la H.E.2? {'SÍ' if hipotesis_cumplida else 'NO'}")

    # Exportar datos a un CSV para su uso en la tesis
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([{
        "resolucion": "60x30",
        "volfrac": 0.5,
        "penal": 3.0,
        "compliance_final": c_final,
        "reduccion_pct": red,
        "betti_1": b1
    }])
    df.to_csv("data/validacion_HE2_simp.csv", index=False)
    print("\\nResultados guardados exitosamente en: data/validacion_HE2_simp.csv")


if __name__ == "__main__":
    run_headless_experiment()