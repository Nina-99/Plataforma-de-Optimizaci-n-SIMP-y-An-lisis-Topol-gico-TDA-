# Módulo de Optimización (SIMP + TDA)

Este módulo implementa el algoritmo SIMP (Solid Isotropic Material with Penalization) acoplado con análisis topológico de datos para resolver problemas de optimización estructural en dos dimensiones.

## Optimización Topológica de Vigas

Este módulo también incluye funcionalidad para optimización topológica de vigas utilizando el método de diferencias finitas y el enfoque SIMP. La implementación se encuentra en el archivo `beam_optimizer.py`.

## Contexto Matemático

La optimización topológica busca encontrar la distribución óptima de material dentro de un dominio para minimizar la flexibilidad (cumplir con la rigidez estructural máxima), sujeta a una restricción sobre el volumen de material total. El modelo matemático SIMP penaliza las densidades intermedias para aproximarse a un diseño binario de vacío (0) o sólido (1):

$$\min_{\boldsymbol{\rho}} \quad c(\boldsymbol{\rho}) = \mathbf{U}^T \mathbf{K}(\boldsymbol{\rho}) \mathbf{U} = \sum_{e=1}^{N} (\rho_e)^p \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$$
$$\text{sujeto a} \quad \frac{V(\boldsymbol{\rho})}{V_0} \leq f$$
$$\mathbf{K}(\boldsymbol{\rho}) \mathbf{U} = \mathbf{F}$$
$$\mathbf{0} < \boldsymbol{\rho}_{min} \leq \boldsymbol{\rho} \leq \mathbf{1}$$

donde:

- $c$: Compliance estructural (trabajo de las cargas externas o inversa de la rigidez).
- $\boldsymbol{\rho}$: Vector de densidades de los elementos.
- $p$: Exponente de penalización (típicamente $p = 3$).
- $\mathbf{U}, \mathbf{F}$: Vectores globales de desplazamientos y fuerzas.
- $\mathbf{K}$: Matriz de rigidez global del sistema, ensamblada mediante Elementos Finitos (FEM) bilineales de 4 nodos.
- $f$: Fracción de volumen objetivo.

Una vez finalizada la optimización, las densidades de material filtradas ($\rho_e > 0.5$) se mapean a una nube de puntos discretos sobre la cual se calcula la homología persistente $H_1$ para extraer el número de Betti $\beta_1$, el cual cuenta el número de "agujeros" estructurales del diseño resultante.

## Relevancia en la Tesis

Este módulo es la piedra angular para la validación de la **Hipótesis Específica H.E.2**. Dicha hipótesis establece que el acoplamiento del método SIMP con TDA permite validar soluciones que logren una reducción de compliance de al menos el 40.0% con respecto al diseño base uniforme, garantizando a su vez restricciones topológicas rigurosas expresadas por un número de ciclos estructurales de control $\beta_1 \leq 2$ (previniendo topologías excesivamente porosas o fracturadas). El caso de validación principal es la Viga Voladizo (_Cantilever Beam_) sobre una malla de discretización de $60 \times 30$ elementos.

## Entradas y Salidas de las Funciones

### `SimpTda2DOptimizer` (Clase Principal)

### `BeamOptimizer` (Clase para Optimización de Vigas)

- **Parámetros del Constructor (`__init__`):**
  - `nelx` (int): Número de elementos en el eje X (por defecto $60$).
  - `nely` (int): Número de elementos en el eje Y (por defecto $30$).
  - `volfrac` (float): Fracción de volumen objetivo $f$ (por defecto $0.5$).
  - `penal` (float): Factor de penalización SIMP $p$ (por defecto $3.0$).
  - `rmin` (float): Radio del filtro de densidad para evitar patrones tipo tablero de ajedrez (por defecto $1.5$).

### `run_optimization(callback=None)`

- **Entradas:**
  - `callback` (callable, opcional): Función de monitoreo de firma `(loop, xPhys, c, reduccion, max_iter)` invocada en cada iteración del lazo de optimización.
- **Salidas:**
  - `xPhys` (np.ndarray): Matriz bidimensional de densidades finales optimizadas.
  - `dgms` (list): Diagramas de persistencia del diseño optimizado generados con ripser.
  - `betti_1` (int): Cantidad de ciclos topológicos persistentes significativos ($\beta_1$).
  - `c` (float): Compliance final del sistema optimizado.
  - `reduccion_pct` (float): Porcentaje de reducción del compliance con respecto a la estructura inicial homogénea.

### `optimizar_viga_completo(L, q, callback=None)`

- **Entradas:**
  - `L` (float): Longitud de la viga (m)
  - `q` (float): Carga distribuida (kN/m)
  - `callback` (callable, opcional): Función de callback para actualizar la visualización en tiempo real
- **Salidas:**
  - `x` (np.ndarray): Posiciones a lo largo de la viga
  - `I` (np.ndarray): Momentos de inercia optimizados
  - `Y` (np.ndarray): Deflexiones optimizadas
  - `Y_original` (np.ndarray): Deflexiones de la viga original
  - `M` (np.ndarray): Momentos flectores
  - `h_v` (np.ndarray): Alturas de la viga optimizada
  - `iterations` (int): Número de iteraciones realizadas
  - `final_error` (float): Error final de convergencia
  - `saving_pct` (float): Porcentaje de ahorro de material
  - `weight_saved` (float): Peso ahorrado (toneladas)

## Dependencias

- `numpy`
- `scipy.sparse`
- `scipy.sparse.linalg`
- `ripser` (para el cálculo de persistencia en la nube de puntos del diseño final)
