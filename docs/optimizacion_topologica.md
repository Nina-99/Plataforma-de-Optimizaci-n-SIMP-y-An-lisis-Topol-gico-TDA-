# Optimización Topológica de Vigas

## Guía de Usuario

### Acceso a la funcionalidad

La nueva funcionalidad de optimización topológica de vigas se encuentra en la pestaña "📈 3. Optimización Topológica" de la aplicación Streamlit. Para acceder a ella:

1. Abre la aplicación Streamlit
2. Selecciona "📈 3. Optimización Topológica" en el menú de navegación lateral

### Parámetros de entrada

Los siguientes parámetros pueden configurarse en el panel lateral:

- **Longitud de Viga (m)**: Longitud total de la viga a optimizar, entre 6.0 y 20.0 metros
- **Carga Distribuida (kN/m)**: Carga uniformemente distribuida sobre la viga, entre 10.0 y 80.0 kN/m
- **Módulo de Elasticidad (kPa)**: Módulo de elasticidad del material (por defecto 30,000,000 kPa para hormigón)
- **Base de la Viga (m)**: Ancho de la sección transversal de la viga
- **Altura de la Viga (m)**: Altura inicial de la sección transversal de la viga
- **Factor de Penalización**: Exponente topológico para el método SIMP (1-5)
- **Número de Nodos**: Resolución del análisis por diferencias finitas (50-200 nodos)

### Controles de visualización

- **Frecuencia de Actualización**: Controla con qué frecuencia se actualiza la visualización durante la optimización
- **Mostrar Armadura**: Activa/desactiva la visualización de la armadura de acero
- **Mostrar Estribos**: Activa/desactiva la visualización de estribos de corte
- **Mostrar Límites Admisibles**: Activa/desactiva la visualización de límites de diseño
- **Velocidad de Animación**: Controla la velocidad de la animación (afecta el tiempo de procesamiento)

### Interpretación de resultados

La visualización incluye tres subplots:

1. **Geometría de la viga**: Muestra la forma optimizada de la viga con:
   - Perfil superior e inferior
   - Relleno de hormigón
   - Armadura de acero (línea roja con grosor variable)
   - Estribos de corte (líneas punteadas grises)

2. **Curvas de deflexión**: Muestra:
   - Deflexión de la viga original (línea azul punteada)
   - Deflexión de la viga optimizada (línea roja continua)
   - Límite de deflexión admisible (línea roja punteada)

3. **Momentos flectores y tensiones**: Muestra:
   - Diagrama de momentos flectores (línea verde)
   - Tensión de compresión máxima (línea roja oscura)
   - Límite de tensión admisible (línea roja punteada)

### Exportación de datos

- **Datos CSV**: Exporta todos los datos numéricos de la optimización en formato CSV
- **Gráficos PDF**: Exporta los gráficos de resultados en formato PDF (funcionalidad en desarrollo)

## Guía Técnica

### Arquitectura de la implementación

La funcionalidad se basa en una arquitectura modular:

1. **Interfaz de usuario**: Implementada con Streamlit en `src/tda/app/app_master.py`
2. **Motor de optimización**: Clase `BeamOptimizer` en `src/tda/optimization/beam_optimizer.py`
3. **Visualización**: Utiliza Plotly para gráficos interactivos

### Proceso de optimización

El proceso utiliza el método SIMP (Solid Isotropic Material with Penalization):

1. **Discretización**: La viga se divide en segmentos según el número de nodos
2. **Análisis de elementos finitos**: Se resuelve la ecuación diferencial de Euler-Bernoulli
3. **Optimización**: Se actualiza iterativamente la distribución de inercia usando SIMP
4. **Convergencia**: El proceso continúa hasta alcanzar una tolerancia de error o máximo de iteraciones

### Callback y actualización en tiempo real

La función `optimizar_viga_completo` en `BeamOptimizer` acepta un callback que se ejecuta en cada iteración. Este callback proporciona datos de visualización que son procesados por la función `update_visualization` en la interfaz de Streamlit.

### Integración con BeamOptimizer

La clase `BeamOptimizer` se ha extendido con:

- Nuevos métodos para optimización completa con callbacks
- Parámetros adicionales para optimización topológica
- Funciones para calcular métricas y tensiones en tiempo real

## Solución de problemas

### Problemas comunes

1. **La optimización tarda demasiado**:
   - Reduce el número de nodos
   - Ajusta la frecuencia de actualización a "cada 5 iteraciones"
   - Desactiva elementos visuales no esenciales

2. **Resultados inesperados**:
   - Verifica que los parámetros de entrada estén dentro de rangos razonables
   - Asegúrate de que el factor de penalización no sea demasiado bajo

### Requisitos del sistema

- Python 3.7+
- Bibliotecas requeridas según `requirements.txt`
- Navegador web moderno para la interfaz de Streamlit

### Limitaciones conocidas

- La exportación de gráficos en PDF aún está en desarrollo
- La optimización puede ser intensiva en recursos para muchos nodos
- Algunas configuraciones de parámetros extremos pueden producir resultados no convergentes