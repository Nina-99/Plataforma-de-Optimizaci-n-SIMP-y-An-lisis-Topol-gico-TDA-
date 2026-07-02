"""
Beam Topology Optimization using Finite Difference Method (MDF) + SIMP

Pure mathematical solver - no matplotlib imports.
Optimization logic extracted from optimizacion_topologica.py

This class now includes functionality for real-time visualization updates
compatible with Streamlit.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_banded

class BeamOptimizer:
    def __init__(self, b, h0, p, N, a_c=1.0, A_maximo_concreto=1.0, E_c=1.0, E_ac=1.0, M_opt=1.0, max_iter=100):
        self.b = b
        self.h0 = h0
        self.p = p
        self.N = N
        self.a_c = a_c
        self.A_maximo_concreto = A_maximo_concreto
        self.E_c = E_c
        self.E_ac = E_ac
        self.M_opt = M_opt
        self.max_iter = max_iter
        # Parámetros adicionales para optimización topológica
        self.I0 = (b * h0**3) / 12  # Inercia inicial de la sección constante (m^4)
        self.I_min = 0.15 * self.I0  # Límite físico inferior de inercia de seguridad

    def calcular_momento(self, x, q, L):
        M = -q * x**2 / 2 + q * L * (L - x)
        return M

    def evaluar_funcion(self, M, Y_original, factor_simplificacion):
        K = np.dot(factor_simplificacion, np.sign(M)) * 70
        D_viga_idealizada = self.D_viga_ideasizada(K=K)
        
        # Calcular los factores de reducción en cada sección
        factors_reduction = D_viga_idealizada / M
        
        # Cálculo de la deflexión idealizada
        Y_optimal = K * factors_reduction
        
        # Función a minimizar
        penalizacion_volume = np.sum(factor_simplificacion)
        penalizacion_deflexion = np.linalg.norm(Y_optimal - Y_original, ord=2)**2
        funcion_objetivo = penalizacion_volume + 0.1 * penalizacion_deflexion
        
        return funcion_objetivo

    def D_viga_ideasizada(self, K):
        # Placeholder para la función D_viga_ideasizada
        # Deberías implementar esta función con base en tu lógica específica
        return np.ones_like(K) * 0.01  # Ejemplo básico

    def optimizar_viga(self, x, I, Y_original, M_initial, L, factor_simplificacion):
        bounds = [(0.05 * L, 0.45 * L)] * len(x)
        constraints = []
        
        def constraint_M(M):
            return np.sum(np.abs(M)) - self.M_opt
        
        A_o = sum(I) / (L ** 3)
        m_acero_permitida = self.a_c * self.b * self.h0 * I[0] * A_o
        factor_simplificacion_minimo = (1 / m_acero_permitida) * np.ones(len(x))
        
        def constraint_as_permitidas(factor_simplificacion):
            As_minima = 0.67 * self.A_maximo_concreto / self.b
            M_admisible = self.M_opt - self.E_c * I[0] * (12 / L **3)
            
            factor_A_min = np.sqrt((self.M_opt**2 + 4*self.E_ac*A_o) / ((self.b*self.h0)**4)) / self.A_maximo_concreto
            
            # Aquí debería ir el cálculo del valor mínimo permitido de factor_simplificación
            return np.full(len(x), factor_A_min)
        
        constraint = {'type': 'ineq', 'fun': constraint_as_permitidas}
        
        result = minimize(self.evaluar_funcion, M_initial + 10**-3 * np.sign(M_initial),
                          args=(Y_original, factor_simplificacion), method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x

    def simular_viga(self, L, q):
        """Simula la viga con el método de optimización topológica SIMP"""
        # Crear arreglo de posiciones
        x = np.linspace(0, L, self.N)
        dx = L / (self.N - 1)
        
        # Momento flector analítico exacto M(x) = q * x * (L - x) / 2
        M = q * x * (L - x) / 2
        
        # Configuración de la representación en bandas de la matriz tridiagonal A para -y'' = f
        ab = np.zeros((3, self.N - 2))
        ab[0, :] = -1.0 # diagonal superior
        ab[1, :] = 2.0  # diagonal principal
        ab[2, :] = -1.0 # diagonal inferior
        
        # 1. Deflexión original (viga con inercia constante I0)
        I_orig = np.ones(self.N) * self.I0
        f_orig = M[1:-1] / (self.E_c * I_orig[1:-1])
        rhs_orig = (dx**2) * f_orig
        
        y_internal_orig = solve_banded((1, 1), ab, rhs_orig)
        
        Y_original = np.zeros(self.N)
        Y_original[1:-1] = y_internal_orig
        
        # Inicialización de la inercia para optimización
        I = np.ones(self.N) * self.I0
        
        # Parámetros de control del bucle de optimización
        error = 1.0
        iteracion = 0
        tol = 1e-5
        
        # Almacenar datos para visualización
        visualization_data = []
        
        while error > tol and iteracion < self.max_iter:
            I_vieja = I.copy()
            
            # Resolución del sistema con la inercia actual
            f = M[1:-1] / (self.E_c * I[1:-1])
            rhs = (dx**2) * f
            y_internal = solve_banded((1, 1), ab, rhs)
            Y = np.zeros(self.N)
            Y[1:-1] = y_internal
            
            # Actualización SIMP vectorizada
            M_max = np.max(np.abs(M))
            if M_max == 0:
                M_max = 1.0
            I_target = np.clip(self.I_min + (self.I0 - self.I_min) * (np.abs(M) / M_max)**self.p, self.I_min, self.I0)
            I = 0.85 * I + 0.15 * I_target
            
            error = np.linalg.norm(I - I_vieja) / np.linalg.norm(I_vieja)
            
            # Preparar datos de visualización para esta iteración
            h_v = (12 * I / self.b) ** (1/3)
            visualization_data.append({
                "iteration": iteracion,
                "x": x,
                "I": I,
                "Y": Y,
                "Y_original": Y_original,
                "M": M,
                "h_v": h_v,
                "error": error
            })
            
            iteracion += 1
        
        return visualization_data

    def optimizar_viga_completo(self, L, q, callback=None):
        """
        Optimiza la viga completa con actualización en tiempo real.
        
        Args:
            L: Longitud de la viga (m)
            q: Carga distribuida (kN/m)
            callback: Función de callback para actualizar la visualización
            
        Returns:
            dict: Resultados finales de la optimización
        """
        # Crear arreglo de posiciones
        x = np.linspace(0, L, self.N)
        dx = L / (self.N - 1)
        
        # Momento flector analítico exacto M(x) = q * x * (L - x) / 2
        M = q * x * (L - x) / 2
        
        # Configuración de la representación en bandas de la matriz tridiagonal A para -y'' = f
        ab = np.zeros((3, self.N - 2))
        ab[0, :] = -1.0 # diagonal superior
        ab[1, :] = 2.0  # diagonal principal
        ab[2, :] = -1.0 # diagonal inferior
        
        # 1. Deflexión original (viga con inercia constante I0)
        I_orig = np.ones(self.N) * self.I0
        f_orig = M[1:-1] / (self.E_c * I_orig[1:-1])
        rhs_orig = (dx**2) * f_orig
        
        y_internal_orig = solve_banded((1, 1), ab, rhs_orig)
        
        Y_original = np.zeros(self.N)
        Y_original[1:-1] = y_internal_orig
        
        # Inicialización de la inercia para optimización
        I = np.ones(self.N) * self.I0
        
        # Parámetros de control del bucle de optimización
        error = 1.0
        iteracion = 0
        tol = 1e-5
        
        # Configurar límites iniciales
        max_def = np.max(np.abs(Y_original * 1000))
        if max_def == 0:
            max_def = 1.0
        y_adm = (L / 300.0) * 1000.0
        
        max_M = np.max(M)
        if max_M == 0:
            max_M = 1.0
            
        while error > tol and iteracion < self.max_iter:
            I_vieja = I.copy()
            
            # Resolución del sistema con la inercia actual
            f = M[1:-1] / (self.E_c * I[1:-1])
            rhs = (dx**2) * f
            y_internal = solve_banded((1, 1), ab, rhs)
            Y = np.zeros(self.N)
            Y[1:-1] = y_internal
            
            # Actualización SIMP vectorizada
            M_max = np.max(np.abs(M))
            if M_max == 0:
                M_max = 1.0
            I_target = np.clip(self.I_min + (self.I0 - self.I_min) * (np.abs(M) / M_max)**self.p, self.I_min, self.I0)
            I = 0.85 * I + 0.15 * I_target
            
            error = np.linalg.norm(I - I_vieja) / np.linalg.norm(I_vieja)
            
            # Calcular métricas para esta iteración
            h_v = (12 * I / self.b) ** (1/3)
            
            # Sugerencia 2: Armadura de acero longitudinal As(x)
            As = (np.abs(M) * 1e6) / (0.9 * (0.9 * h_v * 1000.0) * 420.0)
            
            # Sugerencia 5: Capacidad al corte y estribos de corte (shear stirrups)
            f_c = 25.0
            V_shear = q * (L / 2.0 - x)
            Vc = 0.17 * np.sqrt(f_c) * self.b * (0.9 * h_v) * 1000.0
            
            # Cálculo de tensiones
            sigma = (6.0 * np.abs(M)) / (self.b * h_v**2) # en kPa
            sigma_MPa = sigma / 1000.0
            
            # Sugerencia 4: Contador de KPIs dinámico
            V_orig = self.b * self.h0 * L
            V_opt = self.b * dx * np.sum(h_v)
            saving_pct = (1.0 - V_opt / V_orig) * 100.0
            weight_saved = (V_orig - V_opt) * 2.5
            
            # Preparar datos para callback
            if callback and iteracion % 2 == 0:  # Actualizar cada 2 iteraciones como en matplotlib
                visualization_data = {
                    "iteration": iteracion,
                    "x": x,
                    "I": I,
                    "Y": Y,
                    "Y_original": Y_original,
                    "M": M,
                    "h_v": h_v,
                    "As": As,
                    "V_shear": V_shear,
                    "Vc": Vc,
                    "sigma_MPa": sigma_MPa,
                    "saving_pct": saving_pct,
                    "weight_saved": weight_saved,
                    "error": error,
                    "L": L,
                    "y_adm": y_adm,
                    "max_M": max_M
                }
                
                # Llamar al callback con los datos de visualización
                callback(visualization_data)
            
            iteracion += 1
        
        # Retornar resultados finales
        h_v_final = (12 * I / self.b) ** (1/3)
        
        # Calcular datos adicionales para la visualización final
        As_final = (np.abs(M) * 1e6) / (0.9 * (0.9 * h_v_final * 1000.0) * 420.0)
        V_shear_final = q * (L / 2.0 - x)
        f_c = 25.0
        Vc_final = 0.17 * np.sqrt(f_c) * self.b * (0.9 * h_v_final) * 1000.0
        sigma_final = (6.0 * np.abs(M)) / (self.b * h_v_final**2) # en kPa
        sigma_MPa_final = sigma_final / 1000.0
        y_adm_final = (L / 300.0) * 1000.0
        
        return {
            "x": x,
            "I": I,
            "Y": Y,
            "Y_original": Y_original,
            "M": M,
            "h_v": h_v_final,
            "iterations": iteracion,
            "final_error": error,
            "saving_pct": saving_pct,
            "weight_saved": weight_saved,
            "As": As_final,
            "V_shear": V_shear_final,
            "Vc": Vc_final,
            "sigma_MPa": sigma_MPa_final,
            "L": L,
            "y_adm": y_adm_final
        }

    def calcular_metricas(self, Y, Y_original, M):
        # Calcular métricas básicas
        max_deflection_opt = np.max(np.abs(Y))
        sigma_max = np.max(np.abs(M)) * 0.1  # Valor de ejemplo, ajustar según necesidad
        savings_pct = 10  # Valor de ejemplo
        deflection_limit = 0.05  # Valor de ejemplo, ajustar según necesidad
        Y_opt = Y * 1.1  # Valor de ejemplo, ajustar según necesidad
        sigma = np.abs(M) * 0.1  # Valor de ejemplo, ajustar según necesidad
        return {
            "max_deflection_opt": max_deflection_opt,
            "sigma_max": sigma_max,
            "savings_pct": savings_pct,
            "deflection_limit": deflection_limit,
            "Y_opt": Y_opt,
            "sigma": sigma
        }

    def solve_generator(self, L, q):
        # Este es un ejemplo básico de generador.
        # Deberías implementar la lógica real aquí.
        for i in range(10):
            # Simulamos un paso de optimización
            x, I, Y, M_opt, Y_original = self.simular_viga(L, q)
            # Calcular h_v como un valor proporcional a Y o basado en alguna lógica
            h_v = np.abs(Y) * 0.1 + 0.05  # Valor de ejemplo, ajustar según necesidad
            # Calcular M como un valor proporcional a M_opt o basado en alguna lógica
            M = M_opt * 1.05  # Valor de ejemplo, ajustar según necesidad
            metrics = self.calcular_metricas(Y, Y_original, M)
            yield i, {"x": x, "I": I, "Y": Y, "M": M, "M_opt": M_opt, "Y_original": Y_original, "h_v": h_v, "metrics": metrics}
