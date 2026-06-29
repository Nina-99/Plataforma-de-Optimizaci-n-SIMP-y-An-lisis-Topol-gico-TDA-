import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from ripser import ripser
import io
from sklearn.decomposition import PCA

# Importación de la lógica refactorizada y separada de la UI
from tda.optimization.simp_optimizer import SimpTda2DOptimizer

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="TDA-SIMP Master Suite", layout="wide", page_icon="📐")
st.title("Plataforma de Optimización SIMP y Análisis Topológico (TDA)")
st.markdown("---")

# ==========================================
# CONFIGURACIÓN EN SIDEBAR (UI/UX Improvement 1)
# ==========================================
st.sidebar.title("Parámetros de Configuración")

# Selector de página
page = st.sidebar.radio(
    "Seleccione el Análisis", 
    ["🏗️ 1. Optimización SIMP (H.E.2)", "📊 2. TDA vs K-Medias (H.E.1)"],
    key="navigation_page"
)

st.sidebar.markdown("---")

if page == "🏗️ 1. Optimización SIMP (H.E.2)":
    st.sidebar.header("🏗️ Optimización SIMP (Pestaña 1)")
    malla_opcion = st.sidebar.selectbox("Resolución de Malla", ["60x30 (Caso Tesis)", "40x40 (1600 elem)", "80x80 (6400 elem)"], key="simp_malla")
    volfrac = st.sidebar.slider("Fracción de Volumen", 0.1, 0.9, 0.5, 0.05, key="simp_volfrac_input")
    penal = st.sidebar.number_input("Factor Penalización (p)", value=3.0, step=1.0, key="simp_penal")
    rmin = st.sidebar.number_input("Radio Filtro", value=1.5, step=0.1, key="simp_rmin")
else:
    st.sidebar.header("📊 TDA vs K-Medias (Pestaña 2)")
    n_points = st.sidebar.slider("Puntos por forma", 100, 500, 250, 50, key="tda_n_points")
    noise_level = st.sidebar.slider("Ruido Gaussiano (%)", 0.0, 0.3, 0.15, 0.05, key="tda_noise_level")
    n_clusters = st.sidebar.number_input("Clústeres (K-medias)", value=2, min_value=2, key="tda_n_clusters")

# ==========================================
# FUNCIONES AUXILIARES (TDA UI)
# ==========================================
def generate_cloud(shape, n_points):
    """Genera una nube de puntos 3D sintética para una esfera o toro.

    Args:
        shape (str): Forma a generar. 'sphere' o 'torus'.
        n_points (int): Número de puntos a muestrear.

    Returns:
        numpy.ndarray: Arreglo de forma (n_points, 3) representando la nube de puntos.

    Raises:
        ValueError: Si se proporciona una forma no reconocida.
    """
    if shape == "sphere":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.arccos(np.random.uniform(-1, 1, n_points))
        return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    elif shape == "torus":
        R, r = 2.0, 1.0
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        return np.column_stack([(R + r * np.cos(phi)) * np.cos(theta), (R + r * np.cos(phi)) * np.sin(theta), r * np.sin(phi)])

def add_gaussian_noise(points, noise_std):
    """Agrega ruido gaussiano a una nube de puntos relativo a su diámetro.

    Args:
        points (numpy.ndarray): Nube de puntos original de forma (N, 3).
        noise_std (float): Factor de desviación estándar del ruido.

    Returns:
        numpy.ndarray: Nube de puntos con ruido de forma (N, 3).
    """
    from scipy.spatial.distance import pdist
    diam = float(pdist(points).max())
    return points + np.random.normal(0, noise_std * diam, points.shape)

# ==========================================
# RENDERIZADO DEL PANEL PRINCIPAL
# ==========================================
if page == "🏗️ 1. Optimización SIMP (H.E.2)":
    # ------------------------------------------
    # PÁGINA 1: SIMP
    # ------------------------------------------
    st.header("Optimización Estructural 2D (SIMP + TDA)")
    
    ejecutar_simp = st.button("▶ Iniciar Optimización SIMP", type="primary")
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("Evolución del Diseño")
        plot_ph = st.empty()
    with col_res2:
        st.subheader("Métricas de Control")
        metric_ph = st.empty()
        pbar = st.empty()
        
    if ejecutar_simp:
        # Definición de dimensiones según selección
        dims = malla_opcion.split(" ")[0].split("x")
        nelx, nely = int(dims[0]), int(dims[1])
        
        progress_bar = pbar.progress(0)
        history = []
        
        # Callback para actualizar la UI
        def ui_callback(loop, xPhys, c, reduccion, max_iter):
            history.append({
                "Iteration": int(loop),
                "Compliance": float(c),
                "Reduction_Pct": float(reduccion)
            })
            if loop % 2 == 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(-xPhys, cmap='gray', aspect='equal')
                ax.axis('off')
                plot_ph.pyplot(fig)
                plt.close(fig)
                metric_ph.markdown(f"**Iteración:** {loop} | **Compliance:** {c:.2f} | **Reducción:** {reduccion:.1f}%")
            progress_bar.progress(min(loop / max_iter, 1.0))

        # Instancia del optimizador refactorizado
        opt = SimpTda2DOptimizer(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin)
        
        with st.spinner('Calculando elemento finito...'):
            x_final, dgms, b1, c_final, red = opt.run_optimization(callback=ui_callback)
            
        st.success("¡Optimización Finalizada!")
        
        # Guardar resultados en session state
        st.session_state.simp_x_final = x_final
        st.session_state.simp_dgms = dgms
        st.session_state.simp_b1 = b1
        st.session_state.simp_c_final = c_final
        st.session_state.simp_red = red
        st.session_state.simp_volfrac = volfrac
        st.session_state.simp_history = history
        st.session_state.simp_optimized = True
        
    # Renderizar resultados si existen en session state
    if st.session_state.get('simp_optimized', False):
        # Asegurarse de re-dibujar el diseño final si fue cargado de la sesión
        fig_final, ax_final = plt.subplots(figsize=(6, 3))
        ax_final.imshow(-st.session_state.simp_x_final, cmap='gray', aspect='equal')
        ax_final.axis('off')
        plot_ph.pyplot(fig_final)
        plt.close(fig_final)
        metric_ph.markdown(f"**Diseño Final Optimizado**")
        
        st.subheader("Métricas Finales")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Compliance Final", f"{st.session_state.simp_c_final:.4f}")
        col_m2.metric("Fracción de Volumen", f"{st.session_state.simp_volfrac:.2f}")
        col_m3.metric("Reducción Compliance", f"{st.session_state.simp_red:.2f}%")
        col_m4.metric("Número Betti_1", f"{st.session_state.simp_b1}")
        
        # Exportación
        st.subheader("Exportar Datos y Reportes")
        df_resultados = pd.DataFrame({
            "Metrica": ["Compliance Final", "Fraccion de Volumen", "Numero de Betti 1", "Reduccion %"],
            "Valor": [st.session_state.simp_c_final, st.session_state.simp_volfrac, st.session_state.simp_b1, st.session_state.simp_red]
        })
        csv_metrics = df_resultados.to_csv(index=False).encode('utf-8')
        
        # Historial de optimización
        history_data = st.session_state.get('simp_history', [])
        if history_data:
            df_history = pd.DataFrame(history_data)
            csv_history = df_history.to_csv(index=False).encode('utf-8')
        else:
            csv_history = csv_metrics

        # Densidades
        nely_size, nelx_size = st.session_state.simp_x_final.shape
        y_indices, x_indices = np.meshgrid(range(nely_size), range(nelx_size), indexing='ij')
        df_densities = pd.DataFrame({
            "Element_X": x_indices.flatten(),
            "Element_Y": y_indices.flatten(),
            "Density": st.session_state.simp_x_final.flatten()
        })
        csv_densities = df_densities.to_csv(index=False).encode('utf-8')
        
        # Renderizado de PDF usando Matplotlib native PDF backend
        pdf_buf = io.BytesIO()
        fig_pdf, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: final structure
        axes[0].imshow(-st.session_state.simp_x_final, cmap='gray', aspect='equal')
        axes[0].axis('off')
        axes[0].set_title("Estructura Optimizada Final (SIMP)")
        
        # Right: compliance history
        if history_data:
            df_hist = pd.DataFrame(history_data)
            axes[1].plot(df_hist["Iteration"], df_hist["Compliance"], 'b-', marker='o', markersize=3)
            axes[1].set_xlabel("Iteración")
            axes[1].set_ylabel("Compliance")
            axes[1].set_title("Historial de Convergencia (Compliance)")
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, "Historial no disponible", ha='center', va='center')
            axes[1].set_axis_off()
            
        fig_pdf.suptitle(f"Reporte de Optimización SIMP\nCompliance Final: {st.session_state.simp_c_final:.4f} | Betti-1: {st.session_state.simp_b1}", fontsize=14)
        plt.tight_layout()
        fig_pdf.savefig(pdf_buf, format='pdf')
        plt.close(fig_pdf)
        pdf_bytes = pdf_buf.getvalue()
        pdf_buf.close()

        col_csv1, col_csv2, col_pdf = st.columns(3)
        with col_csv1:
            st.download_button(
                label="📥 Descargar Historial de Optimización (CSV)",
                data=csv_history,
                file_name="historial_optimizacion.csv",
                mime="text/csv",
                key="btn_download_history"
            )
        with col_csv2:
            st.download_button(
                label="📥 Descargar Densidades Finales (CSV)",
                data=csv_densities,
                file_name="densidades_optimizacion.csv",
                mime="text/csv",
                key="btn_download_densities"
            )
        with col_pdf:
            st.download_button(
                label="📥 Descargar Reporte de Optimización (PDF)",
                data=pdf_bytes,
                file_name="reporte_optimizacion.pdf",
                mime="application/pdf",
                key="btn_download_pdf"
            )
        
        # Diagrama de persistencia interactivo con Plotly (UI/UX Improvement 2)
        st.subheader("Diagrama de Persistencia Interactivo")
        dgms = st.session_state.simp_dgms
        
        # Filtrar puntos no finitos (por ejemplo, np.inf) de dgms[0] y dgms[1]
        dgms_finite = []
        for dgm in dgms:
            if len(dgm) > 0:
                finite_mask = np.isfinite(dgm).all(axis=1)
                dgms_finite.append(dgm[finite_mask])
            else:
                dgms_finite.append(dgm)
        
        fig_tda = go.Figure()
        # H0
        if len(dgms_finite[0]) > 0:
            fig_tda.add_trace(go.Scatter(
                x=dgms_finite[0][:, 0],
                y=dgms_finite[0][:, 1],
                mode='markers',
                name='H0 (Componentes conexas)',
                marker=dict(size=8, opacity=0.7, color='blue')
            ))
        # H1
        if len(dgms_finite[1]) > 0:
            fig_tda.add_trace(go.Scatter(
                x=dgms_finite[1][:, 0],
                y=dgms_finite[1][:, 1],
                mode='markers',
                name='H1 (Cavidades/Túneles)',
                marker=dict(symbol='triangle-up', size=10, opacity=0.8, color='orange')
            ))
            
        # Diagonal de persistencia usando solo los valores finitos
        max_val = max(
            np.max(dgms_finite[0]) if len(dgms_finite[0]) > 0 else 1.0,
            np.max(dgms_finite[1]) if len(dgms_finite[1]) > 0 else 1.0
        )
        fig_tda.add_trace(go.Scatter(
            x=[0, max_val * 1.1],
            y=[0, max_val * 1.1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Nacimiento = Muerte',
            showlegend=False
        ))
        
        fig_tda.update_layout(
            xaxis_title="Tiempo de Nacimiento (Birth)",
            yaxis_title="Tiempo de Muerte (Death)",
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=40, b=40),
            height=450
        )
        st.plotly_chart(fig_tda, use_container_width=True)

else:
    # ------------------------------------------
    # PÁGINA 2: TDA vs K-MEDIAS
    # ------------------------------------------
    st.header("Robustez Topológica vs Distancia Euclidiana (H.E.1)")
    
    ejecutar_tda = st.button("▶ Ejecutar Comparación", type="primary", key="btn_tda")
    st.markdown("---")
    
    if ejecutar_tda:
        with st.spinner("Procesando geometrías..."):
            pts_sphere = generate_cloud("sphere", n_points)
            pts_torus = generate_cloud("torus", n_points)
            pts_torus[:, 0] += 1.5
            
            dataset = np.vstack([pts_sphere, pts_torus])
            dataset_noisy = add_gaussian_noise(dataset, noise_level)
            y_true = np.array([0]*n_points + [1]*n_points)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            y_kmeans = kmeans.fit_predict(dataset_noisy)
            
            res_sphere = ripser(add_gaussian_noise(pts_sphere, noise_level), maxdim=1)['dgms']
            res_torus = ripser(add_gaussian_noise(pts_torus, noise_level), maxdim=1)['dgms']
            
            # Guardar en session state
            st.session_state.tda_dataset_noisy = dataset_noisy
            st.session_state.tda_y_kmeans = y_kmeans
            st.session_state.tda_y_true = y_true
            st.session_state.tda_res_sphere = res_sphere
            st.session_state.tda_res_torus = res_torus
            st.session_state.tda_run = True

    # Renderizar resultados si existen en session state
    if st.session_state.get('tda_run', False):
        dataset_noisy = st.session_state.tda_dataset_noisy
        y_kmeans = st.session_state.tda_y_kmeans
        y_true = st.session_state.tda_y_true
        res_sphere = st.session_state.tda_res_sphere
        res_torus = st.session_state.tda_res_torus
        
        ct1, ct2 = st.columns(2)
        with ct1:
            st.subheader("Clasificación Euclidiana (K-Medias)")
            st.markdown("*Falla al separar geometrías superpuestas*")
            fig_k = go.Figure(data=[go.Scatter3d(
                x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
                mode='markers', marker=dict(size=3, color=y_kmeans, colorscale='Viridis')
            )])
            fig_k.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig_k, use_container_width=True)

        with ct2:
            st.subheader("Clasificación Real (Ground Truth)")
            st.markdown("*Separación perfecta por invariantes topológicos*")
            fig_true = go.Figure(data=[go.Scatter3d(
                x=dataset_noisy[:, 0], y=dataset_noisy[:, 1], z=dataset_noisy[:, 2],
                mode='markers', marker=dict(size=3, color=y_true, colorscale='Plasma')
            )])
            fig_true.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig_true, use_container_width=True)

        rt1, rt2 = st.columns(2)
        rt1.info(f"**Esfera:** $\beta_0 = {len(res_sphere[0])}$ | $\beta_1 = {len(res_sphere[1])}$")
        rt2.warning(f"**Toro:** $\beta_0 = {len(res_torus[0])}$ | $\beta_1 = {len(res_torus[1])}$")

        # Exportación
        st.subheader("Exportar Datos y Reportes de TDA")
        
        # 1. Crear CSV
        # dataset_noisy has shape (2*n_points, 3)
        # We need: coordinates, shape labels, K-Means labels, and calculated Betti numbers
        n_pts = len(dataset_noisy) // 2
        shape_labels = ["sphere"] * n_pts + ["torus"] * n_pts
        betti_0_list = [len(res_sphere[0])] * n_pts + [len(res_torus[0])] * n_pts
        betti_1_list = [len(res_sphere[1])] * n_pts + [len(res_torus[1])] * n_pts
        
        df_tda_export = pd.DataFrame({
            "x": dataset_noisy[:, 0],
            "y": dataset_noisy[:, 1],
            "z": dataset_noisy[:, 2],
            "shape_label": shape_labels,
            "kmeans_label": y_kmeans,
            "betti_0": betti_0_list,
            "betti_1": betti_1_list
        })
        csv_tda = df_tda_export.to_csv(index=False).encode('utf-8')
        
        # 2. Crear PDF
        pdf_buf_tda = io.BytesIO()
        fig_pdf_tda = plt.figure(figsize=(12, 10))
        
        # K-Means 3D
        ax1 = fig_pdf_tda.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_kmeans, cmap='viridis', s=5)
        ax1.set_title("Clasificación Euclidiana (K-Medias)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Ground Truth 3D
        ax2 = fig_pdf_tda.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(dataset_noisy[:, 0], dataset_noisy[:, 1], dataset_noisy[:, 2], c=y_true, cmap='plasma', s=5)
        ax2.set_title("Clasificación Real (Ground Truth)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        # Sphere persistence diagram
        ax3 = fig_pdf_tda.add_subplot(2, 2, 3)
        if len(res_sphere[0]) > 0:
            finite_h0 = res_sphere[0][np.isfinite(res_sphere[0][:, 1])] if res_sphere[0].ndim > 1 else res_sphere[0]
            if len(finite_h0) > 0:
                ax3.scatter(finite_h0[:, 0], finite_h0[:, 1], label="H0 (Componentes conexas)", color="blue", alpha=0.6)
        if len(res_sphere[1]) > 0:
            finite_h1 = res_sphere[1][np.isfinite(res_sphere[1][:, 1])] if res_sphere[1].ndim > 1 else res_sphere[1]
            if len(finite_h1) > 0:
                ax3.scatter(finite_h1[:, 0], finite_h1[:, 1], label="H1 (Cavidades/Túneles)", color="orange", marker="^", alpha=0.8)
        
        # diagonal
        max_val_s = 1.0
        for dgm in res_sphere:
            if len(dgm) > 0:
                finite_vals = dgm[np.isfinite(dgm)]
                if len(finite_vals) > 0:
                    max_val_s = max(max_val_s, np.max(finite_vals))
        ax3.plot([0, max_val_s * 1.1], [0, max_val_s * 1.1], 'k--', alpha=0.5)
        ax3.set_xlabel("Nacimiento (Birth)")
        ax3.set_ylabel("Muerte (Death)")
        ax3.set_title(f"Diagrama de Persistencia - Esfera\n(b0={len(res_sphere[0])}, b1={len(res_sphere[1])})")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # Torus persistence diagram
        ax4 = fig_pdf_tda.add_subplot(2, 2, 4)
        if len(res_torus[0]) > 0:
            finite_h0_t = res_torus[0][np.isfinite(res_torus[0][:, 1])] if res_torus[0].ndim > 1 else res_torus[0]
            if len(finite_h0_t) > 0:
                ax4.scatter(finite_h0_t[:, 0], finite_h0_t[:, 1], label="H0 (Componentes conexas)", color="blue", alpha=0.6)
        if len(res_torus[1]) > 0:
            finite_h1_t = res_torus[1][np.isfinite(res_torus[1][:, 1])] if res_torus[1].ndim > 1 else res_torus[1]
            if len(finite_h1_t) > 0:
                ax4.scatter(finite_h1_t[:, 0], finite_h1_t[:, 1], label="H1 (Cavidades/Túneles)", color="orange", marker="^", alpha=0.8)
        
        # diagonal
        max_val_t = 1.0
        for dgm in res_torus:
            if len(dgm) > 0:
                finite_vals = dgm[np.isfinite(dgm)]
                if len(finite_vals) > 0:
                    max_val_t = max(max_val_t, np.max(finite_vals))
        ax4.plot([0, max_val_t * 1.1], [0, max_val_t * 1.1], 'k--', alpha=0.5)
        ax4.set_xlabel("Nacimiento (Birth)")
        ax4.set_ylabel("Muerte (Death)")
        ax4.set_title(f"Diagrama de Persistencia - Toro\n(b0={len(res_torus[0])}, b1={len(res_torus[1])})")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)
        
        fig_pdf_tda.suptitle(f"Análisis Robustez Topológica (TDA vs K-Medias)\nRuido: {noise_level:.2f} | K-Medias Clusters: {n_clusters}", fontsize=14)
        plt.tight_layout()
        fig_pdf_tda.savefig(pdf_buf_tda, format='pdf')
        plt.close(fig_pdf_tda)
        pdf_bytes_tda = pdf_buf_tda.getvalue()
        pdf_buf_tda.close()
        
        col_tda_csv, col_tda_pdf = st.columns(2)
        with col_tda_csv:
            st.download_button(
                label="📥 Descargar Datos Nube de Puntos (CSV)",
                data=csv_tda,
                file_name="nube_puntos_tda.csv",
                mime="text/csv",
                key="btn_download_tda_csv"
            )
        with col_tda_pdf:
            st.download_button(
                label="📥 Descargar Gráficos de Análisis (PDF)",
                data=pdf_bytes_tda,
                file_name="reporte_analisis_tda.pdf",
                mime="application/pdf",
                key="btn_download_tda_pdf"
            )
