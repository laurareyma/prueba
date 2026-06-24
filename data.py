"""
Aliados Frescos — Tablero Interactivo (Grupo 1)
Refleja los 4 frentes del notebook Proyecto_AliadosFrescos_Grupo1:
  F1 Recorridos (SA/A*/IDA*) · F2 Zonas (SA) · F3 Reabastecimiento (MDP) ·
  F4 Inventario largo plazo (PI/VI, robustez) · Tablero OKR · Validacion ·
  Metricas operativas (combustible, CPPA, PEDVH).

Toda la logica vive en core.py (port fiel del notebook); aqui solo se
orquesta el cache y se renderiza con Plotly/Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(
    page_title="Aliados Frescos — 4 Frentes",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta ───────────────────────────────────────────────────────────
C_SA    = "#EF553B"
C_AST   = "#636EFA"
C_IDA   = "#00CC96"
C_CD    = "#AB63FA"
C_OK    = "#00CC96"
C_OPT   = "#00CC96"
C_BASE  = "#EF553B"
ZONAS   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
DARK    = "#0e1117"


def _layout(fig, **kw):
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK, font=dict(color="white"),
        margin=dict(l=10, r=10, t=50, b=10), **kw)
    return fig


# ════════════════════════════════════════════════════════════════════
# CACHE — datos y experimentos (se calculan una sola vez por semilla)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_data(seed):
    red = core.generar_red_logistica(semilla=seed)
    DIST = core.matriz_distancias(red)
    DEMANDA = core.generar_demanda_historica(semilla=seed)
    return red, DIST, DEMANDA


@st.cache_data(show_spinner=False)
def c_f1(seed):
    red, DIST, _ = get_data(seed)
    return core.experimento_frente1(red, DIST)


@st.cache_data(show_spinner=False)
def c_f1_creciente(seed):
    red, DIST, _ = get_data(seed)
    return core.comparacion_creciente_f1(red, DIST)


@st.cache_data(show_spinner=False)
def c_f1_recal(seed):
    red, DIST, _ = get_data(seed)
    res = c_f1(seed)
    return core.recalibracion_sa(res["idx_grande"], DIST)


@st.cache_data(show_spinner=False)
def c_f1_sagrande(seed):
    red, DIST, _ = get_data(seed)
    return core.sa_grande_multisemilla(red, DIST)


@st.cache_data(show_spinner=False)
def c_f2(seed):
    red, DIST, _ = get_data(seed)
    return core.experimento_frente2(red, DIST)


@st.cache_data(show_spinner=False)
def c_f2_mapa():
    return core.mapa_zonas_grande()


@st.cache_data(show_spinner=False)
def c_f2_exacto(seed):
    red, DIST, _ = get_data(seed)
    return core.tabla_zonificacion_exacta(red, DIST)


@st.cache_data(show_spinner=False)
def c_f2_escala():
    return core.estudio_escalabilidad_zonas()


@st.cache_data(show_spinner=False)
def c_f3(seed, punto):
    _, _, DEMANDA = get_data(seed)
    return core.experimento_frente3(DEMANDA, punto=punto)


@st.cache_data(show_spinner=False)
def c_f4_pi(seed, punto):
    _, _, DEMANDA = get_data(seed)
    return core.sensibilidad_gamma_vidautil(DEMANDA, punto=punto)


@st.cache_data(show_spinner=False)
def c_f4_vi(seed, punto):
    _, _, DEMANDA = get_data(seed)
    return core.sensibilidad_gamma_vidautil_vi(DEMANDA, punto=punto)


@st.cache_data(show_spinner=False)
def c_f4_pivi(seed, punto):
    _, _, DEMANDA = get_data(seed)
    return core.comparar_pi_vi(DEMANDA, punto=punto)


@st.cache_data(show_spinner=False)
def c_f4_robustez(seed):
    _, _, DEMANDA = get_data(seed)
    return core.robustez_politica_frente4(DEMANDA)


@st.cache_data(show_spinner=False)
def c_valid(seed):
    red, DIST, _ = get_data(seed)
    return core.validacion_estadistica(red, DIST)


@st.cache_data(show_spinner=False)
def c_combustible(seed):
    red, DIST, _ = get_data(seed)
    res = c_f1(seed)
    return core.comparacion_combustible(res["idx_grande"], DIST)


@st.cache_data(show_spinner=False)
def c_cppa(seed):
    red, DIST, _ = get_data(seed)
    res = c_f1(seed)
    return core.metricas_cppa_pedvh(res["idx_grande"], red, DIST)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🥦 Aliados Frescos")
    st.caption("Grupo Éxito — Optimización Logística · Grupo 1")
    st.divider()
    seed = st.number_input("Semilla global (SEED)", 0, 9999, core.SEED, step=1)
    punto = st.number_input("Punto de venta (Frentes 3/4)", 0, 59, 0, step=1)
    st.divider()
    st.caption(
        "Los 4 frentes replican el notebook. Los experimentos pesados "
        "(robustez F4, validación, escalabilidad) se ejecutan bajo demanda.")
    st.caption("Datos sintéticos reproducibles con la semilla indicada.")

red, DIST, DEMANDA = get_data(seed)


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════

st.title("Tablero de Optimización — La red que no logra seguirle el paso a la ciudad")
st.caption(
    "Frente 1: SA · A* · IDA*  |  Frente 2: SA zonas  |  "
    "Frente 3: MDP perecedero · Policy Iteration  |  Frente 4: PI vs VI · robustez")

tab_res, tab_f1, tab_f2, tab_f3, tab_f4, tab_val = st.tabs([
    "📊 Resumen / OKR", "🚚 F1 Recorridos", "🗺️ F2 Zonas",
    "📦 F3 Reabastecimiento", "📈 F4 Largo plazo", "🔬 Validación",
])


# ────────────────────────────────────────────────────────────────────
# TAB RESUMEN — Tablero KPI vs OKR + métricas operativas
# ────────────────────────────────────────────────────────────────────

with tab_res:
    st.subheader("Tablero de indicadores — Solución integrada (4 frentes)")
    with st.spinner("Calculando Frentes 2 y 3 para el tablero integrado..."):
        tabla_f2, _, _ = c_f2(seed)
        tabla_f3, pi_opt, V_f3, ciclos_f3, mdp_f3, mdp_np_f3 = c_f3(seed, punto)
        tablero = core.tablero_kpi(tabla_f2, tabla_f3)

    st.dataframe(tablero, hide_index=True, width="stretch")
    st.caption(
        "Línea base = heurística / zonificación actual · "
        "Con solución = política óptima (PI) y zonificación SA.")

    st.divider()
    st.subheader("Métricas operativas por vehículo (Frente 1)")
    with st.spinner("Calculando combustible, CPPA y PEDVH..."):
        tabla_comb, ahorro_comb, ahorro_cop = c_combustible(seed)
        tabla_cppa, red_cppa, pedvh_sin, pedvh_opt, n_veh = c_cppa(seed)

    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**Consumo de combustible**")
        st.dataframe(tabla_comb, hide_index=True, width="stretch")
        st.metric("Ahorro de distancia y combustible", f"{ahorro_comb:.1f}%")
        st.caption(f"Ahorro económico por vehículo y jornada: {ahorro_cop:,.0f} COP")
    with o2:
        st.markdown(f"**CPPA y PEDVH** (vehículo de {n_veh} puntos)")
        st.dataframe(tabla_cppa, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("Reducción CPPA", f"{red_cppa:.1f}%", help="Meta OKR1-RC1: ≥20%")
        c2.metric("PEDVH (SA)", f"{pedvh_opt:.1f}%",
                  delta=f"{pedvh_opt - pedvh_sin:+.1f} pp", help="Meta OKR2-RC1: ≥93%")


# ────────────────────────────────────────────────────────────────────
# TAB F1 — Recorridos
# ────────────────────────────────────────────────────────────────────

def _mapa_ruta(orden, dist, red, color, titulo, deposito=0):
    ruta = [deposito] + list(orden) + [deposito]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[red["x"].iloc[p] for p in ruta], y=[red["y"].iloc[p] for p in ruta],
        mode="lines", line=dict(color=color, width=1.5), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[red["x"].iloc[p] for p in orden], y=[red["y"].iloc[p] for p in orden],
        mode="markers+text", marker=dict(color=color, size=11, line=dict(color="white", width=1)),
        text=[red["id"].iloc[p] for p in orden], textposition="top center",
        textfont=dict(size=9), showlegend=False,
        hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[red["x"].iloc[deposito]], y=[red["y"].iloc[deposito]],
        mode="markers+text", marker=dict(color=C_CD, size=18, symbol="star"),
        text=["CD"], textposition="top center", showlegend=False,
        hovertemplate="Centro de Distribución<extra></extra>"))
    return _layout(fig, title=titulo, height=460,
                   xaxis=dict(showgrid=False, zeroline=False),
                   yaxis=dict(showgrid=False, zeroline=False))


with tab_f1:
    st.subheader("Frente 1 — Optimización de recorridos de distribución")
    res1 = c_f1(seed)
    tabla_ex = res1["tabla_exacto"]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**SA vs A\\* vs IDA\\*** — instancia exacta (7 puntos)")
        st.dataframe(
            tabla_ex.style.format({"Distancia (km)": "{:.2f}", "Tiempo (ms)": "{:.1f}",
                                   "Gap vs optimo (%)": "{:.2f}"}),
            hide_index=True, width="stretch")
        st.caption(
            "A* e IDA* garantizan el óptimo (gap 0%); SA lo alcanza o casi, "
            "en una fracción del tiempo. Por eso SA escala a instancias grandes.")
        # barras de distancia por algoritmo (replica fig integrada del notebook)
        figb = go.Figure(go.Bar(
            x=tabla_ex["Algoritmo"], y=tabla_ex["Distancia (km)"],
            marker_color=[C_SA, C_AST, C_IDA],
            text=tabla_ex["Distancia (km)"].round(1), textposition="outside"))
        st.plotly_chart(_layout(figb, title="Distancia total por algoritmo",
                                height=320, yaxis_title="km"), width="stretch")
    with c2:
        st.plotly_chart(
            _mapa_ruta(res1["orden_sa_grande"], DIST, red, C_SA,
                       f"Ruta SA — {res1['n_grande']} puntos · "
                       f"{res1['costo_sa_grande']:.1f} km"),
            width="stretch")

    st.divider()
    st.markdown("**Comparación en instancias pequeñas crecientes (timeout 15 s)**")
    st.dataframe(c_f1_creciente(seed), hide_index=True, width="stretch")
    st.caption("Se reportan los 3 algoritmos en todas las instancias; "
               "'no viable' = supera el timeout exacto.")

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Recalibración SA ante cierres viales**")
        costo_rec, t_rec, orden_rec, cierres = c_f1_recal(seed)
        st.metric("Costo recalculado", f"{costo_rec:.1f} km", help=f"en {t_rec:.0f} ms")
        st.caption(f"Se penalizaron 5 tramos (×3). SA re-optimiza la ruta sobre "
                   f"la red modificada. Tramos cerrados: {cierres}")
    with cc2:
        st.markdown("**Instancias grandes (40 y 60 puntos) — solo SA, 10 semillas**")
        if st.button("Ejecutar multi-semilla grande (~15 s)", key="b_sagrande"):
            st.session_state["sagrande"] = True
        if st.session_state.get("sagrande"):
            with st.spinner("Ejecutando SA en 40 y 60 puntos..."):
                st.dataframe(c_f1_sagrande(seed), hide_index=True, width="stretch")

    st.divider()
    st.markdown("**Consumo de combustible: optimizado vs manual**")
    tabla_comb, ahorro_comb, ahorro_cop = c_combustible(seed)
    st.dataframe(tabla_comb, hide_index=True, width="stretch")
    st.caption(f"Ahorro de distancia/combustible por vehículo: {ahorro_comb:.1f}% "
               f"· {ahorro_cop:,.0f} COP por jornada.")


# ────────────────────────────────────────────────────────────────────
# TAB F2 — Zonas
# ────────────────────────────────────────────────────────────────────

with tab_f2:
    st.subheader("Frente 2 — Reasignación de zonas de atención")
    with st.spinner("Ejecutando SA de zonas (enfriamiento rápido vs lento)..."):
        tabla_f2, asig_final, prob_f2 = c_f2(seed)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            tabla_f2.style.format({"Costo total red": "{:.1f}", "Tiempo (ms)": "{:.0f}",
                                   "Reduccion vs actual (%)": "{:.1f}"}),
            hide_index=True, width="stretch")
        red_lento = tabla_f2.loc[2, "Reduccion vs actual (%)"]
        st.metric("Reducción de costo (T alta + enfriamiento lento)", f"{red_lento:.1f}%")
        st.caption("El enfriamiento lento explora más y escapa de óptimos locales, "
                   "reduciendo el costo total de la red frente a la zonificación actual.")
    with c2:
        # mapa de zonas sobre la red real con la asignacion optimizada
        fig = go.Figure()
        zonas = {c: [] for c in range(prob_f2.n)}
        for p, c in zip(prob_f2.puntos, asig_final):
            zonas[c].append(p)
        for c, pts in zonas.items():
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[red["x"].iloc[p] for p in pts], y=[red["y"].iloc[p] for p in pts],
                mode="markers", marker=dict(color=ZONAS[c % len(ZONAS)], size=9),
                name=f"Cuadrilla {c} ({len(pts)})",
                hovertemplate="%{text}<extra></extra>",
                text=[red["id"].iloc[p] for p in pts]))
        fig.add_trace(go.Scatter(
            x=[red["x"].iloc[0]], y=[red["y"].iloc[0]], mode="markers+text",
            marker=dict(color="white", size=18, symbol="star"), text=["CD"],
            textposition="top center", name="CD", hoverinfo="skip"))
        st.plotly_chart(
            _layout(fig, title="Zonas reasignadas (red real, 60 puntos)", height=460,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.18)),
            width="stretch")

    st.divider()
    st.markdown("**¿Por qué SA y no búsqueda exacta?**")
    e1, e2 = st.columns(2)
    with e1:
        if st.button("Demostrar inviabilidad de zonificación exacta", key="b_zexacto"):
            st.session_state["zexacto"] = True
        if st.session_state.get("zexacto"):
            with st.spinner("Enumerando particiones con rutas A* (timeout 15 s)..."):
                st.dataframe(c_f2_exacto(seed), hide_index=True, width="stretch")
            st.caption(f"Red real (60 pts, 4 cuadrillas): 4⁶⁰ ≈ {4**60:.2e} particiones. "
                       "La zonificación exacta es combinatoriamente inviable ⇒ se usa SA.")
    with e2:
        if st.button("Estudio de escalabilidad de SA (~20 s)", key="b_escala"):
            st.session_state["escala"] = True
        if st.session_state.get("escala"):
            with st.spinner("Midiendo SA en 60/200/500/1000 puntos..."):
                tabla_esc, coef, pred_1800, factor, maxit = c_f2_escala()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tabla_esc["n puntos"], y=tabla_esc["tiempo (ms)"],
                mode="lines+markers", name="medido", line=dict(color=C_IDA)))
            xs = np.linspace(60, 1800, 100)
            fig.add_trace(go.Scatter(
                x=xs, y=np.polyval(coef, xs), mode="lines",
                name="tendencia (grado 2)", line=dict(color="gray", dash="dash")))
            fig.add_trace(go.Scatter(
                x=[1800], y=[pred_1800], mode="markers",
                marker=dict(color=C_SA, size=12),
                name=f"extrap. 1800 (~{pred_1800/1000:.1f} s)"))
            st.plotly_chart(_layout(fig, title="Escalabilidad de SA en zonas",
                                    height=340, xaxis_title="nº puntos",
                                    yaxis_title="tiempo SA (ms)"), width="stretch")
            st.caption(f"Con calidad de producción (max_iter=40000, ×{factor:.0f}) "
                       f"⇒ ~{pred_1800/1000*factor/60:.0f} min para 1800 puntos: "
                       "no es práctico en tiempo real (sí offline).")


# ────────────────────────────────────────────────────────────────────
# TAB F3 — Reabastecimiento
# ────────────────────────────────────────────────────────────────────

with tab_f3:
    st.subheader("Frente 3 — Reabastecimiento diario de producto PERECEDERO (vida útil 3 días)")
    with st.spinner("Resolviendo MDP con Policy Iteration..."):
        tabla_f3, pi_opt, V_f3, ciclos_f3, mdp_f3, mdp_np_f3 = c_f3(seed, punto)

    st.caption(f"Policy Iteration convergió en {ciclos_f3} ciclos · "
               f"{len(mdp_f3.S)} estados (inv_max={mdp_f3.inv_max}).")
    st.dataframe(
        tabla_f3.style.format({"Rentabilidad": "{:.1f}", "Nivel servicio": "{:.1%}",
                               "Tasa agotados": "{:.1%}", "Tasa merma": "{:.1%}"}),
        hide_index=True, width="stretch")
    st.caption("Ignorar la vida útil tiene doble costo: política subóptima en el mundo real "
               "y métricas engañosas (confunde inventario arrastrado con merma).")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        # V(s) optima sobre slice vence_manana = 0
        slice_h = [s for s in mdp_f3.S if mdp_f3.dec[s][1] == 0]
        xs_f3 = [mdp_f3.dec[s][0] for s in slice_h]
        fig = go.Figure(go.Scatter(
            x=xs_f3, y=[V_f3[s] for s in slice_h], mode="lines+markers",
            line=dict(color=C_AST), marker=dict(size=8)))
        st.plotly_chart(_layout(fig, title="V(s) óptima (perecedero, vence_mañana=0)",
                                height=340, xaxis_title="Unidades que vencen hoy",
                                yaxis_title="Valor esperado"), width="stretch")
    with c2:
        # politica optima: stock que vence hoy -> cuanto pedir
        pol = {mdp_f3.dec[s][0]: pi_opt[s] for s in slice_h}
        xs = sorted(pol)
        fig = go.Figure(go.Bar(
            x=xs, y=[pol[x] for x in xs], marker_color=C_OPT))
        st.plotly_chart(_layout(fig, title="Política óptima: stock hoy → cuánto pedir",
                                height=340, xaxis_title="Unidades que vencen hoy",
                                yaxis_title="Pedido óptimo"), width="stretch")


# ────────────────────────────────────────────────────────────────────
# TAB F4 — Largo plazo
# ────────────────────────────────────────────────────────────────────

with tab_f4:
    st.subheader("Frente 4 — Política de inventario a largo plazo (perecedero 3 días)")
    with st.spinner("Sensibilidad a γ con Policy Iteration..."):
        tabla_pi, politicas_pi = c_f4_pi(seed, punto)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Sensibilidad a γ (Policy Iteration)**")
        st.dataframe(
            tabla_pi.style.format({"Nivel servicio": "{:.3f}", "Tasa merma": "{:.3f}",
                                   "Rentabilidad": "{:.1f}", "Pedido medio": "{:.3f}"}),
            hide_index=True, width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tabla_pi["gamma"], y=tabla_pi["Nivel servicio"],
                                 mode="lines+markers", name="Nivel servicio", line=dict(color=C_OK)))
        fig.add_trace(go.Scatter(x=tabla_pi["gamma"], y=tabla_pi["Tasa merma"],
                                 mode="lines+markers", name="Tasa merma", line=dict(color=C_SA)))
        st.plotly_chart(_layout(fig, title="Sensibilidad a γ (servicio vs merma)",
                                height=320, xaxis_title="γ"), width="stretch")
    with c2:
        n_dist = len(set(politicas_pi.values()))
        st.metric("Políticas distintas entre los 5 valores de γ", n_dist)
        st.caption("Al modelar la perecibilidad, γ **sí** cambia la política "
                   "(a mayor γ, mayor buffer). El modelo sin memoria daría "
                   "sensibilidad plana.")
        with st.spinner("Verificando convergencia PI vs VI..."):
            ciclos_pi, iters_vi, coinciden = c_f4_pivi(seed, punto)
        st.markdown("**Policy Iteration vs Value Iteration** (γ=0.95)")
        st.write(f"- PI converge en **{ciclos_pi}** ciclos")
        st.write(f"- VI converge en **{iters_vi}** iteraciones")
        st.write(f"- Misma política óptima: **{'✅ sí' if coinciden else '❌ no'}**")

    st.divider()
    with st.expander("Sensibilidad a γ con Value Iteration (verificación)"):
        with st.spinner("Value Iteration sobre 5 valores de γ..."):
            tabla_vi, _ = c_f4_vi(seed, punto)
        st.dataframe(
            tabla_vi.style.format({"Nivel servicio": "{:.3f}", "Tasa merma": "{:.3f}",
                                   "Rentabilidad": "{:.1f}", "Pedido medio": "{:.3f}"}),
            hide_index=True, width="stretch")

    st.divider()
    st.markdown("**Robustez de la política global (una política por perfil)**")
    st.caption("Toda la red se cubre con solo 3 políticas (una por perfil económico), "
               "no una por punto. Experimento pesado (~60 s).")
    if st.button("Ejecutar análisis de robustez F4 (~60 s)", key="b_rob"):
        st.session_state["rob"] = True
    if st.session_state.get("rob"):
        with st.spinner("Entrenando políticas por perfil y midiendo degradación..."):
            tabla_perf, degradacion, tabla_esc, conteo = c_f4_robustez(seed)
        st.write("Puntos por perfil:", conteo)
        st.dataframe(
            tabla_perf.style.format({
                "Nivel serv. media": "{:.3f}", "Nivel serv. std": "{:.3f}",
                "Tasa merma media": "{:.3f}", "Tasa merma std": "{:.3f}",
                "Rentab. media": "{:.0f}", "Rentab. std": "{:.0f}"}),
            hide_index=True, width="stretch")
        st.metric("Degradación media vs óptimo individual", f"{degradacion:.2f}%",
                  help="Baja degradación ⇒ una política por perfil generaliza sin "
                       "recalibrar punto por punto.")
        st.markdown("**Desempeño bajo escenarios cambiantes**")
        fig = go.Figure()
        for perfil in tabla_esc["Perfil"].unique():
            sub = tabla_esc[tabla_esc["Perfil"] == perfil]
            fig.add_trace(go.Bar(x=sub["Escenario"], y=sub["Nivel servicio"], name=perfil))
        st.plotly_chart(_layout(fig, title="Nivel de servicio por escenario y perfil",
                                height=340, barmode="group", yaxis_title="Nivel servicio"),
                        width="stretch")


# ────────────────────────────────────────────────────────────────────
# TAB VALIDACIÓN
# ────────────────────────────────────────────────────────────────────

with tab_val:
    st.subheader("Diseño experimental y validación estadística (sección 7)")
    st.caption("Multi-semilla + pruebas no paramétricas (Wilcoxon, Friedman). "
               "Experimento pesado (~30 s).")
    if st.button("Ejecutar validación estadística (~30 s)", key="b_val"):
        st.session_state["val"] = True
    if st.session_state.get("val"):
        with st.spinner("Repitiendo SA con múltiples semillas y corriendo tests..."):
            v = c_valid(seed)

        st.markdown("**Frente 1 — Multi-semilla: SA vs óptimo exacto (15 corridas)**")
        st.dataframe(v["tabla_ms_f1"], hide_index=True, width="stretch")
        st.caption(f"Brecha media de SA respecto al óptimo (A*): {v['brecha']:.2f}% · "
                   f"SA encontró el óptimo en {v['n_opt']}/{v['n_corridas']} corridas.")

        # boxplot de SA vs optimo
        fig = go.Figure()
        fig.add_trace(go.Box(y=v["costos_sa_f1"], name="SA (15 semillas)",
                             marker_color=C_SA, boxpoints="all", jitter=0.3))
        fig.add_hline(y=v["costo_opt_f1"], line_dash="dash", line_color=C_IDA,
                      annotation_text=f"Óptimo A* = {v['costo_opt_f1']:.1f}",
                      annotation_position="right")
        st.plotly_chart(_layout(fig, title="Distribución de costo SA vs óptimo",
                                height=340, yaxis_title="km"), width="stretch")

        st.divider()
        st.markdown("**Frente 2 — Multi-semilla por configuración (10 corridas)**")
        st.dataframe(v["tabla_ms_f2"], hide_index=True, width="stretch")
        st.caption(f"El enfriamiento lento mejora el costo medio en "
                   f"{v['mejora_lento']:.1f}% frente al rápido.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            stat_w, p_w = v["wilcoxon"]
            st.markdown("**Wilcoxon** (F2: lento vs rápido)")
            st.write(f"- estadístico = {stat_w:.3f} · p-valor = {p_w:.5f}")
            st.write("- " + ("Diferencia significativa (p<0.05) ✅" if p_w < 0.05
                             else "Sin diferencia significativa"))
        with c2:
            stat_f, p_f = v["friedman"]
            st.markdown("**Friedman** (F1: tres enfriamientos)")
            st.write(f"- estadístico = {stat_f:.3f} · p-valor = {p_f:.5f}")
            st.write("- " + ("Al menos una configuración difiere (p<0.05) ✅" if p_f < 0.05
                             else "Sin diferencia significativa"))
        st.dataframe(v["tabla_friedman"], hide_index=True, width="stretch")


# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Proyecto Aliados Frescos — Grupo Éxito · Grupo 1  |  "
    f"Semilla={seed}  |  Algoritmos del notebook: SA · A* · IDA* · "
    "Policy/Value Iteration  |  Datos sintéticos reproducibles")
