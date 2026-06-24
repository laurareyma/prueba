"""
Aliados Frescos — Tablero Interactivo de Optimizacion de Rutas
Frente 1: SA-TSP vs SA-TSPTW con ventanas horarias en la funcion objetivo.
"""

import io
import math
import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Configuracion de pagina ─────────────────────────────────────────
st.set_page_config(
    page_title="Aliados Frescos — Rutas",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores corporativa ───────────────────────────────────
COLOR_TSP   = "#EF553B"   # rojo: SA-TSP (sin ventanas)
COLOR_TSPTW = "#00CC96"   # verde: SA-TSPTW (con ventanas)
COLOR_CD    = "#AB63FA"   # morado: Centro de Distribucion
COLOR_OK    = "#00CC96"
COLOR_EARLY = "#FFA15A"
COLOR_LATE  = "#EF553B"
META_PEDVH  = 93.0        # OKR2-KR1: PEDVH >= 93 %


# ════════════════════════════════════════════════════════════════════
# LOGICA DE DATOS Y ALGORITMOS
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def generar_red(seed: int = 42):
    """
    Replica generar_red_logistica() del notebook (Frente 1).
    Devuelve red (DataFrame), DIST (ndarray 61x61).
    """
    rng = np.random.default_rng(seed)
    n   = 61   # 1 CD + 60 puntos de venta

    x   = rng.uniform(0, 100, n)
    y   = rng.uniform(0, 100, n)
    x[0], y[0] = 50.0, 50.0   # CD fijo al centro

    ids    = [f"CD"] + [f"P{i:02d}" for i in range(1, n)]
    tipos  = ["CD"] + ["PV"] * (n - 1)

    vent_ini = np.array([0.0] + list(rng.choice([5.0, 6.0, 7.0], size=n-1)))
    vent_fin = np.array([0.0] + [
        vent_ini[i] + (3.0 if rng.random() < 0.4 else 2.0)
        for i in range(1, n)
    ])

    red = pd.DataFrame({
        "x": x, "y": y, "id": ids, "tipo": tipos,
        "ventana_ini": vent_ini, "ventana_fin": vent_fin,
    })

    # Matriz de distancias euclidiana (km-equiv.)
    coords = red[["x", "y"]].values
    diff   = coords[:, None, :] - coords[None, :, :]
    DIST   = np.sqrt((diff ** 2).sum(axis=-1))

    return red, DIST


def _idx_muestra(red, seed: int = 42, n_paradas: int = 12):
    """Selecciona n_paradas puntos de venta (indices en red)."""
    pv = red[red["tipo"] == "PV"].index.tolist()
    rng = random.Random(seed)
    return rng.sample(pv, min(n_paradas, len(pv)))


# ── SA-TSP (sin ventanas) ────────────────────────────────────────────

def _costo_tsp(orden, DIST, deposito=0):
    ruta = [deposito] + list(orden) + [deposito]
    return sum(DIST[ruta[k], ruta[k+1]] for k in range(len(ruta)-1))


def _vecino_2opt(s):
    n = len(s)
    if n < 2:
        return s[:]
    i, j = sorted(random.sample(range(n), 2))
    v    = s[:]
    v[i:j+1] = v[i:j+1][::-1]
    return v


def _vecino_or_opt(s):
    if len(s) < 3:
        return s[:]
    v    = s[:]
    i    = random.randrange(len(v))
    nodo = v.pop(i)
    j    = random.randrange(len(v))
    v.insert(j, nodo)
    return v


def _vecino_swap(s):
    v    = s[:]
    i, j = random.sample(range(len(v)), 2)
    v[i], v[j] = v[j], v[i]
    return v


@st.cache_data
def ejecutar_sa_tsp(idx_tuple, dist_bytes, seed: int = 42,
                    T0=100.0, alpha=0.995, Tmin=0.01, max_iter=20_000):
    """SA-TSP clasico: minimiza solo distancia."""
    n_side = int(math.isqrt(len(dist_bytes) // 8))
    DIST   = np.frombuffer(dist_bytes, dtype=np.float64).reshape(n_side, n_side)

    idx = list(idx_tuple)
    random.seed(seed)
    s      = idx[:]
    random.shuffle(s)
    mejor  = s[:]
    T      = T0
    hist_m = []
    hist_a = []

    for it in range(max_iter):
        r = random.random()
        if   r < 0.60: v = _vecino_2opt(s)
        elif r < 0.85: v = _vecino_or_opt(s)
        else:          v = _vecino_swap(s)

        fc = _costo_tsp(v, DIST)
        fa = _costo_tsp(s, DIST)
        delta = fa - fc   # positivo si v es mejor (menor costo)
        if delta > 0 or random.random() < math.exp(min(-delta / T, 700)):
            s = v
        if _costo_tsp(s, DIST) < _costo_tsp(mejor, DIST):
            mejor = s[:]

        T *= alpha
        if it % 500 == 0:
            hist_m.append(_costo_tsp(mejor, DIST))
            hist_a.append(_costo_tsp(s, DIST))
        if T < Tmin:
            break

    return mejor, hist_m, hist_a


# ── SA-TSPTW (con ventanas en objetivo) ─────────────────────────────

def _costo_tsptw(orden, DIST, red, deposito=0,
                 vel=22.0, hs=4.0, serv=4.0/60.0, lam=200.0):
    ruta  = [deposito] + list(orden) + [deposito]
    dist_ = sum(DIST[ruta[k], ruta[k+1]] for k in range(len(ruta)-1))
    t, pen = hs, 0.0
    for i in range(1, len(ruta)-1):
        t   += DIST[ruta[i-1], ruta[i]] / vel
        t   += serv
        ini  = red["ventana_ini"].iloc[ruta[i]]
        fin  = red["ventana_fin"].iloc[ruta[i]]
        if   t > fin: pen += (t - fin) * lam
        elif t < ini: pen += (ini - t) * lam
    return dist_ + pen


@st.cache_data
def ejecutar_sa_tsptw(idx_tuple, dist_bytes, red_json: str, seed: int = 42,
                      T0=500.0, alpha=0.997, Tmin=0.05, max_iter=60_000,
                      vel=22.0, hs=4.0, lam=200.0):
    """SA-TSPTW: minimiza distancia + penalizacion simetrica de ventanas."""
    n_side = int(math.isqrt(len(dist_bytes) // 8))
    DIST   = np.frombuffer(dist_bytes, dtype=np.float64).reshape(n_side, n_side)
    red    = pd.read_json(io.StringIO(red_json))
    serv   = 4.0 / 60.0
    idx    = list(idx_tuple)

    random.seed(seed)
    s     = idx[:]
    random.shuffle(s)
    mejor = s[:]
    T     = T0
    hist_m, hist_a = [], []

    for it in range(max_iter):
        r = random.random()
        if   r < 0.60: v = _vecino_2opt(s)
        elif r < 0.85: v = _vecino_or_opt(s)
        else:          v = _vecino_swap(s)

        fc    = _costo_tsptw(v, DIST, red, vel=vel, hs=hs, serv=serv, lam=lam)
        fa    = _costo_tsptw(s, DIST, red, vel=vel, hs=hs, serv=serv, lam=lam)
        delta = fa - fc
        if delta > 0 or random.random() < math.exp(min(-delta / T, 700)):
            s = v
        if _costo_tsptw(s,DIST,red,vel=vel,hs=hs,serv=serv,lam=lam) < \
           _costo_tsptw(mejor,DIST,red,vel=vel,hs=hs,serv=serv,lam=lam):
            mejor = s[:]

        T *= alpha
        if it % 1000 == 0:
            hist_m.append(_costo_tsptw(mejor,DIST,red,vel=vel,hs=hs,serv=serv,lam=lam))
            hist_a.append(_costo_tsptw(s,DIST,red,vel=vel,hs=hs,serv=serv,lam=lam))
        if T < Tmin:
            break

    return mejor, hist_m, hist_a


# ── Metricas ─────────────────────────────────────────────────────────

def calcular_llegadas(orden, DIST, red, deposito=0,
                      vel=22.0, hs=4.0, serv=4.0/60.0):
    ruta   = [deposito] + list(orden)
    t      = hs
    filas  = []
    for i in range(1, len(ruta)):
        t   += DIST[ruta[i-1], ruta[i]] / vel
        t   += serv
        ini  = red["ventana_ini"].iloc[ruta[i]]
        fin  = red["ventana_fin"].iloc[ruta[i]]
        ok   = ini <= t <= fin
        estado = "OK" if ok else ("TEMPRANO" if t < ini else "TARDE")
        filas.append({
            "Stop": i,
            "Punto": red["id"].iloc[ruta[i]],
            "Llegada (h)": round(t, 3),
            "Ventana": f"{ini:.0f}h – {fin:.0f}h",
            "Estado": estado,
            "_ok": ok,
        })
    return pd.DataFrame(filas)


def pedvh(orden, DIST, red, **kw):
    df = calcular_llegadas(orden, DIST, red, **kw)
    if df.empty:
        return 0.0
    return df["_ok"].mean() * 100.0


def dist_ruta(orden, DIST, deposito=0):
    ruta = [deposito] + list(orden) + [deposito]
    return sum(DIST[ruta[k], ruta[k+1]] for k in range(len(ruta)-1))


@st.cache_data
def multi_semilla(idx_tuple, dist_bytes, red_json: str, n_seeds=15,
                  vel=22.0, hs=4.0, lam=200.0, seed_base=42):
    """Ejecuta SA-TSP y SA-TSPTW con n_seeds semillas distintas."""
    n_side = int(math.isqrt(len(dist_bytes) // 8))
    DIST   = np.frombuffer(dist_bytes, dtype=np.float64).reshape(n_side, n_side)
    red    = pd.read_json(io.StringIO(red_json))
    serv   = 4.0 / 60.0

    p_tsp, p_tw = [], []
    for k in range(n_seeds):
        # SA-TSP
        orden_tsp, _, _ = ejecutar_sa_tsp(idx_tuple, dist_bytes, seed=seed_base+k)
        p_tsp.append(pedvh(orden_tsp, DIST, red, vel=vel, hs=hs, serv=serv))
        # SA-TSPTW
        orden_tw, _, _  = ejecutar_sa_tsptw(
            idx_tuple, dist_bytes, red_json, seed=seed_base+k,
            vel=vel, hs=hs, lam=lam)
        p_tw.append(pedvh(orden_tw, DIST, red, vel=vel, hs=hs, serv=serv))

    return p_tsp, p_tw


# ════════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMETROS
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
        "Grupo_%C3%89xito_logo.svg/320px-Grupo_%C3%89xito_logo.svg.png",
        width=160,
    )
    st.title("Aliados Frescos")
    st.caption("Grupo Éxito — Optimización Logística")
    st.divider()

    st.subheader("Configuracion de instancia")
    seed_data  = st.number_input("Semilla red logistica", 0, 9999, 42, step=1)
    n_paradas  = st.slider("Numero de paradas", 6, 25, 12, step=1)

    st.subheader("Parametros de ruta")
    vel        = st.slider("Velocidad (km/h)", 10.0, 40.0, 22.0, step=1.0)
    hs         = st.slider("Hora de salida CD (h)", 3.0, 7.0, 4.0, step=0.5)

    st.subheader("SA-TSPTW")
    lam        = st.slider("Lambda penalizacion (λ)", 0, 500, 200, step=10)

    st.subheader("Analisis multi-semilla")
    n_seeds    = st.slider("Numero de semillas", 5, 30, 15, step=5)

    st.divider()
    st.caption("Los algoritmos se recalculan automaticamente al cambiar parametros.")


# ════════════════════════════════════════════════════════════════════
# CALCULO PRINCIPAL (con cache)
# ════════════════════════════════════════════════════════════════════

red, DIST = generar_red(seed=seed_data)
dist_bytes = DIST.tobytes()
red_json   = red.to_json()
idx        = _idx_muestra(red, seed=seed_data, n_paradas=n_paradas)
idx_tuple  = tuple(idx)

with st.spinner("Ejecutando SA-TSP..."):
    orden_tsp, hist_m_tsp, hist_a_tsp = ejecutar_sa_tsp(
        idx_tuple, dist_bytes, seed=42)

with st.spinner("Ejecutando SA-TSPTW..."):
    orden_tw, hist_m_tw, hist_a_tw = ejecutar_sa_tsptw(
        idx_tuple, dist_bytes, red_json, seed=42,
        vel=vel, hs=hs, lam=lam)

serv = 4.0 / 60.0
df_llegadas_tsp = calcular_llegadas(orden_tsp, DIST, red, vel=vel, hs=hs, serv=serv)
df_llegadas_tw  = calcular_llegadas(orden_tw,  DIST, red, vel=vel, hs=hs, serv=serv)
p_tsp_1  = pedvh(orden_tsp, DIST, red, vel=vel, hs=hs, serv=serv)
p_tw_1   = pedvh(orden_tw,  DIST, red, vel=vel, hs=hs, serv=serv)
d_tsp    = dist_ruta(orden_tsp, DIST)
d_tw     = dist_ruta(orden_tw,  DIST)


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════

st.title("Aliados Frescos — Tablero de Optimizacion de Rutas")
st.caption(
    f"Frente 1: SA-TSP vs SA-TSPTW  |  "
    f"{n_paradas} paradas  |  vel={vel} km/h  |  salida={hs:.1f}h  |  λ={lam}"
)

# KPI cards
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("PEDVH SA-TSP",   f"{p_tsp_1:.1f}%",
          help="Entregas dentro de ventana horaria — SA sin ventanas en objetivo")
k2.metric("PEDVH SA-TSPTW", f"{p_tw_1:.1f}%",
          delta=f"+{p_tw_1 - p_tsp_1:.1f} pp",
          help="Entregas dentro de ventana — SA con ventanas en objetivo")
k3.metric("Meta OKR2-KR1",  f"{META_PEDVH:.0f}%",
          delta=f"{p_tw_1 - META_PEDVH:.1f} pp",
          delta_color="normal")
k4.metric("Dist SA-TSP",    f"{d_tsp:.1f} km-eq")
k5.metric("Dist SA-TSPTW",  f"{d_tw:.1f} km-eq",
          delta=f"{d_tw - d_tsp:.1f}",
          delta_color="inverse")

st.divider()


# ════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════════

tab_rutas, tab_pedvh, tab_conv, tab_datos = st.tabs([
    "Mapas de Ruta", "Metricas PEDVH", "Convergencia SA", "Datos de Llegada"
])


# ────────────────────────────────────────────────────────────────────
# TAB 1: MAPAS DE RUTA
# ────────────────────────────────────────────────────────────────────

def _trazar_ruta(orden, DIST, red, color, nombre, deposito=0,
                 vel=22.0, hs=4.0, serv=4.0/60.0):
    ruta   = [deposito] + list(orden) + [deposito]
    xs     = [red["x"].iloc[p] for p in ruta]
    ys     = [red["y"].iloc[p] for p in ruta]
    labels = [red["id"].iloc[p] for p in ruta]
    df_ll  = calcular_llegadas(orden, DIST, red, vel=vel, hs=hs, serv=serv)
    estado_map = dict(zip(df_ll["Punto"], df_ll["Estado"]))
    vent_map   = dict(zip(df_ll["Punto"], df_ll["Ventana"]))
    llegada_map= dict(zip(df_ll["Punto"], df_ll["Llegada (h)"]))

    traces = []

    # Aristas de la ruta
    for k in range(len(ruta)-1):
        x0, y0 = red["x"].iloc[ruta[k]],   red["y"].iloc[ruta[k]]
        x1, y1 = red["x"].iloc[ruta[k+1]], red["y"].iloc[ruta[k+1]]
        traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Nodos por estado
    for est, col in [("OK", COLOR_OK), ("TEMPRANO", COLOR_EARLY), ("TARDE", COLOR_LATE)]:
        puntos = [p for p in orden if estado_map.get(red["id"].iloc[p], "") == est]
        if not puntos:
            continue
        traces.append(go.Scatter(
            x=[red["x"].iloc[p] for p in puntos],
            y=[red["y"].iloc[p] for p in puntos],
            mode="markers+text",
            marker=dict(color=col, size=11, line=dict(color="white", width=1.5)),
            text=[red["id"].iloc[p] for p in puntos],
            textposition="top center",
            textfont=dict(size=9),
            name=est,
            customdata=[[
                red["id"].iloc[p],
                llegada_map.get(red["id"].iloc[p], "—"),
                vent_map.get(red["id"].iloc[p], "—"),
                est,
            ] for p in puntos],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Llegada: %{customdata[1]:.2f}h<br>"
                "Ventana: %{customdata[2]}<br>"
                "Estado: %{customdata[3]}<extra></extra>"
            ),
        ))

    # CD
    traces.append(go.Scatter(
        x=[red["x"].iloc[deposito]], y=[red["y"].iloc[deposito]],
        mode="markers+text",
        marker=dict(color=COLOR_CD, size=16, symbol="star"),
        text=["CD"], textposition="top center",
        name="CD",
        hovertemplate="<b>Centro Distribucion</b><extra></extra>",
    ))

    return traces


with tab_rutas:
    col_l, col_r = st.columns(2)

    # SA-TSP
    with col_l:
        st.subheader(f"SA-TSP  |  PEDVH = {p_tsp_1:.1f}%")
        fig1 = go.Figure()
        for t in _trazar_ruta(orden_tsp, DIST, red, COLOR_TSP, "SA-TSP",
                               vel=vel, hs=hs, serv=serv):
            fig1.add_trace(t)
        fig1.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig1, width='stretch')

    # SA-TSPTW
    with col_r:
        st.subheader(f"SA-TSPTW  |  PEDVH = {p_tw_1:.1f}%")
        fig2 = go.Figure()
        for t in _trazar_ruta(orden_tw, DIST, red, COLOR_TSPTW, "SA-TSPTW",
                               vel=vel, hs=hs, serv=serv):
            fig2.add_trace(t)
        fig2.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig2, width='stretch')

    st.caption(
        "Verde = dentro de ventana horaria | "
        "Naranja = llegada temprana (antes de apertura) | "
        "Rojo = llegada tarde (despues de cierre)"
    )


# ────────────────────────────────────────────────────────────────────
# TAB 2: METRICAS PEDVH
# ────────────────────────────────────────────────────────────────────

with tab_pedvh:

    # Gauges
    ga, gb = st.columns(2)

    def _gauge(val, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=val,
            delta={"reference": META_PEDVH, "valueformat": ".1f",
                   "suffix": " pp vs meta"},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, 50],  "color": "#1f2937"},
                    {"range": [50, 93], "color": "#374151"},
                    {"range": [93, 100],"color": "#065f46"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": META_PEDVH,
                },
            },
            number={"suffix": "%", "valueformat": ".1f"},
        ))
        fig.update_layout(
            height=300, margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="#0e1117", font=dict(color="white"),
        )
        return fig

    ga.plotly_chart(_gauge(p_tsp_1,  "SA-TSP  (sin ventanas)", COLOR_TSP),
                    width='stretch')
    gb.plotly_chart(_gauge(p_tw_1, "SA-TSPTW (con ventanas)", COLOR_TSPTW),
                    width='stretch')

    st.divider()

    # Multi-semilla
    with st.expander("Analisis multi-semilla (puede tardar unos segundos)", expanded=True):
        with st.spinner(f"Ejecutando {n_seeds} semillas para cada algoritmo..."):
            p_list_tsp, p_list_tw = multi_semilla(
                idx_tuple, dist_bytes, red_json, n_seeds=n_seeds,
                vel=vel, hs=hs, lam=lam)

        media_tsp = float(np.mean(p_list_tsp))
        std_tsp   = float(np.std(p_list_tsp, ddof=1)) if n_seeds > 1 else 0.0
        media_tw  = float(np.mean(p_list_tw))
        std_tw    = float(np.std(p_list_tw,  ddof=1)) if n_seeds > 1 else 0.0
        ic_tsp    = 1.96 * std_tsp / math.sqrt(n_seeds)
        ic_tw     = 1.96 * std_tw  / math.sqrt(n_seeds)

        c1, c2, c3 = st.columns(3)
        c1.metric("SA-TSP media",   f"{media_tsp:.1f}%", f"±{ic_tsp:.1f}% IC95")
        c2.metric("SA-TSPTW media", f"{media_tw:.1f}%",  f"±{ic_tw:.1f}% IC95")
        c3.metric("Mejora media",   f"+{media_tw - media_tsp:.1f} pp")

        # Box plot comparativo
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=p_list_tsp, name="SA-TSP",
            marker_color=COLOR_TSP,
            boxpoints="all", jitter=0.3, pointpos=-1.8,
        ))
        fig_box.add_trace(go.Box(
            y=p_list_tw, name="SA-TSPTW",
            marker_color=COLOR_TSPTW,
            boxpoints="all", jitter=0.3, pointpos=-1.8,
        ))
        fig_box.add_hline(
            y=META_PEDVH, line_dash="dash", line_color="white",
            annotation_text=f"Meta {META_PEDVH:.0f}%",
            annotation_position="right",
        )
        fig_box.update_layout(
            title=f"Distribucion PEDVH — {n_seeds} semillas",
            yaxis_title="PEDVH (%)", yaxis_range=[0, 105],
            height=380, paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="white"),
            margin=dict(l=10, r=80, t=50, b=10),
        )
        st.plotly_chart(fig_box, width='stretch')

        # Barras por semilla
        seeds_x = list(range(n_seeds))
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[f"s{i}" for i in seeds_x], y=p_list_tsp,
            name="SA-TSP", marker_color=COLOR_TSP, opacity=0.85,
        ))
        fig_bar.add_trace(go.Bar(
            x=[f"s{i}" for i in seeds_x], y=p_list_tw,
            name="SA-TSPTW", marker_color=COLOR_TSPTW, opacity=0.85,
        ))
        fig_bar.add_hline(y=META_PEDVH, line_dash="dot", line_color="white",
                          annotation_text="Meta 93%", annotation_position="right")
        fig_bar.update_layout(
            barmode="group",
            title="PEDVH por semilla",
            yaxis_title="PEDVH (%)", yaxis_range=[0, 105],
            height=340, paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117", font=dict(color="white"),
            margin=dict(l=10, r=80, t=50, b=10),
        )
        st.plotly_chart(fig_bar, width='stretch')

    # Analisis de sensibilidad lambda
    with st.expander("Sensibilidad a lambda (fijo semilla=42)"):
        lambdas = [0, 50, 100, 150, 200, 300, 500]
        p_por_lam = []
        with st.spinner("Calculando..."):
            for lv in lambdas:
                o, _, _ = ejecutar_sa_tsptw(
                    idx_tuple, dist_bytes, red_json, seed=42,
                    vel=vel, hs=hs, lam=float(lv))
                p_por_lam.append(pedvh(o, DIST, red, vel=vel, hs=hs, serv=serv))

        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(
            x=lambdas, y=p_por_lam, mode="lines+markers",
            marker=dict(color=COLOR_TSPTW, size=9),
            line=dict(color=COLOR_TSPTW, width=2),
            name="PEDVH SA-TSPTW",
        ))
        fig_lam.add_hline(y=p_tsp_1, line_dash="dash", line_color=COLOR_TSP,
                          annotation_text=f"SA-TSP baseline {p_tsp_1:.1f}%",
                          annotation_position="right")
        fig_lam.add_hline(y=META_PEDVH, line_dash="dot", line_color="white",
                          annotation_text="Meta 93%", annotation_position="right")
        fig_lam.update_layout(
            title="PEDVH vs lambda de penalizacion",
            xaxis_title="lambda", yaxis_title="PEDVH (%)",
            yaxis_range=[0, 105], height=340,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"), margin=dict(l=10, r=100, t=50, b=10),
        )
        st.plotly_chart(fig_lam, width='stretch')


# ────────────────────────────────────────────────────────────────────
# TAB 3: CONVERGENCIA SA
# ────────────────────────────────────────────────────────────────────

with tab_conv:
    fig_conv = make_subplots(
        rows=1, cols=2,
        subplot_titles=["SA-TSP (distancia, semilla=42)",
                        "SA-TSPTW (costo mixto, semilla=42)"],
    )

    iters_tsp = [i * 500 for i in range(len(hist_m_tsp))]
    iters_tw  = [i * 1000 for i in range(len(hist_m_tw))]

    fig_conv.add_trace(go.Scatter(
        x=iters_tsp, y=hist_a_tsp, name="Actual TSP",
        line=dict(color=COLOR_TSP, width=1, dash="dot"), opacity=0.5,
    ), row=1, col=1)
    fig_conv.add_trace(go.Scatter(
        x=iters_tsp, y=hist_m_tsp, name="Mejor TSP",
        line=dict(color=COLOR_TSP, width=2.5),
    ), row=1, col=1)

    fig_conv.add_trace(go.Scatter(
        x=iters_tw, y=hist_a_tw, name="Actual TSPTW",
        line=dict(color=COLOR_TSPTW, width=1, dash="dot"), opacity=0.5,
    ), row=1, col=2)
    fig_conv.add_trace(go.Scatter(
        x=iters_tw, y=hist_m_tw, name="Mejor TSPTW",
        line=dict(color=COLOR_TSPTW, width=2.5),
    ), row=1, col=2)

    fig_conv.update_xaxes(title_text="Iteracion", row=1, col=1)
    fig_conv.update_xaxes(title_text="Iteracion", row=1, col=2)
    fig_conv.update_yaxes(title_text="Costo (km-eq)", row=1, col=1)
    fig_conv.update_yaxes(title_text="Costo mixto (dist + lambda*pen)", row=1, col=2)
    fig_conv.update_layout(
        height=420,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_conv, width='stretch')

    st.caption(
        "Linea solida = mejor solucion encontrada hasta la iteracion i.  "
        "Linea punteada = solucion actual (puede empeorar por aceptacion Boltzmann)."
    )

    # Parametros SA resumen
    st.divider()
    col_p1, col_p2 = st.columns(2)
    col_p1.markdown("**SA-TSP** (base)")
    col_p1.dataframe(pd.DataFrame({
        "Parametro": ["T0", "alpha", "T_min", "max_iter", "Vecindad"],
        "Valor":     ["100", "0.995", "0.01", "20 000",
                      "60% 2-opt / 25% Or-opt / 15% swap"],
    }), hide_index=True)

    col_p2.markdown("**SA-TSPTW** (con ventanas)")
    col_p2.dataframe(pd.DataFrame({
        "Parametro": ["T0", "alpha", "T_min", "max_iter", "lambda", "Vecindad"],
        "Valor":     ["500", "0.997", "0.05", "60 000", str(lam),
                      "60% 2-opt / 25% Or-opt / 15% swap"],
    }), hide_index=True)


# ────────────────────────────────────────────────────────────────────
# TAB 4: DATOS DE LLEGADA
# ────────────────────────────────────────────────────────────────────

with tab_datos:
    d1, d2 = st.columns(2)

    def _colorear(row):
        colors = {
            "OK":       "background-color:#065f46; color:white",
            "TEMPRANO": "background-color:#92400e; color:white",
            "TARDE":    "background-color:#7f1d1d; color:white",
        }
        return [colors.get(row["Estado"], "") if col == "Estado" else ""
                for col in row.index]

    with d1:
        st.subheader(f"SA-TSP  ({p_tsp_1:.1f}%)")
        show_tsp = df_llegadas_tsp.drop(columns=["_ok"])
        show_tsp["Llegada (h)"] = show_tsp["Llegada (h)"].apply(
            lambda h: f"{h:.2f}  ({int(h)}:{int((h%1)*60):02d})")
        st.dataframe(
            show_tsp.style.apply(_colorear, axis=1),
            width='stretch', hide_index=True,
        )

    with d2:
        st.subheader(f"SA-TSPTW  ({p_tw_1:.1f}%)")
        show_tw = df_llegadas_tw.drop(columns=["_ok"])
        show_tw["Llegada (h)"] = show_tw["Llegada (h)"].apply(
            lambda h: f"{h:.2f}  ({int(h)}:{int((h%1)*60):02d})")
        st.dataframe(
            show_tw.style.apply(_colorear, axis=1),
            width='stretch', hide_index=True,
        )

    st.divider()

    # Grafico timeline de llegadas
    fig_tl = go.Figure()
    colores_estado = {"OK": COLOR_OK, "TEMPRANO": COLOR_EARLY, "TARDE": COLOR_LATE}

    for alg, df_ll, color_alg in [
        ("SA-TSP",   df_llegadas_tsp, COLOR_TSP),
        ("SA-TSPTW", df_llegadas_tw,  COLOR_TSPTW),
    ]:
        for _, row in df_ll.iterrows():
            ini_v = red["ventana_ini"].iloc[0]   # placeholder; use row data
            # ventana bar
            v_parts = row["Ventana"].replace("h", "").split("–")
            v_ini, v_fin = float(v_parts[0].strip()), float(v_parts[1].strip())
            fig_tl.add_trace(go.Scatter(
                x=[v_ini, v_fin],
                y=[f"{alg} {row['Punto']}", f"{alg} {row['Punto']}"],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.15)", width=10),
                showlegend=False, hoverinfo="skip",
            ))
            # llegada marker
            fig_tl.add_trace(go.Scatter(
                x=[row["Llegada (h)"]],
                y=[f"{alg} {row['Punto']}"],
                mode="markers",
                marker=dict(
                    color=colores_estado.get(row["Estado"], "gray"),
                    size=12, symbol="diamond",
                ),
                name=row["Estado"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{alg} — {row['Punto']}</b><br>"
                    f"Llegada: {row['Llegada (h)']:.2f}h<br>"
                    f"Ventana: {row['Ventana']}<br>"
                    f"Estado: {row['Estado']}<extra></extra>"
                ),
            ))

    fig_tl.update_layout(
        title="Timeline de llegadas  (barra blanca = ventana horaria, diamante = llegada real)",
        xaxis_title="Hora del dia (h decimal)",
        height=max(380, 28 * n_paradas * 2),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=120, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_tl, width='stretch')

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Proyecto Aliados Frescos — Grupo Éxito · Bogotá  |  "
    f"Fuente: Generacion sintetica semilla={seed_data}  |  "
    "Algoritmo: Simulated Annealing con operadores 2-opt / Or-opt / Swap"
)
