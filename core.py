"""
core.py — Aliados Frescos (Grupo 1)
Portabilidad fiel de los algoritmos del notebook Proyecto_AliadosFrescos_Grupo1
a un modulo reutilizable por el tablero Streamlit (data.py).

Los 4 frentes + validacion estadistica + metricas operativas se transcriben
sin alterar la logica del notebook para que los resultados coincidan.

Frente 1: Optimizacion de recorridos  (SA / A* / IDA*)
Frente 2: Reasignacion de zonas        (SA: enfriamiento rapido vs lento)
Frente 3: Reabastecimiento perecedero  (MDP vida util 3 dias, Policy Iteration)
Frente 4: Politica de inventario       (sensibilidad gamma, PI vs VI, robustez)
"""

import math
import random
import time
import heapq
from itertools import product

import numpy as np
import pandas as pd

SEED = 42


# ════════════════════════════════════════════════════════════════════
# 1. GENERACION Y PREPARACION DE DATOS
# ════════════════════════════════════════════════════════════════════

def generar_red_logistica(n_puntos=60, n_cuadrillas=4, semilla=SEED):
    rng = np.random.default_rng(semilla)
    centro = np.array([0.0, 0.0])                       # CD fijo en el origen
    coords = rng.uniform(-25, 25, size=(n_puntos, 2))   # cuadrado 50x50 km
    coords = np.vstack([centro, coords])
    ids = ["CD"] + [f"P{i:02d}" for i in range(1, n_puntos + 1)]
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["id"] = ids
    df["tipo"] = ["CD"] + list(rng.choice(
        ["tienda", "surtimax", "carulla"], size=n_puntos, p=[0.6, 0.25, 0.15]))
    apertura = rng.choice([5, 6, 7], size=n_puntos)
    df.loc[1:, "ventana_ini"] = apertura
    df.loc[1:, "ventana_fin"] = apertura + rng.choice([2, 3], size=n_puntos)
    df.loc[0, ["ventana_ini", "ventana_fin"]] = [0, 24]   # CD 24h
    df["cuadrilla_actual"] = [-1] + list(rng.integers(0, n_cuadrillas, size=n_puntos))
    return df


def matriz_distancias(df):
    P = df[["x", "y"]].to_numpy()
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def generar_demanda_historica(n_puntos=60, n_dias=180, semilla=SEED):
    rng = np.random.default_rng(semilla + 1)
    dias = np.arange(n_dias)
    base = rng.uniform(20, 80, size=n_puntos)
    demanda = np.zeros((n_dias, n_puntos))
    for p in range(n_puntos):
        estacional = 1 + 0.3 * np.sin(2 * np.pi * dias / 7)
        quincena = np.where((dias % 15 == 0) | (dias % 30 == 0), 1.4, 1.0)
        ruido = rng.normal(1.0, 0.15, size=n_dias)
        demanda[:, p] = np.clip(base[p] * estacional * quincena * ruido, 0, None)
    return demanda.round().astype(int)


# ════════════════════════════════════════════════════════════════════
# FRENTE 1 — OPTIMIZACION DE RECORRIDOS (SA / A* / IDA*)
# ════════════════════════════════════════════════════════════════════

class ProblemaRuta:
    def __init__(self, indices_puntos, dist, deposito=0):
        self.puntos = list(indices_puntos)
        self.dist = dist
        self.deposito = deposito

    def costo_ruta(self, orden):
        ruta = [self.deposito] + list(orden) + [self.deposito]
        return sum(self.dist[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1))

    def estado_aleatorio(self):
        s = self.puntos[:]
        random.shuffle(s)
        return s

    def f(self, orden):
        return -self.costo_ruta(orden)

    def vecino_aleatorio(self, orden):
        s = list(orden)
        i, j = random.sample(range(len(s)), 2)
        s[i], s[j] = s[j], s[i]
        return s


def heuristica_euclidiana(actual, pendientes, deposito, dist):
    if not pendientes:
        return dist[actual, deposito]
    al_mas_cercano = min(dist[actual, p] for p in pendientes)
    regreso = min(dist[p, deposito] for p in pendientes)
    return al_mas_cercano + regreso


def simulated_annealing(prob, T0=100.0, alpha=0.995, Tmin=0.01, max_iter=20000):
    s = prob.estado_aleatorio()
    mejor = s
    T = T0
    it = 0
    while T > Tmin and it < max_iter:
        vecino = prob.vecino_aleatorio(s)
        delta = prob.f(vecino) - prob.f(s)
        if delta > 0 or random.random() < math.exp(delta / T):
            s = vecino
        if prob.f(s) > prob.f(mejor):
            mejor = s
        T *= alpha
        it += 1
    return mejor


def resolver_ruta_sa(indices, dist, **kwargs):
    prob = ProblemaRuta(indices, dist)
    t0 = time.perf_counter()
    orden = simulated_annealing(prob, **kwargs)
    t1 = time.perf_counter()
    return {"orden": orden, "costo": prob.costo_ruta(orden), "tiempo_ms": (t1 - t0) * 1000}


def a_estrella_ruta(indices, dist, deposito=0):
    puntos = frozenset(indices)
    g_inicio = 0
    h_inicio = heuristica_euclidiana(deposito, puntos, deposito, dist)
    frontera = [(h_inicio, g_inicio, deposito, frozenset(), [deposito])]
    explorados = {}
    while frontera:
        f, g, actual, visitados, camino = heapq.heappop(frontera)
        if visitados == puntos:
            return camino[1:], g + dist[actual, deposito]
        clave = (actual, visitados)
        if clave in explorados and explorados[clave] <= g:
            continue
        explorados[clave] = g
        for sig in puntos - visitados:
            g_nuevo = g + dist[actual, sig]
            nuevos_visitados = visitados | {sig}
            pendientes = puntos - nuevos_visitados
            h = heuristica_euclidiana(sig, pendientes, deposito, dist)
            heapq.heappush(frontera, (g_nuevo + h, g_nuevo, sig, nuevos_visitados, camino + [sig]))
    return None, float("inf")


def ida_estrella_ruta(indices, dist, deposito=0):
    puntos = frozenset(indices)
    cutoff = heuristica_euclidiana(deposito, puntos, deposito, dist)
    mejor_camino = [None]

    def buscar(actual, visitados, g, cutoff, camino):
        if visitados == puntos:
            f = g + dist[actual, deposito]
            if f > cutoff:
                return f
            mejor_camino[0] = camino[1:]
            return "ENCONTRADO"
        pendientes = puntos - visitados
        f = g + heuristica_euclidiana(actual, pendientes, deposito, dist)
        if f > cutoff:
            return f
        minimo = float("inf")
        for sig in pendientes:
            g_nuevo = g + dist[actual, sig]
            r = buscar(sig, visitados | {sig}, g_nuevo, cutoff, camino + [sig])
            if r == "ENCONTRADO":
                return "ENCONTRADO"
            if r < minimo:
                minimo = r
        return minimo

    while True:
        r = buscar(deposito, frozenset(), 0, cutoff, [deposito])
        if r == "ENCONTRADO":
            ruta = mejor_camino[0]
            full = [deposito] + ruta + [deposito]
            costo = sum(dist[full[i], full[i + 1]] for i in range(len(full) - 1))
            return ruta, costo
        if r == float("inf"):
            return None, float("inf")
        cutoff = r


# --- Variantes con control de tiempo (deadline interno) ---------------

TIMEOUT_EXACTO_S = 15.0


class _Timeout(Exception):
    pass


def a_estrella_timeout(indices, dist, deposito=0, timeout_s=TIMEOUT_EXACTO_S):
    t0 = time.perf_counter()
    puntos = frozenset(indices)
    h0 = heuristica_euclidiana(deposito, puntos, deposito, dist)
    frontera = [(h0, 0, deposito, frozenset(), [deposito])]
    explorados = {}
    while frontera:
        if time.perf_counter() - t0 > timeout_s:
            return None, float("inf")
        f, g, actual, visitados, camino = heapq.heappop(frontera)
        if visitados == puntos:
            return camino[1:], g + dist[actual, deposito]
        clave = (actual, visitados)
        if clave in explorados and explorados[clave] <= g:
            continue
        explorados[clave] = g
        for sig in puntos - visitados:
            g_nuevo = g + dist[actual, sig]
            nuevos = visitados | {sig}
            h = heuristica_euclidiana(sig, puntos - nuevos, deposito, dist)
            heapq.heappush(frontera, (g_nuevo + h, g_nuevo, sig, nuevos, camino + [sig]))
    return None, float("inf")


def ida_estrella_timeout(indices, dist, deposito=0, timeout_s=TIMEOUT_EXACTO_S):
    t0 = time.perf_counter()
    puntos = frozenset(indices)
    cutoff = heuristica_euclidiana(deposito, puntos, deposito, dist)
    mejor = [None]

    def buscar(actual, visitados, g, cutoff, camino):
        if time.perf_counter() - t0 > timeout_s:
            raise _Timeout()
        if visitados == puntos:
            f = g + dist[actual, deposito]
            if f > cutoff:
                return f
            mejor[0] = camino[1:]
            return "ENCONTRADO"
        pend = puntos - visitados
        f = g + heuristica_euclidiana(actual, pend, deposito, dist)
        if f > cutoff:
            return f
        minimo = float("inf")
        for sig in pend:
            r = buscar(sig, visitados | {sig}, g + dist[actual, sig], cutoff, camino + [sig])
            if r == "ENCONTRADO":
                return "ENCONTRADO"
            if r < minimo:
                minimo = r
        return minimo

    try:
        while True:
            r = buscar(deposito, frozenset(), 0, cutoff, [deposito])
            if r == "ENCONTRADO":
                ruta = mejor[0]
                full = [deposito] + ruta + [deposito]
                costo = sum(dist[full[i], full[i + 1]] for i in range(len(full) - 1))
                return ruta, costo
            if r == float("inf"):
                return None, float("inf")
            cutoff = r
    except _Timeout:
        return None, float("inf")


def comparacion_creciente_f1(red, dist, tamanos=(5, 6, 7, 8),
                             timeout_s=TIMEOUT_EXACTO_S, semilla=SEED):
    rng = np.random.default_rng(semilla + 5)
    base = list(rng.choice(range(1, len(red)), size=max(tamanos), replace=False))
    filas = []
    for n in tamanos:
        idx = base[:n]
        prob = ProblemaRuta(idx, dist)
        random.seed(SEED); np.random.seed(SEED)
        t0 = time.perf_counter(); orden = simulated_annealing(prob)
        t_sa = (time.perf_counter() - t0) * 1000
        c_sa = prob.costo_ruta(orden)
        t0 = time.perf_counter(); _, c_a = a_estrella_timeout(idx, dist, timeout_s=timeout_s)
        t_a = (time.perf_counter() - t0) * 1000
        a_ok = c_a != float("inf")
        t0 = time.perf_counter(); _, c_ida = ida_estrella_timeout(idx, dist, timeout_s=timeout_s)
        t_ida = (time.perf_counter() - t0) * 1000
        ida_ok = c_ida != float("inf")
        nv = f"no viable (>{timeout_s:.0f}s)"
        filas.append({
            "n": n,
            "SA dist (km)": round(c_sa, 2), "SA (ms)": round(t_sa, 1),
            "A* dist (km)": round(c_a, 2) if a_ok else nv, "A* (ms)": round(t_a, 1),
            "IDA* dist (km)": round(c_ida, 2) if ida_ok else nv, "IDA* (ms)": round(t_ida, 1),
        })
    return pd.DataFrame(filas)


def sa_grande_multisemilla(red, dist, tamanos=(40, 60), n_semillas=10, semilla=SEED):
    filas = []
    for n in tamanos:
        rng = np.random.default_rng(semilla + 100 + n)
        idx = list(rng.choice(range(1, len(red)), size=n, replace=False))
        prob = ProblemaRuta(idx, dist)
        costos, tiempos = [], []
        for k in range(n_semillas):
            random.seed(SEED + k); np.random.seed(SEED + k)
            t0 = time.perf_counter(); orden = simulated_annealing(prob)
            tiempos.append((time.perf_counter() - t0) * 1000)
            costos.append(prob.costo_ruta(orden))
        random.seed(SEED); np.random.seed(SEED)
        m = float(np.mean(costos)); s = float(np.std(costos, ddof=1))
        margen = 1.96 * s / math.sqrt(len(costos))
        filas.append({
            "n puntos": n,
            "Dist media (km)": round(m, 2), "Desv. est.": round(s, 2),
            "IC95 inf": round(m - margen, 2), "IC95 sup": round(m + margen, 2),
            "Tiempo medio (ms)": round(np.mean(tiempos), 1),
        })
    return pd.DataFrame(filas)


def experimento_frente1(red, dist, n_grande=25, n_exacto=7, semilla=SEED):
    rng = np.random.default_rng(semilla + 5)
    idx_grande = list(rng.choice(range(1, len(red)), size=n_grande, replace=False))
    prob_g = ProblemaRuta(idx_grande, dist)
    random.seed(SEED); np.random.seed(SEED)
    t0 = time.perf_counter()
    orden_sa_g = simulated_annealing(prob_g)
    t_sa_g = (time.perf_counter() - t0) * 1000
    costo_sa_g = prob_g.costo_ruta(orden_sa_g)
    idx_exacto = idx_grande[:n_exacto]
    prob_e = ProblemaRuta(idx_exacto, dist)
    t0 = time.perf_counter(); orden_sa_e = simulated_annealing(prob_e); t_sa_e = (time.perf_counter() - t0) * 1000
    costo_sa_e = prob_e.costo_ruta(orden_sa_e)
    t0 = time.perf_counter(); _, costo_a = a_estrella_ruta(idx_exacto, dist); t_a = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); _, costo_ida = ida_estrella_ruta(idx_exacto, dist); t_ida = (time.perf_counter() - t0) * 1000
    tabla_exacto = pd.DataFrame([
        ("Simulated Annealing", costo_sa_e, t_sa_e),
        ("A*", costo_a, t_a),
        ("IDA*", costo_ida, t_ida),
    ], columns=["Algoritmo", "Distancia (km)", "Tiempo (ms)"])
    tabla_exacto["Gap vs optimo (%)"] = (
        tabla_exacto["Distancia (km)"] / tabla_exacto["Distancia (km)"].min() - 1) * 100
    return {
        "tabla_exacto": tabla_exacto, "idx_grande": idx_grande, "idx_exacto": idx_exacto,
        "orden_sa_grande": orden_sa_g, "costo_sa_grande": costo_sa_g, "t_sa_grande": t_sa_g,
        "n_grande": n_grande,
    }


def recalibracion_sa(indices, dist, n_cierres=5, semilla=SEED):
    rng = np.random.default_rng(semilla + 9)
    dist_mod = dist.copy()
    cierres = []
    for _ in range(n_cierres):
        i, j = rng.choice(indices, size=2, replace=False)
        dist_mod[i, j] *= 3
        dist_mod[j, i] *= 3
        cierres.append((int(i), int(j)))
    random.seed(SEED); np.random.seed(SEED)
    t0 = time.perf_counter()
    orden = simulated_annealing(ProblemaRuta(indices, dist_mod), T0=50, alpha=0.99)
    t = (time.perf_counter() - t0) * 1000
    return ProblemaRuta(indices, dist_mod).costo_ruta(orden), t, orden, cierres


# ════════════════════════════════════════════════════════════════════
# FRENTE 2 — REASIGNACION DE ZONAS
# ════════════════════════════════════════════════════════════════════

class ProblemaZonas:
    def __init__(self, red, dist, n_cuadrillas=4, deposito=0, w_comp=1.0):
        self.dist = dist
        self.n = n_cuadrillas
        self.deposito = deposito
        self.puntos = list(range(1, len(red)))
        self.coords = red[["x", "y"]].to_numpy()
        self.w_comp = w_comp

    def costo_ruta_aprox(self, asignados):
        if not asignados:
            return 0.0
        no_visitados = set(asignados)
        actual = self.deposito
        total = 0.0
        while no_visitados:
            sig = min(no_visitados, key=lambda p: self.dist[actual, p])
            total += self.dist[actual, sig]
            actual = sig
            no_visitados.remove(sig)
        return total + self.dist[actual, self.deposito]

    def compacidad(self, zonas):
        total = 0.0
        for c in zonas:
            pts = zonas[c]
            if pts:
                centroide = self.coords[pts].mean(axis=0)
                total += np.sqrt(((self.coords[pts] - centroide) ** 2).sum(axis=1)).sum()
        return total

    def costo_total(self, asignacion):
        zonas = {c: [] for c in range(self.n)}
        for p, c in zip(self.puntos, asignacion):
            zonas[c].append(p)
        distancia = sum(self.costo_ruta_aprox(zonas[c]) for c in zonas)
        tamanos = [len(zonas[c]) for c in zonas]
        penal_balance = np.std(tamanos) * 8.0 + sum(40.0 for t in tamanos if t == 0)
        compacidad = self.compacidad(zonas) * self.w_comp
        return distancia + penal_balance + compacidad

    def f(self, asignacion):
        return -self.costo_total(asignacion)

    def estado_aleatorio(self):
        return [random.randrange(self.n) for _ in self.puntos]

    def vecino_aleatorio(self, asignacion):
        s = list(asignacion)
        i = random.randrange(len(s))
        s[i] = random.randrange(self.n)
        return s


def reasignar_zonas_sa(red, dist, n_cuadrillas=4, T0=500.0, alpha=0.999, Tmin=0.1, max_iter=40000):
    prob = ProblemaZonas(red, dist, n_cuadrillas)
    t0 = time.perf_counter()
    asignacion = simulated_annealing(prob, T0=T0, alpha=alpha, Tmin=Tmin, max_iter=max_iter)
    t = (time.perf_counter() - t0) * 1000
    return prob, asignacion, t


def experimento_frente2(red, dist, n_cuadrillas=4):
    prob = ProblemaZonas(red, dist, n_cuadrillas)
    costo_inicial = prob.costo_total(red["cuadrilla_actual"].iloc[1:].tolist())
    random.seed(SEED); np.random.seed(SEED)
    _, asig_rapido, t_rapido = reasignar_zonas_sa(
        red, dist, n_cuadrillas, T0=10, alpha=0.95, Tmin=0.1, max_iter=5000)
    costo_rapido = prob.costo_total(asig_rapido)
    random.seed(SEED); np.random.seed(SEED)
    _, asig_lento, t_lento = reasignar_zonas_sa(
        red, dist, n_cuadrillas, T0=500, alpha=0.999, Tmin=0.1, max_iter=40000)
    costo_lento = prob.costo_total(asig_lento)
    tabla = pd.DataFrame([
        ("Zonificacion actual", costo_inicial, 0.0),
        ("SA enfriamiento rapido", costo_rapido, t_rapido),
        ("SA T alta + enfriamiento lento", costo_lento, t_lento),
    ], columns=["Configuracion", "Costo total red", "Tiempo (ms)"])
    tabla["Reduccion vs actual (%)"] = (1 - tabla["Costo total red"] / costo_inicial) * 100
    return tabla, asig_lento, prob


def zonificacion_exacta(idx_puntos, dist, n_cuadrillas=2, timeout_s=TIMEOUT_EXACTO_S):
    t0 = time.perf_counter()
    mejor_costo = float("inf")
    evaluadas = 0
    agotado = False
    for asignacion in product(range(n_cuadrillas), repeat=len(idx_puntos)):
        if time.perf_counter() - t0 > timeout_s:
            agotado = True
            break
        zonas = {c: [] for c in range(n_cuadrillas)}
        for p, c in zip(idx_puntos, asignacion):
            zonas[c].append(p)
        costo = 0.0
        for c in zonas:
            if zonas[c]:
                _, c_ruta = a_estrella_timeout(zonas[c], dist, timeout_s=timeout_s)
                costo += c_ruta
        mejor_costo = min(mejor_costo, costo)
        evaluadas += 1
    return mejor_costo, evaluadas, (time.perf_counter() - t0) * 1000, agotado


def tabla_zonificacion_exacta(red, dist, tamanos=(5, 6, 7), semilla=SEED):
    rng_c = np.random.default_rng(semilla + 7)
    idx_base_c = list(rng_c.choice(range(1, len(red)), size=max(tamanos), replace=False))
    filas_c = []
    for n in tamanos:
        costo, evald, tms, agotado = zonificacion_exacta(idx_base_c[:n], dist, n_cuadrillas=2)
        filas_c.append({
            "n puntos": n, "cuadrillas": 2, "particiones (2^n)": 2 ** n,
            "evaluadas": evald, "mejor costo": round(costo, 2), "tiempo (ms)": round(tms, 1),
            "estado": "completo" if not agotado else f"timeout >{TIMEOUT_EXACTO_S:.0f}s",
        })
    return pd.DataFrame(filas_c)


def estudio_escalabilidad_zonas(tamanos=(60, 200, 500, 1000), n_cuadrillas=4, max_iter_test=150):
    filas = []
    for n in tamanos:
        red_n = generar_red_logistica(n_puntos=n, n_cuadrillas=n_cuadrillas)
        dist_n = matriz_distancias(red_n)
        prob = ProblemaZonas(red_n, dist_n, n_cuadrillas)
        random.seed(SEED); np.random.seed(SEED)
        t0 = time.perf_counter()
        simulated_annealing(prob, T0=500, alpha=0.999, Tmin=0.1, max_iter=max_iter_test)
        filas.append((n, (time.perf_counter() - t0) * 1000))
    tabla = pd.DataFrame(filas, columns=["n puntos", "tiempo (ms)"])
    coef = np.polyfit(tabla["n puntos"], tabla["tiempo (ms)"], 2)
    pred_1800 = float(np.polyval(coef, 1800))
    factor_calidad = 40000 / max_iter_test
    return tabla, coef, pred_1800, factor_calidad, max_iter_test


def mapa_zonas_grande(n_puntos=500, n_cuadrillas=4, max_iter=300):
    """Reproduce la visualizacion de zonas del notebook (celda 40)."""
    red_vis = generar_red_logistica(n_puntos=n_puntos, n_cuadrillas=n_cuadrillas)
    dist_vis = matriz_distancias(red_vis)
    prob_vis = ProblemaZonas(red_vis, dist_vis, n_cuadrillas)
    random.seed(SEED); np.random.seed(SEED)
    asig_vis = simulated_annealing(prob_vis, T0=500, alpha=0.999, Tmin=0.1, max_iter=max_iter)
    return red_vis, prob_vis, asig_vis


# ════════════════════════════════════════════════════════════════════
# FRENTE 3 — REABASTECIMIENTO (MDP)
# ════════════════════════════════════════════════════════════════════

class MDPReabastecimiento:
    """Inventario SIN vida util (modelo de comparacion)."""
    def __init__(self, demanda_punto, inv_max=20, precio=5.0, costo_unit=2.0,
                 costo_merma=3.0, costo_agotado=4.0, gamma=0.95):
        self.inv_max = inv_max
        self.precio = precio
        self.costo_unit = costo_unit
        self.costo_merma = costo_merma
        self.costo_agotado = costo_agotado
        self.gamma = gamma
        self.S = list(range(inv_max + 1))
        self.A = list(range(inv_max + 1))
        self.dist_demanda = self._estimar_demanda(demanda_punto)

    def _estimar_demanda(self, serie):
        serie = np.asarray(serie, dtype=float)
        escala = self.inv_max / (serie.max() + 1e-9)
        lotes = np.clip(np.round(serie * escala), 0, self.inv_max).astype(int)
        valores, conteos = np.unique(lotes, return_counts=True)
        probs = conteos / conteos.sum()
        return dict(zip(valores.tolist(), probs.tolist()))

    def recompensa(self, s, a, d):
        disponible = min(s + a, self.inv_max)
        vendido = min(disponible, d)
        sobrante = disponible - vendido
        faltante = max(d - disponible, 0)
        ingreso = self.precio * vendido
        costo_compra = self.costo_unit * a
        penal_merma = self.costo_merma * sobrante
        penal_agotado = self.costo_agotado * faltante
        return ingreso - costo_compra - penal_merma - penal_agotado

    def transicion(self, s, a):
        disponible = min(s + a, self.inv_max)
        resultados = {}
        for d, p in self.dist_demanda.items():
            sobrante = max(disponible - d, 0)
            s_sig = int(min(sobrante, self.inv_max))
            r = self.recompensa(s, a, d)
            if s_sig not in resultados:
                resultados[s_sig] = [0.0, 0.0]
            resultados[s_sig][0] += p
            resultados[s_sig][1] += p * r
        return {ss: (pr, rp / pr if pr > 0 else 0.0)
                for ss, (pr, rp) in resultados.items()}


class MDPReabastecimientoVidaUtil:
    """MDP PERECEDERO con vida util configurable (2 o 3 dias). Modelo base F3/F4."""
    def __init__(self, demanda_punto, inv_max=None, precio=5.0, costo_unit=2.0,
                 costo_merma=3.0, costo_agotado=4.0, gamma=0.95, vida_util=2):
        if vida_util not in (2, 3):
            raise ValueError("vida_util debe ser 2 o 3")
        self.vida_util = vida_util
        if inv_max is None:
            inv_max = 20 if vida_util == 2 else 15
        self.inv_max = inv_max
        self.precio = precio
        self.costo_unit = costo_unit
        self.costo_merma = costo_merma
        self.costo_agotado = costo_agotado
        self.gamma = gamma
        self.A = list(range(inv_max + 1))
        self.cod = {}
        self.dec = {}
        if vida_util == 2:
            for h in range(inv_max + 1):
                self.cod[h] = h
                self.dec[h] = h
        else:
            k = 0
            for h in range(inv_max + 1):
                for m in range(inv_max + 1 - h):
                    self.cod[(h, m)] = k
                    self.dec[k] = (h, m)
                    k += 1
        self.S = list(self.dec.keys())
        self.dist_demanda = self._estimar_demanda(demanda_punto)

    def _estimar_demanda(self, serie):
        serie = np.asarray(serie, dtype=float)
        escala = self.inv_max / (serie.max() + 1e-9)
        lotes = np.clip(np.round(serie * escala), 0, self.inv_max).astype(int)
        valores, conteos = np.unique(lotes, return_counts=True)
        probs = conteos / conteos.sum()
        return dict(zip(valores.tolist(), probs.tolist()))

    def _codificar(self, h, m=0):
        h = int(min(h, self.inv_max))
        if self.vida_util == 2:
            return self.cod[h]
        m = int(min(m, self.inv_max - h))
        return self.cod[(h, m)]

    def _dinamica(self, s_cod, a, d):
        a = int(a)
        fresco = int(min(a, self.inv_max))
        if self.vida_util == 2:
            viejo = int(min(self.dec[s_cod], self.inv_max))
            disponible = viejo + fresco
            vendido = min(disponible, d)
            vendido_viejo = min(viejo, vendido)
            vendido_fresco = vendido - vendido_viejo
            caduca = viejo - vendido_viejo
            sobra_fresco = fresco - vendido_fresco
            faltante = max(d - disponible, 0)
            s_sig = self._codificar(sobra_fresco)
        else:
            h, m = self.dec[s_cod]
            disponible = h + m + fresco
            vendido = min(disponible, d)
            vendido_h = min(h, vendido)
            resto = vendido - vendido_h
            vendido_m = min(m, resto)
            vendido_f = resto - vendido_m
            caduca = h - vendido_h
            nuevo_h = m - vendido_m
            nuevo_m = fresco - vendido_f
            faltante = max(d - disponible, 0)
            s_sig = self._codificar(nuevo_h, nuevo_m)
        r = (self.precio * vendido - self.costo_unit * a
             - self.costo_merma * caduca - self.costo_agotado * faltante)
        return s_sig, r, vendido, caduca, faltante

    def recompensa(self, s, a, d):
        return self._dinamica(s, a, d)[1]

    def transicion(self, s, a):
        resultados = {}
        for d, p in self.dist_demanda.items():
            s_sig, r, *_ = self._dinamica(s, a, d)
            if s_sig not in resultados:
                resultados[s_sig] = [0.0, 0.0]
            resultados[s_sig][0] += p
            resultados[s_sig][1] += p * r
        return {ss: (pr, rp / pr if pr > 0 else 0.0)
                for ss, (pr, rp) in resultados.items()}


def simular_politica_vidautil(mdp, politica, demanda_serie_lotes, s_inicial=0):
    s = s_inicial
    reg = {"ingreso": 0.0, "vendido": 0, "merma": 0, "agotado": 0, "demanda": 0}
    for d in demanda_serie_lotes:
        d = int(min(d, mdp.inv_max))
        a = politica[s]
        s_sig, r, vendido, caduca, faltante = mdp._dinamica(s, a, d)
        reg["ingreso"] += r
        reg["vendido"] += vendido
        reg["merma"] += caduca
        reg["agotado"] += faltante
        reg["demanda"] += d
        s = s_sig
    dem = max(reg["demanda"], 1)
    reg["nivel_servicio"] = reg["vendido"] / dem
    reg["tasa_agotado"] = reg["agotado"] / dem
    reg["tasa_merma"] = reg["merma"] / max(reg["vendido"] + reg["merma"], 1)
    return reg


def policy_evaluation(politica, mdp, epsilon=1e-4):
    V = {s: 0.0 for s in mdp.S}
    while True:
        delta = 0
        for s in mdp.S:
            v_anterior = V[s]
            a = politica[s]
            trans = mdp.transicion(s, a)
            V[s] = sum(prob * (r + mdp.gamma * V[s_sig])
                       for s_sig, (prob, r) in trans.items())
            delta = max(delta, abs(v_anterior - V[s]))
        if delta < epsilon:
            break
    return V


def policy_improvement(V, politica_actual, mdp):
    politica_nueva = {}
    politica_estable = True
    for s in mdp.S:
        q_valores = {}
        for a in mdp.A:
            trans = mdp.transicion(s, a)
            q_valores[a] = sum(prob * (r + mdp.gamma * V[s_sig])
                               for s_sig, (prob, r) in trans.items())
        mejor_accion = max(q_valores, key=q_valores.get)
        politica_nueva[s] = mejor_accion
        if mejor_accion != politica_actual[s]:
            politica_estable = False
    return politica_nueva, politica_estable


def policy_iteration(mdp, epsilon=1e-4):
    politica = {s: random.choice(mdp.A) for s in mdp.S}
    ciclos = 0
    while True:
        ciclos += 1
        V = policy_evaluation(politica, mdp, epsilon)
        politica, estable = policy_improvement(V, politica, mdp)
        if estable:
            break
    return politica, V, ciclos


def value_iteration(mdp, epsilon=1e-4):
    V = {s: 0.0 for s in mdp.S}
    n_iteraciones = 0
    while True:
        n_iteraciones += 1
        delta = 0
        for s in mdp.S:
            v_anterior = V[s]
            mejores = []
            for a in mdp.A:
                trans = mdp.transicion(s, a)
                mejores.append(sum(prob * (r + mdp.gamma * V[s_sig])
                                   for s_sig, (prob, r) in trans.items()))
            V[s] = max(mejores)
            delta = max(delta, abs(v_anterior - V[s]))
        if delta < epsilon:
            break
    politica = {}
    for s in mdp.S:
        q_valores = {}
        for a in mdp.A:
            trans = mdp.transicion(s, a)
            q_valores[a] = sum(prob * (r + mdp.gamma * V[s_sig])
                               for s_sig, (prob, r) in trans.items())
        politica[s] = max(q_valores, key=q_valores.get)
    return politica, V, n_iteraciones


def escalar_a_lotes(serie, ref, inv_max):
    escala = inv_max / (np.asarray(ref, dtype=float).max() + 1e-9)
    return np.clip(np.round(np.asarray(serie, dtype=float) * escala), 0, inv_max).astype(int)


def simular_politica(mdp, politica, demanda_serie_lotes, s_inicial=0):
    s = s_inicial
    registro = {"ingreso": 0.0, "vendido": 0, "merma": 0, "agotado": 0, "demanda": 0}
    for d in demanda_serie_lotes:
        d = int(min(d, mdp.inv_max))
        a = politica[s]
        disponible = min(s + a, mdp.inv_max)
        vendido = min(disponible, d)
        sobrante = disponible - vendido
        faltante = max(d - disponible, 0)
        registro["ingreso"] += mdp.recompensa(s, a, d)
        registro["vendido"] += vendido
        registro["merma"] += sobrante
        registro["agotado"] += faltante
        registro["demanda"] += d
        s = int(min(sobrante, mdp.inv_max))
    dem = max(registro["demanda"], 1)
    registro["nivel_servicio"] = registro["vendido"] / dem
    registro["tasa_agotado"] = registro["agotado"] / dem
    registro["tasa_merma"] = registro["merma"] / max(registro["vendido"] + registro["merma"], 1)
    return registro


def politica_heuristica(mdp):
    demanda_media = sum(d * p for d, p in mdp.dist_demanda.items())
    pedido_fijo = int(round(demanda_media))
    return {s: pedido_fijo for s in mdp.S}


def experimento_frente3(demanda, punto=0, vida_util=3):
    serie = demanda[:, punto]
    train, test = serie[:120], serie[120:]
    test_lotes = escalar_a_lotes(test, train, 15)

    mdp = MDPReabastecimientoVidaUtil(train, vida_util=vida_util)
    random.seed(SEED); np.random.seed(SEED)
    pi_opt, V, ciclos = policy_iteration(mdp)
    pi_heur = politica_heuristica(mdp)
    r_opt = simular_politica_vidautil(mdp, pi_opt, test_lotes)
    r_heur = simular_politica_vidautil(mdp, pi_heur, test_lotes)

    mdp_np = MDPReabastecimiento(train, inv_max=mdp.inv_max)
    random.seed(SEED); np.random.seed(SEED)
    pi_np, V_np, _ = policy_iteration(mdp_np)
    r_np_pred = simular_politica(mdp_np, pi_np, test_lotes)
    pi_np_real = {s: pi_np[min(sum(mdp.dec[s]), mdp_np.inv_max)] for s in mdp.S}
    r_np_real = simular_politica_vidautil(mdp, pi_np_real, test_lotes)

    tabla = pd.DataFrame([
        ("Perecedero 3d - Policy Iteration (optima, real)", r_opt["ingreso"], r_opt["nivel_servicio"], r_opt["tasa_agotado"], r_opt["tasa_merma"]),
        ("Perecedero 3d - Heuristica (nivel fijo, real)", r_heur["ingreso"], r_heur["nivel_servicio"], r_heur["tasa_agotado"], r_heur["tasa_merma"]),
        ("Sin vida util - lo que PREDICE su marco", r_np_pred["ingreso"], r_np_pred["nivel_servicio"], r_np_pred["tasa_agotado"], r_np_pred["tasa_merma"]),
        ("Sin vida util - REAL en mundo perecedero", r_np_real["ingreso"], r_np_real["nivel_servicio"], r_np_real["tasa_agotado"], r_np_real["tasa_merma"]),
    ], columns=["Politica / modelo", "Rentabilidad", "Nivel servicio", "Tasa agotados", "Tasa merma"])
    return tabla, pi_opt, V, ciclos, mdp, mdp_np


# ════════════════════════════════════════════════════════════════════
# FRENTE 4 — POLITICA DE INVENTARIO A LARGO PLAZO
# ════════════════════════════════════════════════════════════════════

def sensibilidad_gamma(demanda, punto=0, gammas=(0.5, 0.8, 0.9, 0.95, 0.99)):
    train = demanda[:120, punto]
    test = demanda[120:, punto]
    filas = []
    for g in gammas:
        mdp = MDPReabastecimiento(train, gamma=g)
        random.seed(SEED); np.random.seed(SEED)
        pi, V, ciclos = policy_iteration(mdp)
        test_lotes = escalar_a_lotes(test, train, mdp.inv_max)
        r = simular_politica(mdp, pi, test_lotes)
        pedido_medio = np.mean([pi[s] for s in mdp.S])
        filas.append((g, pedido_medio, r["nivel_servicio"], r["tasa_merma"], r["ingreso"]))
    return pd.DataFrame(filas, columns=["gamma", "Pedido medio", "Nivel servicio", "Tasa merma", "Rentabilidad"])


def sensibilidad_gamma_vidautil(demanda, punto=0, gammas=(0.5, 0.8, 0.9, 0.95, 0.99), vida_util=3):
    train = demanda[:120, punto]; test = demanda[120:, punto]
    filas = []; politicas = {}
    for gm in gammas:
        random.seed(SEED); np.random.seed(SEED)
        mdp = MDPReabastecimientoVidaUtil(train, gamma=gm, vida_util=vida_util)
        pi, V, ciclos = policy_iteration(mdp)
        test_lotes = escalar_a_lotes(test, train, mdp.inv_max)
        r = simular_politica_vidautil(mdp, pi, test_lotes)
        politicas[gm] = tuple(pi[s] for s in mdp.S)
        filas.append((gm, round(float(np.mean([pi[s] for s in mdp.S])), 3),
                      round(r["nivel_servicio"], 4), round(r["tasa_merma"], 4),
                      round(r["ingreso"], 1)))
    tabla = pd.DataFrame(filas, columns=["gamma", "Pedido medio", "Nivel servicio", "Tasa merma", "Rentabilidad"])
    return tabla, politicas


def sensibilidad_gamma_vidautil_vi(demanda, punto=0, gammas=(0.5, 0.8, 0.9, 0.95, 0.99), vida_util=3):
    train = demanda[:120, punto]; test = demanda[120:, punto]
    filas = []; politicas = {}
    for gm in gammas:
        random.seed(SEED); np.random.seed(SEED)
        mdp = MDPReabastecimientoVidaUtil(train, gamma=gm, vida_util=vida_util)
        pi, V, iters = value_iteration(mdp)
        test_lotes = escalar_a_lotes(test, train, mdp.inv_max)
        r = simular_politica_vidautil(mdp, pi, test_lotes)
        politicas[gm] = tuple(pi[s] for s in mdp.S)
        filas.append((gm, round(float(np.mean([pi[s] for s in mdp.S])), 3),
                      round(r["nivel_servicio"], 4), round(r["tasa_merma"], 4),
                      round(r["ingreso"], 1)))
    tabla = pd.DataFrame(filas, columns=["gamma", "Pedido medio", "Nivel servicio", "Tasa merma", "Rentabilidad"])
    return tabla, politicas


def comparar_pi_vi(demanda, punto=0, gamma=0.95, vida_util=3):
    random.seed(SEED); np.random.seed(SEED)
    mdp_cmp = MDPReabastecimientoVidaUtil(demanda[:120, punto], gamma=gamma, vida_util=vida_util)
    random.seed(SEED); np.random.seed(SEED)
    pi_pi, _, ciclos_pi = policy_iteration(mdp_cmp)
    random.seed(SEED); np.random.seed(SEED)
    pi_vi, _, iters_vi = value_iteration(mdp_cmp)
    coinciden = all(pi_pi[s] == pi_vi[s] for s in mdp_cmp.S)
    return ciclos_pi, iters_vi, coinciden


def construir_escenarios(demanda, punto=0):
    serie = demanda[:, punto].astype(float)
    n = len(serie)
    dias = np.arange(n)
    es_quincena = (dias % 15 == 0) | (dias % 30 == 0)
    umbral_alta = np.quantile(serie, 0.80)
    escenarios = {
        "ordinario": serie[(~es_quincena) & (serie < umbral_alta)],
        "alta_demanda": serie[(serie >= umbral_alta) | es_quincena],
        "perturbacion": serie * np.where(np.random.default_rng(SEED + 50).random(n) < 0.15, 1.5, 1.0),
    }
    dificultad = {"ordinario": "baja", "alta_demanda": "media", "perturbacion": "alta"}
    resumen = pd.DataFrame([
        (k, dificultad[k], len(v), round(float(np.mean(v)), 1), round(float(np.std(v)), 1))
        for k, v in escenarios.items()
    ], columns=["Escenario", "Dificultad", "N dias", "Demanda media", "Desv. estandar"])
    return escenarios, resumen


PERFILES_F4 = {
    "alta_rotacion":      dict(precio=6.0, costo_agotado=5.0, costo_merma=2.0),
    "perecedero_critico": dict(precio=5.0, costo_agotado=3.0, costo_merma=4.0),
    "bajo_margen":        dict(precio=3.5, costo_agotado=2.0, costo_merma=2.5),
}


def robustez_politica_frente4(demanda, gamma=0.95, n_muestra_degradacion=5):
    n_puntos = demanda.shape[1]
    nombres = list(PERFILES_F4)
    rng = np.random.default_rng(SEED)
    perfil_de_punto = {p: nombres[int(rng.integers(0, len(nombres)))] for p in range(n_puntos)}
    puntos_por_perfil = {nom: [p for p in range(n_puntos) if perfil_de_punto[p] == nom]
                         for nom in nombres}

    politicas_por_perfil = {}
    mdp_por_perfil = {}
    ref_por_perfil = {}
    for nom in nombres:
        pts = puntos_por_perfil[nom]
        train_medio = demanda[:120, pts].mean(axis=1)
        random.seed(SEED); np.random.seed(SEED)
        mdp = MDPReabastecimientoVidaUtil(train_medio, gamma=gamma, vida_util=3, **PERFILES_F4[nom])
        pi, _, _ = policy_iteration(mdp)
        politicas_por_perfil[nom] = pi
        mdp_por_perfil[nom] = mdp
        ref_por_perfil[nom] = train_medio

    filas_perfil = []
    for nom in nombres:
        mdp = mdp_por_perfil[nom]
        pi = politicas_por_perfil[nom]; ref = ref_por_perfil[nom]
        ns, tm, rent = [], [], []
        for p in puntos_por_perfil[nom]:
            test_lotes = escalar_a_lotes(demanda[120:, p], ref, mdp.inv_max)
            r = simular_politica_vidautil(mdp, pi, test_lotes)
            ns.append(r["nivel_servicio"]); tm.append(r["tasa_merma"]); rent.append(r["ingreso"])
        filas_perfil.append((nom, len(puntos_por_perfil[nom]),
                             round(float(np.mean(ns)), 4), round(float(np.std(ns)), 4),
                             round(float(np.mean(tm)), 4), round(float(np.std(tm)), 4),
                             round(float(np.mean(rent)), 1), round(float(np.std(rent)), 1)))
    tabla_robustez_perfil = pd.DataFrame(filas_perfil, columns=[
        "Perfil", "N puntos", "Nivel serv. media", "Nivel serv. std",
        "Tasa merma media", "Tasa merma std", "Rentab. media", "Rentab. std"])

    degradaciones = []
    for nom in nombres:
        for p in puntos_por_perfil[nom][:n_muestra_degradacion]:
            train_p = demanda[:120, p]
            random.seed(SEED); np.random.seed(SEED)
            mdp_p = MDPReabastecimientoVidaUtil(train_p, gamma=gamma, vida_util=3, **PERFILES_F4[nom])
            pi_p, _, _ = policy_iteration(mdp_p)
            test_lotes_p = escalar_a_lotes(demanda[120:, p], train_p, mdp_p.inv_max)
            r_opt = simular_politica_vidautil(mdp_p, pi_p, test_lotes_p)
            r_perfil = simular_politica_vidautil(mdp_p, politicas_por_perfil[nom], test_lotes_p)
            if r_opt["ingreso"] > 0:
                degradaciones.append((r_opt["ingreso"] - r_perfil["ingreso"]) / r_opt["ingreso"] * 100)
    degradacion_media = float(np.mean(degradaciones)) if degradaciones else 0.0

    filas_esc = []
    for nom in nombres:
        p_repr = puntos_por_perfil[nom][0]
        escenarios, _ = construir_escenarios(demanda, punto=p_repr)
        mdp = mdp_por_perfil[nom]; pi = politicas_por_perfil[nom]; ref = ref_por_perfil[nom]
        for esc_nom, serie_esc in escenarios.items():
            test_lotes = escalar_a_lotes(serie_esc, ref, mdp.inv_max)
            r = simular_politica_vidautil(mdp, pi, test_lotes)
            filas_esc.append((nom, esc_nom, round(r["nivel_servicio"], 4), round(r["ingreso"], 1)))
    tabla_robustez_escenarios = pd.DataFrame(filas_esc, columns=[
        "Perfil", "Escenario", "Nivel servicio", "Rentabilidad"])

    return (tabla_robustez_perfil, degradacion_media, tabla_robustez_escenarios,
            {k: len(v) for k, v in puntos_por_perfil.items()})


# ════════════════════════════════════════════════════════════════════
# RESULTADOS INTEGRADOS — TABLERO KPI vs OKR
# ════════════════════════════════════════════════════════════════════

def tablero_kpi(tabla_f2, tabla_f3):
    dist_actual = tabla_f2.loc[0, "Costo total red"]
    dist_optim = tabla_f2.loc[2, "Costo total red"]
    reduccion_dist = (1 - dist_optim / dist_actual) * 100
    r_opt = tabla_f3[tabla_f3["Politica / modelo"].str.contains("Policy")].iloc[0]
    r_heur = tabla_f3[tabla_f3["Politica / modelo"].str.contains("Heur")].iloc[0]
    filas = [
        ("DPV — Distancia por vehiculo", f"{dist_actual:.0f}", f"{dist_optim:.0f}", f"-{reduccion_dist:.1f}%", "OKR1: -15%"),
        ("NS — Nivel de servicio", f"{r_heur['Nivel servicio']:.0%}", f"{r_opt['Nivel servicio']:.0%}", "↑", "OKR2: ≥91%"),
        ("TA — Tasa de agotados", f"{r_heur['Tasa agotados']:.0%}", f"{r_opt['Tasa agotados']:.0%}", "↓", "OKR2: ≤5%"),
        ("TMP — Tasa de merma", f"{r_heur['Tasa merma']:.0%}", f"{r_opt['Tasa merma']:.0%}", "↓", "OKR3: -35%"),
        ("Rentabilidad acumulada", f"{r_heur['Rentabilidad']:.0f}", f"{r_opt['Rentabilidad']:.0f}",
         f"+{(r_opt['Rentabilidad']/max(r_heur['Rentabilidad'],1)-1)*100:.0f}%", "OKR3: +12%"),
    ]
    return pd.DataFrame(filas, columns=["KPI", "Linea base", "Con solucion", "Variacion", "Meta OKR"])


# ════════════════════════════════════════════════════════════════════
# VALIDACION ESTADISTICA (seccion 7)
# ════════════════════════════════════════════════════════════════════

def intervalo_confianza_95(muestras):
    m = np.mean(muestras)
    s = np.std(muestras, ddof=1) if len(muestras) > 1 else 0.0
    margen = 1.96 * s / math.sqrt(len(muestras)) if len(muestras) > 1 else 0.0
    return m, s, (m - margen, m + margen)


def repetir_sa_ruta(indices, dist, n_semillas=15):
    costos, tiempos = [], []
    for k in range(n_semillas):
        random.seed(SEED + k); np.random.seed(SEED + k)
        prob = ProblemaRuta(indices, dist)
        t0 = time.perf_counter()
        orden = simulated_annealing(prob)
        tiempos.append((time.perf_counter() - t0) * 1000)
        costos.append(prob.costo_ruta(orden))
    random.seed(SEED); np.random.seed(SEED)
    return np.array(costos), np.array(tiempos)


def experimento_multisemilla_f1(red, dist, n_exacto=7, n_semillas=15, semilla=SEED):
    rng = np.random.default_rng(semilla + 5)
    idx = list(rng.choice(range(1, len(red)), size=n_exacto, replace=False))
    costos_sa, tiempos_sa = repetir_sa_ruta(idx, dist, n_semillas)
    _, costo_a = a_estrella_ruta(idx, dist)
    _, costo_ida = ida_estrella_ruta(idx, dist)
    m_sa, s_sa, ic_sa = intervalo_confianza_95(costos_sa)
    tabla = pd.DataFrame([
        ("Simulated Annealing", round(m_sa, 2), round(s_sa, 2), f"[{ic_sa[0]:.1f}, {ic_sa[1]:.1f}]", round(costos_sa.min(), 2)),
        ("A* (exacto)", round(costo_a, 2), 0.0, "-", round(costo_a, 2)),
        ("IDA* (exacto)", round(costo_ida, 2), 0.0, "-", round(costo_ida, 2)),
    ], columns=["Algoritmo", "Media (km)", "Desv. est.", "IC 95%", "Mejor (km)"])
    return tabla, costos_sa, costo_a, idx


def repetir_sa_zonas(red, dist, n_cuadrillas=4, n_semillas=10, config="lento"):
    params = {"lento": dict(T0=500, alpha=0.999, Tmin=0.1, max_iter=40000),
              "rapido": dict(T0=10, alpha=0.95, Tmin=0.1, max_iter=5000)}[config]
    costos = []
    prob = ProblemaZonas(red, dist, n_cuadrillas)
    for k in range(n_semillas):
        random.seed(SEED + k); np.random.seed(SEED + k)
        asign = simulated_annealing(prob, **params)
        costos.append(prob.costo_total(asign))
    random.seed(SEED); np.random.seed(SEED)
    return np.array(costos)


def comparar_configuraciones_sa(indices, dist, n_semillas=15):
    configs = {
        "alpha=0.99": dict(T0=100, alpha=0.99, Tmin=0.01),
        "alpha=0.995": dict(T0=100, alpha=0.995, Tmin=0.01),
        "alpha=0.999": dict(T0=100, alpha=0.999, Tmin=0.01),
    }
    resultados = {nombre: [] for nombre in configs}
    prob = ProblemaRuta(indices, dist)
    for k in range(n_semillas):
        for nombre, params in configs.items():
            random.seed(SEED + k); np.random.seed(SEED + k)
            orden = simulated_annealing(prob, **params)
            resultados[nombre].append(prob.costo_ruta(orden))
    random.seed(SEED); np.random.seed(SEED)
    return {k: np.array(v) for k, v in resultados.items()}


def validacion_estadistica(red, dist):
    """Wilcoxon (F2 lento vs rapido) + Friedman (F1 tres enfriamientos)."""
    from scipy.stats import wilcoxon, friedmanchisquare
    costos_lento = repetir_sa_zonas(red, dist, config="lento", n_semillas=10)
    costos_rapido = repetir_sa_zonas(red, dist, config="rapido", n_semillas=10)
    m_l, s_l, ic_l = intervalo_confianza_95(costos_lento)
    m_r, s_r, ic_r = intervalo_confianza_95(costos_rapido)
    tabla_ms_f2 = pd.DataFrame([
        ("SA enfriamiento rapido", round(m_r, 2), round(s_r, 2), f"[{ic_r[0]:.1f}, {ic_r[1]:.1f}]", round(costos_rapido.min(), 2)),
        ("SA T alta + enfriamiento lento", round(m_l, 2), round(s_l, 2), f"[{ic_l[0]:.1f}, {ic_l[1]:.1f}]", round(costos_lento.min(), 2)),
    ], columns=["Configuracion", "Media costo", "Desv. est.", "IC 95%", "Mejor"])
    stat_w, p_w = wilcoxon(costos_lento, costos_rapido)

    tabla_ms_f1, costos_sa_f1, costo_opt_f1, idx_ms_f1 = experimento_multisemilla_f1(red, dist)
    res_sa = comparar_configuraciones_sa(idx_ms_f1, dist)
    stat_f, p_f = friedmanchisquare(*res_sa.values())
    tabla_friedman = pd.DataFrame([
        (nombre, round(v.mean(), 2), round(v.std(ddof=1), 2)) for nombre, v in res_sa.items()
    ], columns=["Config. enfriamiento", "Costo medio", "Desv. est."])

    brecha = (costos_sa_f1.mean() / costo_opt_f1 - 1) * 100
    n_opt = int((np.abs(costos_sa_f1 - costo_opt_f1) < 1e-6).sum())
    return {
        "tabla_ms_f1": tabla_ms_f1, "costos_sa_f1": costos_sa_f1, "costo_opt_f1": costo_opt_f1,
        "brecha": brecha, "n_opt": n_opt, "n_corridas": len(costos_sa_f1),
        "tabla_ms_f2": tabla_ms_f2, "mejora_lento": (1 - m_l / m_r) * 100,
        "wilcoxon": (stat_w, p_w), "friedman": (stat_f, p_f), "tabla_friedman": tabla_friedman,
    }


# ════════════════════════════════════════════════════════════════════
# METRICAS OPERATIVAS COMPLEMENTARIAS (seccion 8)
# ════════════════════════════════════════════════════════════════════

CONSUMO_L_POR_KM = 0.12
PRECIO_COMBUSTIBLE = 16000.0
VELOCIDAD_KMH = 22.0
SERVICIO_MIN = 4.0
HORA_SALIDA = 4.0
COSTO_FIJO_VEHICULO = 85000.0


def combustible_ruta(distancia_km, consumo=CONSUMO_L_POR_KM, precio=PRECIO_COMBUSTIBLE):
    litros = distancia_km * consumo
    return litros, litros * precio


def comparacion_combustible(idx_grande, dist):
    prob_demo = ProblemaRuta(idx_grande, dist)
    random.seed(SEED)
    costos_aleatorios = [prob_demo.costo_ruta(prob_demo.estado_aleatorio()) for _ in range(30)]
    dist_sin_opt = np.mean(costos_aleatorios)
    random.seed(SEED); np.random.seed(SEED)
    orden_opt = simulated_annealing(prob_demo)
    dist_opt = prob_demo.costo_ruta(orden_opt)
    lit_sin, cop_sin = combustible_ruta(dist_sin_opt)
    lit_opt, cop_opt = combustible_ruta(dist_opt)
    tabla = pd.DataFrame([
        ("Sin optimizar (orden manual ~aleatorio)", round(dist_sin_opt, 1), round(lit_sin, 2), f"{cop_sin:,.0f}"),
        ("Optimizada (Simulated Annealing)", round(dist_opt, 1), round(lit_opt, 2), f"{cop_opt:,.0f}"),
    ], columns=["Recorrido", "Distancia (km)", "Combustible (L)", "Costo (COP)"])
    ahorro = (1 - dist_opt / dist_sin_opt) * 100
    return tabla, ahorro, cop_sin - cop_opt


def simular_ventanas(orden, red, dist, deposito=0):
    ruta = [deposito] + list(orden)
    t = HORA_SALIDA
    en_ventana = 0
    for i in range(1, len(ruta)):
        t += (dist[ruta[i - 1], ruta[i]] / VELOCIDAD_KMH) + (SERVICIO_MIN / 60.0)
        ini = red["ventana_ini"].iloc[ruta[i]]
        fin = red["ventana_fin"].iloc[ruta[i]]
        if ini <= t <= fin:
            en_ventana += 1
    return en_ventana / (len(ruta) - 1) * 100


def cppa(distancia_km, n_puntos):
    litros, costo_comb = combustible_ruta(distancia_km)
    costo_total = costo_comb + COSTO_FIJO_VEHICULO
    return costo_total / max(n_puntos, 1), costo_total


def metricas_cppa_pedvh(idx_grande, red, dist, n_veh=12):
    idx_veh = idx_grande[:n_veh]
    prob_veh = ProblemaRuta(idx_veh, dist)
    random.seed(SEED); np.random.seed(SEED)
    orden_veh_opt = simulated_annealing(prob_veh)
    dist_veh_opt = prob_veh.costo_ruta(orden_veh_opt)
    random.seed(SEED)
    ordenes_manual = [prob_veh.estado_aleatorio() for _ in range(30)]
    dist_veh_sin = np.mean([prob_veh.costo_ruta(o) for o in ordenes_manual])
    pedvh_opt = simular_ventanas(orden_veh_opt, red, dist)
    pedvh_sin = np.mean([simular_ventanas(o, red, dist) for o in ordenes_manual])
    cppa_opt, _ = cppa(dist_veh_opt, len(idx_veh))
    cppa_sin, _ = cppa(dist_veh_sin, len(idx_veh))
    tabla = pd.DataFrame([
        ("Sin optimizar (orden manual)", f"{cppa_sin:,.0f}", f"{pedvh_sin:.1f}%"),
        ("Optimizada (SA)", f"{cppa_opt:,.0f}", f"{pedvh_opt:.1f}%"),
    ], columns=["Recorrido", "CPPA (COP/punto)", "PEDVH"])
    return tabla, (1 - cppa_opt / cppa_sin) * 100, pedvh_sin, pedvh_opt, len(idx_veh)
