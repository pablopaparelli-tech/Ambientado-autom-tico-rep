"""K-means con selección automática de k (codo, silueta, Calinski-Harabasz,
Davies-Bouldin) por votación, con restricción de superficie mínima por ambiente."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


@dataclass
class DatosCluster:
    claves: list
    X: np.ndarray          # (n_pixeles, n_variables) normalizado y filtrado
    filas: np.ndarray
    cols: np.ndarray
    forma: tuple


def preparar(stack, claves, cfg) -> DatosCluster:
    m = stack.grilla.mascara
    capas = []
    for k in claves:
        a = stack.norm[k].copy()
        if cfg.filtro_mediana_px and cfg.filtro_mediana_px > 1:
            relleno = np.where(np.isfinite(a), a, np.nanmean(a[m]))
            a = np.where(m, ndimage.median_filter(relleno, size=cfg.filtro_mediana_px), np.nan)
        capas.append(a)
    pila = np.stack(capas, axis=-1)
    validos = m & np.all(np.isfinite(pila), axis=-1)
    filas, cols = np.nonzero(validos)
    return DatosCluster(list(claves), pila[filas, cols], filas, cols, m.shape)


def _codo(ks, inercias):
    """k con máxima distancia a la recta entre extremos de la curva de inercia."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inercias) - inercias[-1]) / (inercias[0] - inercias[-1] + 1e-12)
    d = np.abs(y - (1 - x)) / np.sqrt(2)
    return ks[int(np.argmax(d))]


def evaluar_k(datos: DatosCluster, cfg, muestra=10000):
    rng = np.random.default_rng(cfg.semilla)
    n = len(datos.X)
    idx = rng.choice(n, min(muestra, n), replace=False)
    filas = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.semilla).fit(datos.X)
        lab = km.labels_
        area_min = 100 * np.bincount(lab, minlength=k).min() / n
        filas.append({
            "k": k, "inercia": km.inertia_,
            "silueta": silhouette_score(datos.X[idx], lab[idx]),
            "calinski_harabasz": calinski_harabasz_score(datos.X[idx], lab[idx]),
            "davies_bouldin": davies_bouldin_score(datos.X[idx], lab[idx]),
            "ambiente_min_pct": area_min,
        })
    df = pd.DataFrame(filas).set_index("k")
    ks = list(df.index)
    df["cumple_area"] = df["ambiente_min_pct"] >= cfg.area_min_ambiente_pct
    validos = df[df["cumple_area"]]
    if validos.empty:
        validos = df
    votos = {
        "codo": _codo(ks, df["inercia"].tolist()),
        "silueta": int(validos["silueta"].idxmax()),
        "calinski_harabasz": int(validos["calinski_harabasz"].idxmax()),
        "davies_bouldin": int(validos["davies_bouldin"].idxmin()),
    }
    if votos["codo"] not in validos.index:
        votos["codo"] = int(validos.index[np.argmin(np.abs(validos.index - votos["codo"]))])
    conteo = Counter(votos.values()).most_common()
    maximo = conteo[0][1]
    empatados = [k for k, c in conteo if c == maximo]
    k_sugerido = votos["silueta"] if votos["silueta"] in empatados else min(empatados)
    return df, votos, k_sugerido


def grafico_k(df, votos, k_sugerido):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    metricas = [("inercia", "Codo (inercia)", "codo"), ("silueta", "Silueta ↑", "silueta"),
                ("calinski_harabasz", "Calinski-Harabasz ↑", "calinski_harabasz"),
                ("davies_bouldin", "Davies-Bouldin ↓", "davies_bouldin")]
    for ax, (col, titulo, voto) in zip(axs, metricas):
        ax.plot(df.index, df[col], "o-", color="#4c72b0")
        ax.axvline(votos[voto], color="#dd8452", ls="--", lw=1, label=f"voto: k={votos[voto]}")
        ax.axvline(k_sugerido, color="#2ca02c", lw=2, alpha=0.4, label=f"elegido: k={k_sugerido}")
        no = df[~df["cumple_area"]]
        ax.plot(no.index, no[col], "x", color="red", ms=9)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("k")
        ax.legend(fontsize=7)
    fig.suptitle("Selección del número de ambientes (× = algún ambiente bajo la superficie mínima)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def ajustar(datos: DatosCluster, k, cfg):
    km = KMeans(n_clusters=k, n_init=20, random_state=cfg.semilla).fit(datos.X)
    raster = np.zeros(datos.forma, dtype="int16")  # 0 = sin dato
    raster[datos.filas, datos.cols] = km.labels_ + 1
    return km, raster
