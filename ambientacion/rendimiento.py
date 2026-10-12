"""Mapas de rendimiento (SHP de puntos ya limpios) → grilla de 10 m.

Cada campaña se interpola (kriging ordinario, con IDW como respaldo) y se pasa
a rendimiento relativo (rinde / media del lote). Con varias campañas se calcula
el promedio y el CV temporal del rendimiento relativo, igual que con los índices.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .entradas import leer_vectorial


def _centros(grilla):
    filas, cols = np.nonzero(grilla.mascara_borde)
    t = grilla.transform
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (filas + 0.5) * t.e
    return filas, cols, x, y


def _idw(xp, yp, zp, x, y, k=16, potencia=2.0):
    arbol = cKDTree(np.c_[xp, yp])
    d, i = arbol.query(np.c_[x, y], k=min(k, len(zp)))
    d = np.maximum(d, 1e-6)
    w = 1.0 / d ** potencia
    return (w * zp[i]).sum(axis=1) / w.sum(axis=1)


def _kriging(xp, yp, zp, x, y, cfg):
    from pykrige.ok import OrdinaryKriging
    rng = np.random.default_rng(cfg.semilla)
    if len(zp) > cfg.max_puntos_kriging:
        sel = rng.choice(len(zp), cfg.max_puntos_kriging, replace=False)
        xp, yp, zp = xp[sel], yp[sel], zp[sel]
    ok = OrdinaryKriging(xp, yp, zp, variogram_model="spherical", nlags=20)
    z, _ = ok.execute("points", x, y, backend="C", n_closest_points=48)
    return np.asarray(z)


def interpolar_campaña(path, columna, lote_utm, grilla, cfg):
    gdf = leer_vectorial(path).to_crs(lote_utm.crs)
    if columna not in gdf.columns:
        raise ValueError(f"La columna '{columna}' no existe en {os.path.basename(path)}. "
                         f"Columnas: {list(gdf.columns)}")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.centroid
    gdf = gdf[gdf.within(lote_utm.buffer(grilla.res).iloc[0])]
    z = pd.to_numeric(gdf[columna], errors="coerce").to_numpy()
    ok = np.isfinite(z)
    xp, yp, z = gdf.geometry.x.to_numpy()[ok], gdf.geometry.y.to_numpy()[ok], z[ok]
    if len(z) < 30:
        raise ValueError(f"{os.path.basename(path)}: muy pocos puntos válidos ({len(z)}).")
    filas, cols, x, y = _centros(grilla)
    metodo = cfg.interpolacion
    try:
        zi = _kriging(xp, yp, z, x, y, cfg) if metodo == "kriging" else _idw(xp, yp, z, x, y)
    except Exception as e:  # noqa: BLE001
        print(f"  Kriging falló ({e}); se usa IDW.")
        metodo = "idw"
        zi = _idw(xp, yp, z, x, y)
    capa = np.full((grilla.alto, grilla.ancho), np.nan, dtype="float32")
    capa[filas, cols] = zi
    resumen = {"n_puntos": int(len(z)), "media": float(np.mean(z)), "desvio": float(np.std(z)),
               "min": float(np.min(z)), "max": float(np.max(z)), "metodo": metodo}
    return capa, resumen


def procesar_rendimiento(tabla_rend, carpeta, lote_utm, grilla, cfg):
    """Devuelve (capas, tabla_resumen). Capas: rend_media (magnitud, si hay una
    campaña) / rend_rel (relativo medio, %) / rend_cv (%, si hay >1 campaña)."""
    if tabla_rend is None or tabla_rend.empty:
        return {}, None
    relativos, filas = [], []
    absolutas = []
    for _, r in tabla_rend.iterrows():
        path = os.path.join(carpeta, str(r["archivo"])) if carpeta else str(r["archivo"])
        capa, res = interpolar_campaña(path, str(r["columna"]), lote_utm, grilla, cfg)
        media_lote = np.nanmean(capa[grilla.mascara])
        relativos.append(capa / media_lote)
        absolutas.append(capa)
        filas.append({"archivo": r["archivo"], "campaña": r["campaña"], "cultivo": r["cultivo"],
                      "columna": r["columna"], **res})
    rel = np.stack(relativos)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
    capas = {"rend_rel": 100 * np.nanmean(rel, axis=0)}
    if len(relativos) == 1:
        capas["rend_media"] = absolutas[0]
    else:
        capas["rend_cv"] = 100 * np.nanstd(rel, axis=0, ddof=1) / np.nanmean(rel, axis=0)
    return capas, pd.DataFrame(filas)
