"""Ranking (1 = mayor potencial), nombres, colores y estadísticas por ambiente."""
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import to_hex

from .variables import VARIABLES

NOMBRES_BASE = {
    1: ["Media"],
    2: ["Alta", "Baja"],
    3: ["Alta", "Media", "Baja"],
    4: ["Alta", "Media", "Baja", "Muy Baja"],
}
VARS_POTENCIAL = ("ndvi_rel", "cire_rel")
VARS_DESCRIPTORAS = ("tpi", "anegamiento", "dem", "ndmi_rel", "ndvi_cv", "cire_cv", "ndmi_cv", "rend_cv")


def indice_potencial(stack):
    """Rendimiento relativo si hay mapa de rinde; si no, media de NDVI y CIre normalizados."""
    if "rend_rel" in stack.norm:
        return stack.norm["rend_rel"], "rendimiento relativo"
    capas = [stack.norm[k] for k in VARS_POTENCIAL if k in stack.norm]
    if not capas:
        capas = [stack.norm[k] for k in stack.norm]
        return np.nanmean(np.stack(capas), axis=0), "media de variables normalizadas"
    return np.nanmean(np.stack(capas), axis=0), "media de NDVI y CIre normalizados"


def rankear(raster, stack):
    """Renumera los clusters: 1 = mayor potencial productivo."""
    pot, criterio = indice_potencial(stack)
    ids = [i for i in np.unique(raster) if i > 0]
    medias = {i: np.nanmean(pot[raster == i]) for i in ids}
    orden = sorted(ids, key=lambda i: medias[i], reverse=True)
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden, start=1)}
    nuevo = np.zeros_like(raster)
    for viejo, n in mapa.items():
        nuevo[raster == viejo] = n
    potencial = pd.Series({mapa[i]: medias[i] for i in ids}).sort_index()
    sd = float(np.nanstd(pot[stack.grilla.mascara]))
    return nuevo, potencial, sd, criterio


def _grupos(potencial: pd.Series, sd, umbral_sd):
    """Agrupa ambientes consecutivos cuyo potencial difiere menos de umbral·sd."""
    ids = list(potencial.index)
    grupos = [[ids[0]]]
    for a, b in zip(ids, ids[1:]):
        if potencial[a] - potencial[b] < umbral_sd * sd:
            grupos[-1].append(b)
        else:
            grupos.append([b])
    while len(grupos) > 4:
        brechas = [potencial[g1[-1]] - potencial[g2[0]] for g1, g2 in zip(grupos, grupos[1:])]
        i = int(np.argmin(brechas))
        grupos[i] = grupos[i] + grupos.pop(i + 1)
    return grupos


def nombrar(raster, stack, potencial, sd, cfg):
    medias = {}
    candidatas = [k for k in VARS_DESCRIPTORAS if k in stack.norm]
    for amb in potencial.index:
        medias[amb] = {k: float(np.nanmean(stack.norm[k][raster == amb])) for k in candidatas}
    medias = pd.DataFrame(medias).T  # filas: ambiente, columnas: variable

    grupos = _grupos(potencial, sd, cfg.umbral_similitud_sd)
    base = NOMBRES_BASE[len(grupos)]
    nombres = {}
    for g, nombre in zip(grupos, base):
        if len(g) == 1 or medias.empty:
            for i, amb in enumerate(g):
                nombres[amb] = nombre if len(g) == 1 else f"{nombre} {i + 1}"
            continue
        sub = medias.loc[g]
        principal = (sub.max() - sub.min()).idxmax()
        alto, bajo = VARIABLES[principal].descriptor
        orden = sub[principal].sort_values(ascending=False).index.tolist()
        nombres[orden[0]] = f"{nombre} ({alto})"
        nombres[orden[-1]] = f"{nombre} ({bajo})"
        for amb in orden[1:-1]:
            otras = (sub.drop(columns=principal).loc[amb] - sub.drop(columns=principal).mean())
            var = otras.abs().idxmax() if not otras.empty else principal
            pal = VARIABLES[var].descriptor[0 if otras.get(var, 0) >= 0 else 1]
            nombres[amb] = f"{nombre} ({pal})"
    # garantizar nombres únicos
    vistos = {}
    for amb in sorted(nombres):
        n = nombres[amb]
        if n in vistos:
            vistos[n] += 1
            nombres[amb] = f"{n} {vistos[n]}"
        else:
            vistos[n] = 1
    return nombres


def colores(k):
    cmap = matplotlib.colormaps["RdYlGn"]
    pos = np.linspace(0.92, 0.08, k) if k > 1 else [0.92]
    return {i + 1: to_hex(cmap(p)) for i, p in enumerate(pos)}


def _superficies(raster, stack, superficies):
    if superficies is None:
        n = {a: (raster == a).sum() * stack.cfg.pixel_ha for a in np.unique(raster) if a > 0}
        superficies = pd.Series(n)
    return superficies, superficies.sum()


def estadisticas(raster, stack, nombres, tabla_rend=None, superficies=None):
    """Tabla de estadísticas por ambiente en magnitudes reales (formato largo y ancho).
    `superficies`: ha por ambiente (de los polígonos finales)."""
    filas = []
    sup, total = _superficies(raster, stack, superficies)
    for amb in sorted(nombres):
        sel = raster == amb
        for k in stack.disponibles:
            v = VARIABLES[k]
            unidad, nombre = v.unidad, v.nombre
            if k == "rend_rel" and "rend_media" in stack.capas and tabla_rend is not None:
                unidad, nombre = f"{tabla_rend['columna'].iloc[0]}", "Rendimiento"
            a = stack.magnitud(k)[sel]
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
            filas.append({
                "ambiente_id": amb, "ambiente": nombres[amb],
                "superficie_ha": sup.get(amb, np.nan), "superficie_pct": 100 * sup.get(amb, np.nan) / total,
                "variable": nombre, "clave": k, "unidad": unidad,
                "media": a.mean(), "desvío": a.std(), "CV %": 100 * a.std() / abs(a.mean()) if a.mean() else np.nan,
                "mín": a.min(), "máx": a.max(),
            })
    largo = pd.DataFrame(filas)
    ancho = largo.pivot_table(index=["ambiente_id", "ambiente"], columns="variable", values="media")
    return largo, ancho


def descripciones(raster, stack, nombres, n_rasgos=4, superficies=None):
    """Texto automático que describe cada ambiente respecto del lote."""
    m = stack.grilla.mascara
    sup_amb, total = _superficies(raster, stack, superficies)
    textos = {}
    for amb in sorted(nombres):
        sel = raster == amb
        rasgos = []
        for k in stack.disponibles:
            v = VARIABLES[k]
            mag = stack.magnitud(k)
            lote = mag[m & np.isfinite(mag)]
            vals = mag[sel & np.isfinite(mag)]
            if vals.size == 0 or lote.std() == 0:
                continue
            z = (vals.mean() - lote.mean()) / lote.std()
            rasgos.append((abs(z), z, v, vals.mean(), lote.mean()))
        rasgos.sort(key=lambda r: r[0], reverse=True)
        partes = []
        for _, z, v, val, lote in rasgos[:n_rasgos]:
            palabra = v.descriptor[0] if z > 0 else v.descriptor[1]
            partes.append(f"{v.nombre} {val:.2f} {v.unidad} (lote {lote:.2f}; {palabra})")
        sup = sup_amb.get(amb, np.nan)
        textos[amb] = (f"{nombres[amb]}: {sup:.1f} ha ({100 * sup / total:.1f} % del lote). "
                       "Rasgos distintivos: " + "; ".join(partes) + ".")
    return textos
