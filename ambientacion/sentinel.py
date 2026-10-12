"""Sentinel-2: búsqueda de fechas sin nubes ni sombras sobre el lote e índices.

- Colección S2_SR_HARMONIZED + Cloud Score+ (cs_cdf) + clasificación de escena (SCL).
- Una fecha es válida sólo si hay 0 píxeles contaminados (o sin dato) en el lote
  más un buffer. Si la fecha solicitada no cumple, se toma la fecha válida más
  cercana dentro de ± `ventana_dias`.
- Índices: NDVI (B8, B4), NDMI (B8, B11) y CIre = B7/B5 − 1 (clorofila red-edge).
  Las bandas de 20 m se remuestrean en forma bilineal a la grilla de 10 m.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
INDICES = ("ndvi", "ndmi", "cire")


def _coleccion(region, inicio, fin):
    import ee
    cs = ee.ImageCollection(CS_PLUS)
    return (ee.ImageCollection(S2)
            .filterBounds(region)
            .filterDate(inicio, fin)
            .linkCollection(cs, ["cs_cdf"]))


def _contaminacion(img, cfg):
    """1 = píxel inutilizable (nube, sombra, cirro, sin dato), 0 = limpio."""
    import ee
    inval = list(cfg.scl_invalidas)
    scl_mala = img.select("SCL").remap(inval, [1] * len(inval), 0).unmask(1)
    cs_mala = img.select("cs_cdf").lt(cfg.cs_umbral).unmask(0)
    sin_dato = img.select("B4").unmask(-1).eq(-1)
    return scl_mala.Or(cs_mala).Or(sin_dato).rename("malo")


def _mosaico_dia(col, dia: str):
    import ee
    d = ee.Date(dia)
    sub = col.filterDate(d, d.advance(1, "day"))
    return sub.map(lambda i: i.resample("bilinear")).mosaic().set({"dia": dia, "n": sub.size()})


def evaluar_ventana(fecha: pd.Timestamp, region_control, crs: str, cfg) -> pd.DataFrame:
    """Evalúa cada día con imagen en la ventana y cuenta píxeles contaminados."""
    import ee
    ini = (fecha - timedelta(days=cfg.ventana_dias)).strftime("%Y-%m-%d")
    fin = (fecha + timedelta(days=cfg.ventana_dias + 1)).strftime("%Y-%m-%d")
    col = _coleccion(region_control, ini, fin)
    tiempos = col.aggregate_array("system:time_start").getInfo()
    if not tiempos:
        return pd.DataFrame(columns=["dia", "malos", "total"])
    dias = sorted({pd.to_datetime(t, unit="ms").strftime("%Y-%m-%d") for t in tiempos})

    def stats(dia):
        m = _mosaico_dia(col, dia)
        malo = _contaminacion(m, cfg)
        r = malo.reduceRegion(ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
                              region_control, scale=cfg.resolucion_m, crs=crs, maxPixels=1e9)
        return ee.Feature(None, {"dia": dia, "malos": r.get("malo_sum"), "total": r.get("malo_count")})

    fc = ee.FeatureCollection([stats(d) for d in dias])
    filas = [f["properties"] for f in fc.getInfo()["features"]]
    return pd.DataFrame(filas)


def seleccionar_fechas(tabla_img: pd.DataFrame, lote_ee, crs: str, cfg) -> pd.DataFrame:
    """Para cada fecha solicitada busca la fecha limpia más cercana."""
    region = lote_ee.buffer(cfg.buffer_nubes_m)
    filas = []
    for _, r in tabla_img.iterrows():
        f = pd.Timestamp(r["fecha"])
        ev = evaluar_ventana(f, region, crs, cfg)
        usada, estado, malos_pct = None, "sin imágenes en la ventana", np.nan
        if not ev.empty:
            ev["dia"] = pd.to_datetime(ev["dia"])
            ev["delta"] = (ev["dia"] - f).dt.days
            ev["malos_pct"] = 100 * ev["malos"] / ev["total"].clip(lower=1)
            limpios = ev[ev["malos"] == 0].copy()
            if limpios.empty:
                estado = "sin fecha limpia en la ventana"
                malos_pct = ev.loc[ev["delta"].abs().idxmin(), "malos_pct"]
            else:
                limpios["abs"] = limpios["delta"].abs()
                mejor = limpios.sort_values(["abs", "delta"]).iloc[0]
                usada = mejor["dia"]
                malos_pct = 0.0
                estado = "fecha solicitada" if mejor["delta"] == 0 else "fecha cercana"
        filas.append({
            "fecha_solicitada": f.date(),
            "fecha_usada": usada.date() if usada is not None else None,
            "dif_dias": int((usada - f).days) if usada is not None else None,
            "cultivo": r.get("cultivo"),
            "campaña": r.get("campaña"),
            "estado": estado,
            "contaminacion_pct": round(float(malos_pct), 2) if not pd.isna(malos_pct) else None,
        })
    out = pd.DataFrame(filas)
    # La misma fecha usada no se repite
    dup = out["fecha_usada"].notna() & out.duplicated("fecha_usada")
    out.loc[dup, "estado"] = "duplicada (excluida)"
    return out


def imagen_indices(fechas_usadas, lote_ee, cfg):
    """ee.Image con bandas ndvi_YYYYMMDD, ndmi_YYYYMMDD, cire_YYYYMMDD."""
    import ee
    region = lote_ee.buffer(cfg.buffer_nubes_m)
    imgs = []
    for f in fechas_usadas:
        dia = pd.Timestamp(f).strftime("%Y-%m-%d")
        sufijo = pd.Timestamp(f).strftime("%Y%m%d")
        col = _coleccion(region, dia, (pd.Timestamp(f) + timedelta(days=1)).strftime("%Y-%m-%d"))
        m = _mosaico_dia(col, dia)
        ndvi = m.normalizedDifference(["B8", "B4"]).rename(f"ndvi_{sufijo}")
        ndmi = m.normalizedDifference(["B8", "B11"]).rename(f"ndmi_{sufijo}")
        cire = m.select("B7").toFloat().divide(m.select("B5").toFloat()).subtract(1).rename(f"cire_{sufijo}")
        imgs.append(ndvi.addBands(ndmi).addBands(cire))
    return ee.Image.cat(imgs)


def compuestos_temporales(bandas: dict, mascara: np.ndarray) -> dict:
    """A partir de los índices por fecha calcula, para cada índice:

    - `<ind>_media`: promedio temporal en magnitud (para el informe).
    - `<ind>_cv`: coeficiente de variación temporal (%) del índice relativo al lote
      (valor / media del lote en esa fecha), libre de escala.
    - `<ind>_rel`: promedio temporal del índice relativo (para el clustering).
    """
    salida = {}
    for ind in INDICES:
        claves = sorted(k for k in bandas if k.startswith(ind + "_"))
        if not claves:
            continue
        pila = np.stack([bandas[k] for k in claves])
        rel = []
        for capa in pila:
            media_lote = np.nanmean(capa[mascara])
            valido = np.isfinite(media_lote) and media_lote != 0
            rel.append(capa / media_lote if valido else capa * np.nan)
        rel = np.stack(rel)
        salida[f"{ind}_media"] = np.nanmean(pila, axis=0)
        salida[f"{ind}_rel"] = np.nanmean(rel, axis=0)
        if len(claves) > 1:
            m = np.nanmean(rel, axis=0)
            salida[f"{ind}_cv"] = 100 * np.nanstd(rel, axis=0, ddof=1) / np.abs(m)
    return salida
