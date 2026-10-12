"""Catálogo de variables, paletas, normalización por lote y exportación GeoTIFF."""
from dataclasses import dataclass, field

import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

PALETAS = {
    "ndvi": LinearSegmentedColormap.from_list("ndvi", ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]),
    "ndmi": LinearSegmentedColormap.from_list("ndmi", ["#f57c00", "#ffd54f", "#4dd0e1", "#00acc1"]),
    "cire": LinearSegmentedColormap.from_list("cire", ["#ffffff", "#a1d99b", "#31a354", "#00441b"]),
    "dem": LinearSegmentedColormap.from_list("dem", ["#1a9850", "#a6d96a", "#ffffbf", "#d8b365", "#8c510a"]),
    "tpi": matplotlib.colormaps["turbo"],
    "agua": LinearSegmentedColormap.from_list("agua", ["#ffffff", "#9ecae1", "#3182bd", "#08306b"]),
    "cv": matplotlib.colormaps["YlOrRd"],
    "rend": LinearSegmentedColormap.from_list("rend", ["#d73027", "#fee08b", "#1a9850"]),
}


@dataclass
class Variable:
    clave: str            # nombre de la capa normalizada usada en el clustering
    nombre: str
    capa_magnitud: str    # capa en unidades reales para mapas/estadísticas
    unidad: str
    fuente: str
    res_nativa: str
    paleta: str
    descripcion: str
    descriptor: tuple = field(default=("", ""))  # palabras (valor alto, valor bajo) para nombres


VARIABLES = {v.clave: v for v in [
    Variable("ndvi_rel", "NDVI promedio", "ndvi_media", "adim.", "Sentinel-2 L2A", "10 m", "ndvi",
             "Promedio temporal del NDVI (B8, B4). El clustering usa el NDVI relativo a la media del lote en cada fecha.",
             ("vigoroso", "poco vigoroso")),
    Variable("ndvi_cv", "CV temporal NDVI", "ndvi_cv", "%", "Sentinel-2 L2A", "10 m", "cv",
             "Coeficiente de variación temporal del NDVI relativo al lote.", ("inestable", "estable")),
    Variable("ndmi_rel", "NDMI promedio", "ndmi_media", "adim.", "Sentinel-2 L2A", "10/20 m", "ndmi",
             "Promedio temporal del NDMI (B8, B11), indicador de contenido hídrico del canopeo.",
             ("húmedo", "seco")),
    Variable("ndmi_cv", "CV temporal NDMI", "ndmi_cv", "%", "Sentinel-2 L2A", "10/20 m", "cv",
             "Coeficiente de variación temporal del NDMI relativo al lote.", ("inestable", "estable")),
    Variable("cire_rel", "Clorofila red-edge (CIre) promedio", "cire_media", "adim.", "Sentinel-2 L2A", "20 m", "cire",
             "Promedio temporal del índice de clorofila red-edge CIre = B7/B5 − 1.",
             ("vigoroso", "poco vigoroso")),
    Variable("cire_cv", "CV temporal CIre", "cire_cv", "%", "Sentinel-2 L2A", "20 m", "cv",
             "Coeficiente de variación temporal del CIre relativo al lote.", ("inestable", "estable")),
    Variable("dem", "Altimetría", "dem", "m s.n.m.", "Copernicus DEM GLO-30", "30 m", "dem",
             "Elevación del terreno remuestreada en forma bilineal a 10 m.", ("cota alta", "cota baja")),
    Variable("tpi", "Índice de posición topográfica (TPI)", "tpi", "m", "Copernicus DEM GLO-30", "30 m", "tpi",
             "Elevación menos la media de un círculo de 150 m de radio. Positivo = loma, negativo = bajo.",
             ("loma", "bajo")),
    Variable("anegamiento", "Frecuencia de anegamiento", "anegamiento", "%", "Landsat 5/8/9 C2 L2", "30 m", "agua",
             "Porcentaje de semestres con agua en superficie (1990–actualidad).", ("inundable", "sin anegamiento")),
    Variable("rend_rel", "Rendimiento relativo", "rend_rel", "% de la media", "Monitor de rendimiento", "puntos",
             "rend", "Rendimiento interpolado a 10 m, relativo a la media del lote y promediado entre campañas.",
             ("alto rinde", "bajo rinde")),
    Variable("rend_cv", "CV temporal del rendimiento", "rend_cv", "%", "Monitor de rendimiento", "puntos", "cv",
             "Coeficiente de variación entre campañas del rendimiento relativo.", ("inestable", "estable")),
]}


def normalizar(capa: np.ndarray, mascara: np.ndarray, percentiles=(2, 98)) -> np.ndarray:
    """Min-max robusto dentro del lote (percentiles), recortado a [0, 1]."""
    vals = capa[mascara & np.isfinite(capa)]
    if vals.size == 0:
        return np.full_like(capa, np.nan)
    lo, hi = np.percentile(vals, percentiles)
    if hi - lo < 1e-12:
        out = np.where(np.isfinite(capa), 0.5, np.nan)
    else:
        out = np.clip((capa - lo) / (hi - lo), 0, 1)
    out = out.astype("float32")
    out[~mascara] = np.nan
    return out


class Stack:
    """Conjunto corregistrado de capas sobre la grilla común."""

    def __init__(self, grilla, cfg):
        self.grilla = grilla
        self.cfg = cfg
        self.capas = {}          # magnitudes y auxiliares
        self.norm = {}           # normalizadas 0-1 (sólo variables del catálogo)

    def agregar(self, capas: dict):
        for k, v in capas.items():
            self.capas[k] = np.asarray(v, dtype="float32")

    def normalizar_todo(self):
        self.norm = {}
        for clave in VARIABLES:
            if clave in self.capas and np.isfinite(self.capas[clave][self.grilla.mascara]).any():
                self.norm[clave] = normalizar(self.capas[clave], self.grilla.mascara,
                                              self.cfg.percentiles_norm)
        return list(self.norm)

    @property
    def disponibles(self):
        return list(self.norm)

    def magnitud(self, clave):
        v = VARIABLES[clave]
        if clave == "rend_rel" and "rend_media" in self.capas:
            return self.capas["rend_media"]
        return self.capas[v.capa_magnitud]

    def exportar_geotiff(self, path, capas: dict):
        g = self.grilla
        perfil = dict(driver="GTiff", height=g.alto, width=g.ancho, count=len(capas),
                      dtype="float32", crs=g.crs.to_wkt(), transform=g.transform,
                      nodata=np.nan, compress="deflate")
        with rasterio.open(path, "w", **perfil) as dst:
            for i, (nombre, arr) in enumerate(capas.items(), start=1):
                a = np.where(g.mascara_borde, arr, np.nan).astype("float32")
                dst.write(a, i)
                dst.set_band_description(i, nombre)
