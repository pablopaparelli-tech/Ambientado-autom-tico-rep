"""Grilla común de 10 m (UTM) a la que se corregistran todas las capas."""
from dataclasses import dataclass
import math

import numpy as np
from affine import Affine
from rasterio import features


@dataclass
class Grilla:
    crs: object          # pyproj.CRS (UTM del lote)
    transform: Affine
    ancho: int
    alto: int
    mascara: np.ndarray          # píxeles con centro dentro del lote (análisis)
    mascara_borde: np.ndarray    # píxeles que tocan el lote (vectorización)

    @property
    def res(self) -> float:
        return self.transform.a

    @property
    def epsg(self) -> str:
        return f"EPSG:{self.crs.to_epsg()}"

    @property
    def crs_transform(self) -> list:
        t = self.transform
        return [t.a, t.b, t.c, t.d, t.e, t.f]

    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax) para matplotlib."""
        t = self.transform
        return (t.c, t.c + t.a * self.ancho, t.f + t.e * self.alto, t.f)

    def a_ee_grid(self, fila0=0, filas=None) -> dict:
        t = self.transform
        filas = self.alto if filas is None else filas
        return {
            "dimensions": {"width": self.ancho, "height": filas},
            "affineTransform": {
                "scaleX": t.a, "shearX": t.b, "translateX": t.c,
                "shearY": t.d, "scaleY": t.e, "translateY": t.f + t.e * fila0,
            },
            "crsCode": self.epsg,
        }


def crear_grilla(lote_utm, res: float = 10.0, margen_px: int = 2) -> Grilla:
    """Grilla alineada a múltiplos de `res` (coincide con la grilla de Sentinel-2
    en la misma zona UTM)."""
    xmin, ymin, xmax, ymax = lote_utm.total_bounds
    x0 = math.floor(xmin / res) * res - margen_px * res
    y1 = math.ceil(ymax / res) * res + margen_px * res
    x1 = math.ceil(xmax / res) * res + margen_px * res
    y0 = math.floor(ymin / res) * res - margen_px * res
    ancho = int(round((x1 - x0) / res))
    alto = int(round((y1 - y0) / res))
    transform = Affine(res, 0, x0, 0, -res, y1)
    geoms = list(lote_utm.geometry)
    mascara = features.rasterize(geoms, out_shape=(alto, ancho), transform=transform,
                                 all_touched=False, dtype="uint8").astype(bool)
    borde = features.rasterize(geoms, out_shape=(alto, ancho), transform=transform,
                               all_touched=True, dtype="uint8").astype(bool)
    return Grilla(lote_utm.crs, transform, ancho, alto, mascara, borde)


def descargar(imagen, grilla: Grilla, nodata: float = -9999.0, limite_bytes=40e6) -> dict:
    """Descarga una ee.Image multibanda sobre la grilla. Devuelve {banda: array}.

    Las bandas se piden como float32 y los píxeles enmascarados quedan en NaN.
    Si la imagen supera el límite de computePixels se descarga por franjas.
    """
    import ee
    bandas = imagen.bandNames().getInfo()
    img = imagen.toFloat().unmask(nodata)
    filas_por_bloque = max(1, int(limite_bytes // (grilla.ancho * len(bandas) * 4)))
    partes = []
    for fila0 in range(0, grilla.alto, filas_por_bloque):
        filas = min(filas_por_bloque, grilla.alto - fila0)
        arr = ee.data.computePixels({
            "expression": img,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grilla.a_ee_grid(fila0, filas),
        })
        partes.append(arr)
    arr = np.concatenate(partes, axis=0)
    salida = {}
    for b in bandas:
        a = arr[b].astype("float32")
        a[a == nodata] = np.nan
        salida[b] = a
    return salida
