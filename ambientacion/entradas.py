"""Lectura de entradas: perímetro del lote, Excel de fechas/cultivos y rendimiento."""
import os
import tempfile
import zipfile

import geopandas as gpd
import pandas as pd

COLS_IMAGENES = ["fecha", "cultivo", "campaña"]
COLS_RENDIMIENTO = ["archivo", "campaña", "cultivo", "columna"]


def _resolver_vectorial(path: str) -> str:
    """Admite .shp, .geojson, .gpkg o un .zip con el shapefile adentro."""
    if path.lower().endswith(".zip"):
        destino = tempfile.mkdtemp(prefix="vec_")
        with zipfile.ZipFile(path) as z:
            z.extractall(destino)
        for raiz, _, archivos in os.walk(destino):
            for a in archivos:
                if a.lower().endswith((".shp", ".geojson", ".json", ".gpkg")):
                    return os.path.join(raiz, a)
        raise FileNotFoundError(f"No se encontró un vectorial dentro de {path}")
    return path


def leer_vectorial(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_resolver_vectorial(path))
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def leer_lote(path: str):
    """Devuelve (lote_utm, crs_utm). El lote se disuelve en un único polígono."""
    gdf = leer_vectorial(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("El archivo del lote no contiene polígonos.")
    crs_utm = gdf.to_crs(4326).estimate_utm_crs()
    lote = gdf.to_crs(crs_utm).dissolve()
    lote["geometry"] = lote.geometry.make_valid()
    lote = lote[["geometry"]].reset_index(drop=True)
    return lote, crs_utm


def lote_a_ee(lote_utm: gpd.GeoDataFrame):
    import ee
    geo = lote_utm.to_crs(4326).geometry.iloc[0].__geo_interface__
    return ee.Geometry(geo, proj="EPSG:4326", geodesic=False)


def leer_excel(path: str):
    """Lee hojas 'imagenes' y 'rendimiento' (esta última opcional)."""
    hojas = pd.read_excel(path, sheet_name=None)
    hojas = {k.strip().lower(): v for k, v in hojas.items()}
    if "imagenes" not in hojas:
        raise ValueError("El Excel debe tener una hoja 'imagenes' con columnas: "
                         + ", ".join(COLS_IMAGENES))
    img = hojas["imagenes"].copy()
    img.columns = [str(c).strip().lower() for c in img.columns]
    faltan = [c for c in COLS_IMAGENES if c not in img.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en 'imagenes': {faltan}")
    img = img.dropna(subset=["fecha"])
    img["fecha"] = pd.to_datetime(img["fecha"], dayfirst=True).dt.normalize()
    img = img.sort_values("fecha").reset_index(drop=True)

    rend = hojas.get("rendimiento")
    if rend is not None:
        rend = rend.copy()
        rend.columns = [str(c).strip().lower() for c in rend.columns]
        rend = rend.dropna(subset=["archivo"])
        faltan = [c for c in COLS_RENDIMIENTO if c not in rend.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en 'rendimiento': {faltan}")
        if rend.empty:
            rend = None
    return img, rend


def crear_plantilla_excel(path: str):
    img = pd.DataFrame({
        "fecha": ["15/01/2024", "10/02/2024", "20/12/2022"],
        "cultivo": ["Soja", "Soja", "Maíz"],
        "campaña": ["2023/24", "2023/24", "2022/23"],
    })
    rend = pd.DataFrame({
        "archivo": ["rinde_soja_2324.shp"],
        "campaña": ["2023/24"],
        "cultivo": ["Soja"],
        "columna": ["Rend_kgha"],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        img.to_excel(w, sheet_name="imagenes", index=False)
        rend.to_excel(w, sheet_name="rendimiento", index=False)
