"""Visor interactivo (folium) para el analista con las paletas acordadas."""
import base64
import io

import folium
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling

from .variables import PALETAS, VARIABLES


def _a_png_4326(arr, grilla, cmap, vmin, vmax):
    src = np.where(grilla.mascara_borde, arr, np.nan).astype("float32")
    dst_t, w, h = calculate_default_transform(grilla.crs, "EPSG:4326", grilla.ancho, grilla.alto,
                                              *_bounds(grilla))
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(src, dst, src_transform=grilla.transform, src_crs=grilla.crs, dst_transform=dst_t,
              dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    norm = np.clip((dst - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cmap(norm)
    rgba[..., 3] = np.where(np.isfinite(dst), 0.9, 0)
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    oeste, norte = dst_t.c, dst_t.f
    este, sur = oeste + dst_t.a * w, norte + dst_t.e * h
    return url, [[sur, oeste], [norte, este]]


def _bounds(g):
    xmin, xmax, ymin, ymax = g.extent
    return xmin, ymin, xmax, ymax


def visor(stack, lote_utm, ambientes=None, normalizadas=False):
    lote4326 = lote_utm.to_crs(4326)
    c = lote4326.geometry.iloc[0].centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satélite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    primero = True
    for k in stack.disponibles:
        v = VARIABLES[k]
        arr = stack.norm[k] if normalizadas else stack.magnitud(k)
        vals = arr[stack.grilla.mascara & np.isfinite(arr)]
        vmin, vmax = (0, 1) if normalizadas else np.percentile(vals, (2, 98))
        url, bounds = _a_png_4326(arr, stack.grilla, PALETAS[v.paleta], vmin, vmax)
        nombre = f"{v.nombre} [{'0-1' if normalizadas else v.unidad}] ({vmin:.2f} – {vmax:.2f})"
        folium.raster_layers.ImageOverlay(url, bounds, name=nombre, show=primero).add_to(m)
        primero = False
    if ambientes is not None:
        a = ambientes.to_crs(4326)
        folium.GeoJson(
            a[["ambiente_id", "ambiente", "sup_ha", "color", "geometry"]].to_json(), name="Ambientes",
            style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#222",
                                      "weight": 1, "fillOpacity": 0.75},
            tooltip=folium.GeoJsonTooltip(["ambiente_id", "ambiente", "sup_ha"],
                                          aliases=["Id", "Ambiente", "Superficie (ha)"]),
        ).add_to(m)
    folium.GeoJson(lote4326.to_json(), name="Lote",
                   style_function=lambda f: {"color": "#000", "weight": 2, "fill": False}).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lote4326.total_bounds[1], lote4326.total_bounds[0]],
                  [lote4326.total_bounds[3], lote4326.total_bounds[2]]])
    return m
