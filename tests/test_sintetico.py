"""Prueba de punta a punta sin Earth Engine: capas sintéticas → informe.

Ejecutar:  python -m tests.test_sintetico   (desde la raíz del repo)
"""
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import Polygon, Point

from ambientacion import Config, Proyecto
from ambientacion import grilla as mod_grilla, sentinel, rendimiento
from ambientacion.variables import Stack


def campo_suave(forma, sigma, rng):
    return ndimage.gaussian_filter(rng.normal(size=forma), sigma)


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="amb_test_")
    cfg = Config(establecimiento="Prueba", lote="Lote 1", salida_dir=os.path.join(tmp, "salidas"))

    # Lote irregular de ~120 ha en UTM 20S
    x0, y0 = 500_000, 6_200_000
    poly = Polygon([(x0, y0), (x0 + 1400, y0 + 80), (x0 + 1300, y0 + 900), (x0 + 600, y0 + 1000),
                    (x0 + 50, y0 + 700)])
    lote = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32720")

    p = Proyecto(cfg)
    p.lote, p.crs = lote, lote.crs
    p.grilla = mod_grilla.crear_grilla(lote, cfg.resolucion_m)
    p.stack = Stack(p.grilla, cfg)
    forma = (p.grilla.alto, p.grilla.ancho)

    relieve = campo_suave(forma, 15, rng)
    relieve = relieve / relieve.std()
    # Índices por fecha (más vigor en lomas, menos en bajos)
    bandas = {}
    fechas = pd.to_datetime(["2023-01-15", "2023-02-10", "2024-01-20"])
    for f in fechas:
        s = f.strftime("%Y%m%d")
        ruido = campo_suave(forma, 3, rng) * 0.3
        bandas[f"ndvi_{s}"] = 0.6 + 0.1 * relieve + 0.05 * ruido
        bandas[f"ndmi_{s}"] = 0.25 - 0.05 * relieve + 0.03 * ruido
        bandas[f"cire_{s}"] = 2.5 + 0.6 * relieve + 0.2 * ruido
    p.stack.agregar(sentinel.compuestos_temporales(bandas, p.grilla.mascara))
    p.stack.agregar({
        "dem": 35 + 1.5 * relieve,
        "tpi": 0.8 * relieve + 0.1 * campo_suave(forma, 2, rng),
        "anegamiento": np.clip(20 - 15 * relieve, 0, 100),
    })
    p.tabla_img = pd.DataFrame({"fecha_solicitada": fechas.date, "fecha_usada": fechas.date, "dif_dias": 0,
                                "cultivo": ["Soja", "Soja", "Maíz"], "campaña": ["22/23", "22/23", "23/24"],
                                "estado": "fecha solicitada", "contaminacion_pct": 0.0})

    # Rendimiento sintético (puntos) → SHP → kriging
    xs = rng.uniform(x0, x0 + 1400, 4000)
    ys = rng.uniform(y0, y0 + 1000, 4000)
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:32720")
    pts = pts[pts.within(poly)]
    cs, rs = ~p.grilla.transform * (pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy())
    rel = relieve[rs.astype(int).clip(0, forma[0] - 1), cs.astype(int).clip(0, forma[1] - 1)]
    pts["Rend_kgha"] = 3500 + 400 * rel + rng.normal(0, 150, len(pts))
    shp = os.path.join(tmp, "rinde.shp")
    pts.to_file(shp)
    tabla_rend = pd.DataFrame({"archivo": ["rinde.shp"], "campaña": ["23/24"], "cultivo": ["Soja"],
                               "columna": ["Rend_kgha"]})
    capas, p.tabla_rend = rendimiento.procesar_rendimiento(tabla_rend, tmp, lote, p.grilla, cfg)
    p.stack.agregar(capas)

    disp = p.normalizar()
    assert "rend_rel" in disp and "tpi" in disp
    res_eda = p.eda()
    print(res_eda["resumen"])
    print("VIF:\n", res_eda["vif"])
    print("Pares correlacionados:", res_eda["pares"])

    k_df, votos, k = p.evaluar_k()
    print(k_df.round(3), votos, "k sugerido:", k)
    nombres = p.clasificar()
    print("Nombres:", nombres)
    exp = p.explicar()
    print(exp["tabla"])
    gdf, val = p.vectorizar()
    print(gdf.drop(columns="geometry"))
    print(val)
    assert val["cumple"].fillna(True).all(), "Falla la validación topológica"
    for t in p.textos.values():
        print(" -", t)
    p.visor().save(os.path.join(tmp, "visor.html"))
    out = p.informe()
    assert os.path.exists(out["html"])
    # Prueba de nombres con más ambientes (grupos parecidos + descriptores)
    for k in (4, 6):
        print(f"k={k}:", p.clasificar(k))
        _, val = p.vectorizar()
        assert val["cumple"].fillna(True).all(), f"Falla la validación topológica con k={k}"
    print("OK →", tmp)
    return out


if __name__ == "__main__":
    main()
