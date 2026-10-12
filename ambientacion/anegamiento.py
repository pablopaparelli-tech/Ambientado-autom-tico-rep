"""Probabilidad (frecuencia) de anegamiento con Landsat 5 / 8 / 9.

Port del script GEE original con estas correcciones:

1. Máscaras de nubes/sombras (QA_PIXEL bits 0-4) y saturación (QA_RADSAT)
   activas, con el escalado de reflectancia de Collection 2.
   Antes fallaban porque un semestre sin observaciones válidas genera un
   `.mean()` sin bandas y el `.select('ndvi')` posterior da error.
2. Semestres ene-jun y jul-dic (el original con meses [1, 6] se superponía en
   junio y dejaba fuera diciembre).
3. Semestres vacíos se descartan automáticamente (sin exclusiones manuales).
4. Dato vacío por píxel: la frecuencia se calcula sobre los semestres válidos de
   cada píxel (agua / válidos). Si un píxel tiene menos de
   `semestres_validos_min` semestres válidos se rellena con la media focal de
   los píxeles vecinos con dato suficiente (radio `radio_relleno_m`).
5. Se incorpora Landsat 9 (2021+), que duplica la revisita junto a Landsat 8.

Regla de agua (sin cambios): ndvi <= 0.25, ndwi_gao >= 0, ndwi_xu >= -0.05,
ndwi_mc >= -0.15 (umbrales configurables).
"""
import pandas as pd

BANDAS_L5 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5"]
BANDAS_L89 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"]
COMUNES = ["blue", "green", "red", "nir", "swir1"]


def _enmascarar(img, bandas):
    import ee
    qa = img.select("QA_PIXEL").bitwiseAnd(0b11111).eq(0)
    sat = img.select("QA_RADSAT").eq(0)
    opt = img.select(bandas).multiply(0.0000275).add(-0.2).rename(COMUNES)
    return ee.Image(opt.updateMask(qa).updateMask(sat)
                    .resample("bilinear")
                    .copyProperties(img, ["system:time_start"]))


def _indices(img):
    import ee
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    gao = img.normalizedDifference(["nir", "swir1"]).rename("ndwi_gao")
    xu = img.normalizedDifference(["green", "swir1"]).rename("ndwi_xu")
    mc = img.normalizedDifference(["green", "nir"]).rename("ndwi_mc")
    return ee.Image(ee.Image.cat([ndvi, gao, xu, mc]).copyProperties(img, ["system:time_start"]))


def coleccion_landsat(region, cfg):
    import ee
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(region)
          .filterDate(cfg.l5_inicio, cfg.l5_fin)
          .filter(ee.Filter.lt("CLOUD_COVER", cfg.nubosidad_max_l5))
          .map(lambda i: _enmascarar(i, BANDAS_L5)))
    sensores = ["LANDSAT/LC08/C02/T1_L2"] + (["LANDSAT/LC09/C02/T1_L2"] if cfg.usar_l9 else [])
    col = l5
    for s in sensores:
        c = (ee.ImageCollection(s).filterBounds(region)
             .filterDate(cfg.l89_inicio, cfg.l89_fin)
             .filter(ee.Filter.lt("CLOUD_COVER", cfg.nubosidad_max_l89))
             .map(lambda i: _enmascarar(i, BANDAS_L89)))
        col = col.merge(c)
    return col.map(_indices)


def _semestres(cfg):
    ini = min(int(cfg.l5_inicio[:4]), int(cfg.l89_inicio[:4]))
    fin = max(int(cfg.l5_fin[:4]), int(cfg.l89_fin[:4]))
    for y in range(ini, fin + 1):
        for m in (1, 7):
            yield f"{y}-{m:02d}-01"


def compuestos_semestrales(region, cfg):
    import ee
    col = coleccion_landsat(region, cfg)
    imgs = []
    for inicio in _semestres(cfg):
        d = ee.Date(inicio)
        sub = col.filterDate(d, d.advance(6, "month"))
        imgs.append(sub.mean().set({"n": sub.size(), "system:time_start": d.millis(),
                                    "semestre": inicio}))
    return ee.ImageCollection.fromImages(imgs).filter(ee.Filter.gt("n", 0))


def imagen_anegamiento(lote_ee, cfg):
    """Bandas: `anegamiento` (% de semestres con agua) y `semestres_validos`."""
    import ee
    region = lote_ee.buffer(cfg.radio_relleno_m * 2)
    comp = compuestos_semestrales(region, cfg)

    def clasificar(img):
        agua = (img.select("ndvi").lte(cfg.umbral_ndvi)
                .And(img.select("ndwi_gao").gte(cfg.umbral_ndwi_gao))
                .And(img.select("ndwi_xu").gte(cfg.umbral_ndwi_xu))
                .And(img.select("ndwi_mc").gte(cfg.umbral_ndwi_mc)))
        valido = img.select("ndvi").gt(-2).unmask(0)
        return agua.unmask(0).rename("agua").addBands(valido.rename("valido"))

    clas = comp.map(clasificar)
    agua = clas.select("agua").sum()
    validos = clas.select("valido").sum()
    freq = agua.divide(validos.max(1)).multiply(100).updateMask(validos.gt(0))
    confiable = validos.gte(cfg.semestres_validos_min)
    vecinos = freq.updateMask(confiable).focalMean(
        radius=cfg.radio_relleno_m, kernelType="circle", units="meters")
    freq = freq.where(confiable.Not(), vecinos.unmask(freq))
    return freq.rename("anegamiento").addBands(validos.rename("semestres_validos").toFloat())


def resumen_semestres(lote_ee, cfg) -> pd.DataFrame:
    """Serie de % del lote con agua por semestre (para el informe)."""
    import ee
    comp = compuestos_semestrales(lote_ee, cfg)

    def stats(img):
        agua = (img.select("ndvi").lte(cfg.umbral_ndvi)
                .And(img.select("ndwi_gao").gte(cfg.umbral_ndwi_gao))
                .And(img.select("ndwi_xu").gte(cfg.umbral_ndwi_xu))
                .And(img.select("ndwi_mc").gte(cfg.umbral_ndwi_mc)))
        r = agua.rename("agua").reduceRegion(ee.Reducer.mean(), lote_ee, scale=30, maxPixels=1e9)
        v = r.get("agua")
        pct = ee.Algorithms.If(ee.Algorithms.IsEqual(v, None), None, ee.Number(v).multiply(100))
        return ee.Feature(None, {"semestre": img.get("semestre"), "n": img.get("n"),
                                 "agua_pct": pct})

    fc = ee.FeatureCollection(comp.map(stats)).getInfo()
    df = pd.DataFrame([f["properties"] for f in fc["features"]])
    if not df.empty:
        df["semestre"] = pd.to_datetime(df["semestre"])
    return df
