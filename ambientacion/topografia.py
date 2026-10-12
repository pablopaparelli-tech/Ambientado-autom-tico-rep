"""Topografía: DEM Copernicus GLO-30 y TPI.

El DEM se remuestrea en forma bilineal a la grilla de 10 m y sobre ese DEM se
calcula el TPI = elevación − media focal en un círculo de radio `tpi_radio_m`.
GEE usa datos fuera del lote para la ventana focal, así que no hay efecto borde.
"""


def imagen_topografia(lote_ee, grilla, cfg):
    import ee
    region = lote_ee.buffer(cfg.tpi_radio_m * 3)
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(region).select("DEM")
    proy = col.first().projection()
    dem = (col.map(lambda i: i.resample("bilinear")).mosaic()
           .setDefaultProjection(proy)
           .reproject(crs=grilla.epsg, crsTransform=grilla.crs_transform))
    media_focal = dem.focalMean(radius=cfg.tpi_radio_m, kernelType="circle", units="meters")
    tpi = dem.subtract(media_focal)
    return dem.rename("dem").addBands(tpi.rename("tpi"))
