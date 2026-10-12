"""Clasificación raster → polígonos generalizados y topológicamente consistentes.

Estrategia (la topología se garantiza por construcción y luego se valida):
1. Relleno de píxeles de borde (tocan el lote) con el ambiente más cercano.
2. Filtro de mayoría 3×3.
3. Unidad mínima de mapeo (UMM): manchas < UMM se fusionan con el ambiente
   vecino con el que comparten más borde.
4. Suavizado: probabilidad por ambiente → filtro gaussiano → sobremuestreo
   bilineal → clase de máxima probabilidad. Al ser una partición raster no
   puede generar huecos ni superposiciones.
5. Vectorización y simplificación de cobertura (shapely.coverage_simplify):
   cada borde compartido se simplifica una sola vez para ambos vecinos.
6. Recorte al perímetro del lote, fusión de astillas < UMM y validación.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape

PRECISION = 0.001  # m
CONECTIVIDAD = ndimage.generate_binary_structure(2, 1)  # 4-conectividad (igual que la vectorización)


def rellenar_borde(raster, mascara_borde):
    vacio = mascara_borde & (raster == 0)
    if not vacio.any():
        return raster
    _, (fi, ci) = ndimage.distance_transform_edt(raster == 0, return_indices=True)
    out = raster.copy()
    out[vacio] = raster[fi[vacio], ci[vacio]]
    return out


def filtro_mayoria(raster, mascara, tam=3):
    clases = [c for c in np.unique(raster) if c > 0]
    conteos = np.stack([ndimage.uniform_filter((raster == c).astype("float32"), tam, mode="nearest")
                        for c in clases])
    out = np.array(clases, dtype=raster.dtype)[np.argmax(conteos, axis=0)]
    out[~mascara] = 0
    return out


def aplicar_umm(raster, umm_px, max_iter=20):
    """Fusiona manchas menores a `umm_px` con el vecino de mayor borde compartido."""
    out = raster.copy()
    alto, ancho = out.shape
    for _ in range(max_iter):
        pequeñas = []
        for c in [c for c in np.unique(out) if c > 0]:
            lab, n = ndimage.label(out == c, structure=CONECTIVIDAD)
            if n == 0:
                continue
            tam = np.bincount(lab.ravel())
            objetos = ndimage.find_objects(lab)
            for i in np.nonzero(tam < umm_px)[0]:
                if i > 0:
                    s0, s1 = objetos[i - 1]
                    sl = (slice(max(s0.start - 1, 0), min(s0.stop + 1, alto)),
                          slice(max(s1.start - 1, 0), min(s1.stop + 1, ancho)))
                    pequeñas.append((tam[i], c, sl, lab[sl] == i))
        pequeñas.sort(key=lambda t: t[0])
        cambios = 0
        for _, c, sl, sub in pequeñas:
            region = out[sl]
            sub = sub & (region == c)  # puede haber cambiado en esta iteración
            if not sub.any():
                continue
            anillo = ndimage.binary_dilation(sub, CONECTIVIDAD) & ~sub
            vecinos = region[anillo]
            vecinos = vecinos[(vecinos > 0) & (vecinos != c)]
            if vecinos.size == 0:
                continue  # isla aislada (p.ej. parte separada del lote)
            region[sub] = np.bincount(vecinos).argmax()
            cambios += 1
        if cambios == 0:
            break
    return out


def suavizar(raster, mascara_borde, factor, sigma_px):
    clases = [c for c in np.unique(raster) if c > 0]
    probs = []
    for c in clases:
        p = ndimage.gaussian_filter((raster == c).astype("float32"), sigma=sigma_px, mode="nearest")
        probs.append(ndimage.zoom(p, factor, order=1))
    up = np.array(clases, dtype=raster.dtype)[np.argmax(np.stack(probs), axis=0)]
    borde_up = ndimage.zoom(mascara_borde.astype("uint8"), factor, order=0).astype(bool)
    up[~borde_up] = 0
    return up


def poligonizar(raster, transform, crs):
    geoms, vals = [], []
    for geom, val in features.shapes(raster.astype("int32"), mask=raster > 0,
                                     transform=transform, connectivity=4):
        geoms.append(shape(geom))
        vals.append(int(val))
    gdf = gpd.GeoDataFrame({"ambiente_id": vals}, geometry=geoms, crs=crs)
    return gdf.dissolve("ambiente_id").reset_index()


def _fusionar_astillas(gdf, area_min):
    """Fusiona partes < area_min con el polígono vecino de mayor borde compartido."""
    partes = gdf.explode(index_parts=False).reset_index(drop=True)
    for _ in range(10):
        partes["area"] = partes.area
        chicas = partes[partes["area"] < area_min].sort_values("area")
        if chicas.empty:
            break
        fusionadas = False
        for idx in chicas.index:
            if idx not in partes.index:
                continue
            g = partes.at[idx, "geometry"]
            cand = partes[(partes.index != idx) & partes.intersects(g)]
            if cand.empty:
                continue
            borde = cand.geometry.intersection(g.boundary).length
            destino = borde.idxmax()
            if borde[destino] <= 0:
                continue
            partes.at[destino, "geometry"] = shapely.union_all([partes.at[destino, "geometry"], g],
                                                               grid_size=PRECISION)
            partes = partes.drop(idx)
            fusionadas = True
        if not fusionadas:
            break
    geoms = [shapely.union_all(list(g.geometry), grid_size=PRECISION) for _, g in partes.groupby("ambiente_id")]
    ids = sorted(partes["ambiente_id"].unique())
    return gpd.GeoDataFrame({"ambiente_id": ids}, geometry=geoms, crs=partes.crs)


def recortar_cobertura(gdf, lote):
    """Recorta la cobertura al lote en una sola operación topológica: se nodan
    juntos todos los bordes de ambientes y el perímetro (grilla de 1 mm), se
    poligonizan las caras y cada cara toma el ambiente que la contiene. Todas
    las caras salen del mismo conjunto de bordes, así que la partición es exacta."""
    lote = shapely.set_precision(lote, PRECISION)
    lineas = [g.boundary for g in gdf.geometry] + [lote.boundary]
    nodado = shapely.union_all(lineas, grid_size=PRECISION)
    caras = list(shapely.polygonize(list(getattr(nodado, "geoms", [nodado]))).geoms)
    arbol = shapely.STRtree(gdf.geometry.to_numpy())
    ids = gdf["ambiente_id"].to_numpy()
    asignadas = {}
    for cara in caras:
        pt = cara.point_on_surface()
        if not lote.covers(pt):
            continue
        cand = arbol.query(pt, predicate="intersects")
        if len(cand) == 0:
            # cara interior sin ambiente (no debería ocurrir): toma el vecino más cercano
            cand = [arbol.nearest(pt)]
        asignadas.setdefault(int(ids[cand[0]]), []).append(cara)
    claves = sorted(asignadas)
    geoms = [shapely.union_all(asignadas[k], grid_size=PRECISION) for k in claves]
    return gpd.GeoDataFrame({"ambiente_id": claves}, geometry=geoms, crs=gdf.crs)


def validar(gdf, lote_utm, umm_ha):
    lote = shapely.set_precision(lote_utm.geometry.iloc[0], PRECISION)
    geoms = gdf.geometry.to_numpy()
    union = shapely.union_all(geoms)
    sup_sum = sum(g.area for g in geoms)
    solape = max(sup_sum - union.area, 0)
    hueco = lote.difference(union, grid_size=PRECISION).area
    fuera = union.difference(lote, grid_size=PRECISION).area
    partes = gdf.explode(index_parts=False)
    chicas = int((partes.area < umm_ha * 10_000 * 0.999).sum())
    try:
        cobertura_ok = bool(np.all(shapely.coverage_is_valid(geoms)))
    except AttributeError:
        cobertura_ok = None
    tol = 0.01  # m²
    checks = [
        ("Geometrías válidas (OGC)", bool(gdf.is_valid.all()), f"{(~gdf.is_valid).sum()} inválidas"),
        ("Cobertura válida (bordes compartidos coincidentes)", cobertura_ok, "shapely.coverage_is_valid"),
        ("Sin superposiciones", solape <= tol, f"{solape:.2f} m²"),
        ("Sin huecos dentro del lote", hueco <= tol, f"{hueco:.2f} m²"),
        ("Sin superficie fuera del lote", fuera <= tol, f"{fuera:.2f} m²"),
        ("Superficie total = lote", abs(union.area - lote.area) <= max(tol, 1e-6 * lote.area),
         f"{union.area / 1e4:.4f} ha vs {lote.area / 1e4:.4f} ha (dif. {abs(union.area - lote.area):.3f} m²)"),
        (f"Sin polígonos < UMM ({umm_ha} ha)", chicas == 0, f"{chicas} polígonos"),
        ("Sin geometrías vacías", bool((~gdf.is_empty).all()), ""),
    ]
    return pd.DataFrame(checks, columns=["regla", "cumple", "detalle"])


def vectorizar(raster_rank, grilla, lote_utm, cfg, nombres, colores, potencial=None):
    r = rellenar_borde(raster_rank, grilla.mascara_borde)
    r = filtro_mayoria(r, grilla.mascara_borde)
    r = aplicar_umm(r, cfg.umm_px)
    raster_gen = r.copy()

    f = max(1, int(cfg.factor_suavizado))
    up = suavizar(r, grilla.mascara_borde, f, cfg.sigma_suavizado_px)
    up = aplicar_umm(up, cfg.umm_px * f * f)
    t = grilla.transform
    t_up = Affine(t.a / f, 0, t.c, 0, t.e / f, t.f)
    gdf = poligonizar(up, t_up, grilla.crs)

    # El borde exterior no se simplifica: queda por fuera del lote y se recorta con su perímetro
    simpl = shapely.coverage_simplify(gdf.geometry.to_numpy(), cfg.tolerancia_simplificacion_m,
                                      simplify_boundary=False)
    gdf = gdf.set_geometry(list(simpl))
    gdf = recortar_cobertura(gdf, lote_utm.geometry.iloc[0])
    gdf = _fusionar_astillas(gdf, cfg.umm_ha * 10_000)

    gdf["ambiente"] = gdf["ambiente_id"].map(nombres)
    gdf["color"] = gdf["ambiente_id"].map(colores)
    gdf["sup_ha"] = gdf.area / 10_000
    gdf["sup_pct"] = 100 * gdf["sup_ha"] / gdf["sup_ha"].sum()
    if potencial is not None:
        gdf["potencial"] = gdf["ambiente_id"].map(potencial)
    gdf = gdf.sort_values("ambiente_id").reset_index(drop=True)
    validacion = validar(gdf, lote_utm, cfg.umm_ha)
    return gdf, validacion, raster_gen


def rasterizar(gdf, grilla):
    """Polígonos finales → grilla de 10 m (para estadísticas coherentes con el mapa)."""
    out = features.rasterize(((g, int(i)) for g, i in zip(gdf.geometry, gdf["ambiente_id"])),
                             out_shape=(grilla.alto, grilla.ancho), transform=grilla.transform,
                             fill=0, dtype="int16")
    out[~grilla.mascara] = 0
    return out
