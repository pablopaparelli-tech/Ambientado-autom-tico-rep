"""Mapas estáticos para el informe: magnitudes reales, leyenda, escala y norte."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar

from .variables import PALETAS


def _path_lote(geom):
    verts, codes = [], []
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        for anillo in [p.exterior, *p.interiors]:
            xy = np.asarray(anillo.coords)
            verts.extend(xy)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _decorar(ax, lote_utm, titulo):
    lote_utm.boundary.plot(ax=ax, color="#222222", lw=1.0)
    ax.add_artist(ScaleBar(1, "m", location="lower left", box_alpha=0.7, font_properties={"size": 8}))
    ax.annotate("N", xy=(0.94, 0.95), xytext=(0.94, 0.80), xycoords="axes fraction",
                ha="center", va="center", fontsize=12, fontweight="bold",
                arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=10))
    ax.set_title(titulo, fontsize=11)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("Este (m, UTM)", fontsize=7)
    ax.set_ylabel("Norte (m, UTM)", fontsize=7)
    xmin, ymin, xmax, ymax = lote_utm.total_bounds
    dx, dy = (xmax - xmin) * 0.06, (ymax - ymin) * 0.06
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_aspect("equal")


def mapa_capa(arr, grilla, lote_utm, titulo, unidad, paleta, ax=None, percentiles=(2, 98)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    vals = arr[grilla.mascara & np.isfinite(arr)]
    vmin, vmax = np.percentile(vals, percentiles) if vals.size else (0, 1)
    cmap = PALETAS[paleta] if isinstance(paleta, str) else paleta
    im = ax.imshow(np.where(grilla.mascara_borde, arr, np.nan), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=grilla.extent, interpolation="nearest")
    patch = PathPatch(_path_lote(lote_utm.geometry.iloc[0]), transform=ax.transData, fc="none", ec="none")
    ax.add_patch(patch)
    im.set_clip_path(patch)
    cb = plt.colorbar(im, ax=ax, shrink=0.75, extend="both")
    cb.set_label(unidad, fontsize=9)
    _decorar(ax, lote_utm, titulo)
    return fig or ax.figure


def mapa_ambientes(gdf, lote_utm, titulo="Ambientes", ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    gdf.plot(ax=ax, color=gdf["color"], edgecolor="#333333", linewidth=0.6)
    leyenda = [Patch(facecolor=r.color, edgecolor="#333333",
                     label=f"{r.ambiente_id} · {r.ambiente} — {r.sup_ha:.1f} ha ({r.sup_pct:.0f} %)")
               for r in gdf.itertuples()]
    ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              title="Ambiente", title_fontsize=10, frameon=False)
    _decorar(ax, lote_utm, titulo)
    (fig or ax.figure).tight_layout()
    return fig or ax.figure
