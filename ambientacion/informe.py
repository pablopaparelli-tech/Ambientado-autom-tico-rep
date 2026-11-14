"""Informe HTML autocontenido + PDF (WeasyPrint) y exportación de productos."""
import base64
import io
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import mapas
from .variables import VARIABLES

DIR = os.path.dirname(__file__)


def fig_a_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _html_tabla(df, index=False, decimales=2):
    return df.to_html(index=index, float_format=lambda x: f"{x:,.{decimales}f}", border=0,
                      classes="tabla", na_rep="–")


def tabla_variables(p):
    filas = []
    for k in p.stack.disponibles:
        v = VARIABLES[k]
        filas.append({"Variable": p.nombre(k), "Fuente": v.fuente, "Resolución nativa": v.res_nativa,
                      "Unidad (mapa)": p.unidad(k), "Usada en clustering": "Sí" if k in p.claves else
                      ("No (" + p.stack.excluidas[k] + ")" if k in p.stack.excluidas else "No"),
                      "Descripción": v.descripcion})
    return pd.DataFrame(filas)


def tabla_estadisticas_informe(p):
    """Variables en filas, ambientes en columnas: media ± desvío (CV %)."""
    df = p.est_largo.copy()
    df["valor"] = df.apply(lambda r: f"{r['media']:.2f} ± {r['desvío']:.2f} (" +
                           (f"{r['CV %']:.0f} %" if pd.notna(r["CV %"]) else "–") + ")", axis=1)
    df["Variable"] = df["variable"] + " [" + df["unidad"] + "]"
    orden_vars = list(dict.fromkeys(df["Variable"]))
    df["Ambiente"] = df["ambiente_id"].astype(str) + " · " + df["ambiente"]
    t = df.pivot_table(index="Variable", columns="Ambiente", values="valor", aggfunc="first")
    sup = (p.gdf.assign(Ambiente=p.gdf["ambiente_id"].astype(str) + " · " + p.gdf["ambiente"])
           .set_index("Ambiente")[["sup_ha", "sup_pct"]])
    t.loc["Superficie [ha]"] = sup["sup_ha"].map(lambda x: f"{x:.1f}")
    t.loc["Superficie [% del lote]"] = sup["sup_pct"].map(lambda x: f"{x:.1f}")
    orden = ["Superficie [ha]", "Superficie [% del lote]"] + orden_vars
    t = t.loc[orden, sorted(t.columns, key=lambda c: int(c.split(" · ")[0]))]
    t.index.name = None
    t.columns.name = None
    return t


def _md_html(nombre):
    path = os.path.join(DIR, "..", "docs", nombre)
    if not os.path.exists(path):
        return ""
    import markdown
    with open(path, encoding="utf-8") as f:
        return markdown.markdown(f.read(), extensions=["tables"])


def _logo_b64(cfg):
    if cfg.logo_path and os.path.exists(cfg.logo_path):
        with open(cfg.logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def _contacto(cfg):
    return " · ".join(x for x in (cfg.contacto_web, cfg.contacto_mail, cfg.contacto_tel) if x)


def _fecha(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y") if pd.notna(d) else "–"


def tabla_imagenes_simple(p):
    t = p.tabla_img[p.tabla_img["fecha_usada"].notna() & ~p.tabla_img["estado"].str.startswith("duplicada")]
    filas = []
    for _, r in t.iterrows():
        nota = "" if r["dif_dias"] == 0 else f"reemplaza al {_fecha(r['fecha_solicitada'])} por nubes"
        filas.append({"Fecha de la imagen": _fecha(r["fecha_usada"]), "Cultivo": r["cultivo"],
                      "Campaña": r["campaña"], "Observación": nota})
    excl = len(p.tabla_img) - len(t)
    df = pd.DataFrame(filas)
    html = _html_tabla(df)
    if excl:
        html += (f'<p class="nota">{excl} fecha(s) solicitada(s) no se usaron por no encontrarse una imagen '
                 "sin nubes ni sombras sobre el lote (detalle en el anexo).</p>")
    return html


def tabla_rendimiento_simple(p):
    if p.tabla_rend is None:
        return None
    t = p.tabla_rend.rename(columns={"campaña": "Campaña", "cultivo": "Cultivo"})
    t["Rinde medio"] = p.tabla_rend.apply(lambda r: f"{r['media']:,.0f} ({r['columna']})", axis=1)
    return _html_tabla(t[["Campaña", "Cultivo", "Rinde medio"]])


def _figs_capas(p, principales):
    from .lectura import SIMPLE, nombre_simple, unidad_simple
    g = p.stack.grilla
    out = []
    for k in p.stack.disponibles:
        s = SIMPLE.get(k)
        if s is None or s.principal != principales:
            continue
        titulo = nombre_simple(p, k)
        fig = mapas.mapa_capa(p.stack.magnitud(k), g, p.lote, titulo, unidad_simple(p, k), VARIABLES[k].paleta)
        epigrafe = s.epigrafe
        if k in p.stack.excluidas:
            epigrafe += " Sin diferencias dentro del lote: no se usó para delimitar ambientes."
        out.append({"titulo": titulo, "epigrafe": epigrafe, "img": fig_a_b64(fig)})
    return out


def generar(p, carpeta):
    from . import lectura
    os.makedirs(carpeta, exist_ok=True)
    cfg = p.cfg

    serie = None
    if p.serie_anegamiento is not None and not p.serie_anegamiento.empty:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        s = p.serie_anegamiento.dropna(subset=["agua_pct"])
        ax.bar(s["semestre"], s["agua_pct"], width=150, color="#3182bd")
        ax.set_ylabel("% del lote con agua")
        ax.set_title("Superficie anegada por semestre (Landsat)")
        fig.tight_layout()
        serie = fig_a_b64(fig)

    tabla_img = p.tabla_img.rename(columns={
        "fecha_solicitada": "Fecha solicitada", "fecha_usada": "Fecha utilizada", "dif_dias": "Δ días",
        "cultivo": "Cultivo", "campaña": "Campaña", "estado": "Estado"}).drop(columns=["contaminacion_pct"], errors="ignore")

    fichas = lectura.fichas(p)
    contacto = _contacto(cfg)
    pie = " · ".join(x for x in (cfg.empresa, cfg.analista and f"Responsable: {cfg.analista}", contacto) if x)
    anexo = bool(cfg.anexo_tecnico)

    ctx = {
        "cfg": cfg,
        "logo": _logo_b64(cfg),
        "contacto": contacto,
        "pie": pie,
        "fecha": date.today().strftime("%d/%m/%Y"),
        "sup_lote": p.lote.area.iloc[0] / 10_000,
        "k": len(p.nombres),
        "mapa_ambientes": fig_a_b64(mapas.mapa_ambientes(p.gdf, p.lote, f"Ambientes — {cfg.lote}")),
        "fichas": fichas,
        "conclusiones": lectura.conclusiones(p, fichas),
        "capas": _figs_capas(p, principales=True),
        "tabla_imagenes_simple": tabla_imagenes_simple(p),
        "tabla_rendimiento_simple": tabla_rendimiento_simple(p),
        "como_se_hizo": lectura.como_se_hizo(p),
        "anexo": anexo,
    }
    if anexo:
        ctx.update({
            "tabla_variables": _html_tabla(tabla_variables(p)),
            "fig_k": fig_a_b64(p.fig_k) if p.fig_k is not None else None,
            "tabla_k": _html_tabla(p.k_df.reset_index(), decimales=3),
            "votos": p.votos,
            "fig_importancia": fig_a_b64(p.explic["fig"]),
            "tabla_importancia": _html_tabla(p.explic["tabla"].drop(columns=["p_valor"], errors="ignore").rename(
                columns={"variable": "Variable", "importancia_permutacion": "Importancia (permutación)",
                         "H_kruskal": "H Kruskal-Wallis", "epsilon2": "ε²", "shap_medio": "|SHAP| medio"}),
                decimales=3),
            "exactitud_rf": p.explic["exactitud_rf"],
            "tabla_estadisticas": _html_tabla(tabla_estadisticas_informe(p), index=True),
            "capas_anexo": _figs_capas(p, principales=False),
            "serie_anegamiento": serie,
            "tabla_imagenes": _html_tabla(tabla_img),
            "tabla_rendimiento": _html_tabla(p.tabla_rend.rename(columns={
                "archivo": "Archivo", "campaña": "Campaña", "cultivo": "Cultivo", "columna": "Columna",
                "n_puntos": "N° puntos", "media": "Media", "desvio": "Desvío", "min": "Mín", "max": "Máx",
                "metodo": "Interpolación"})) if p.tabla_rend is not None else None,
            "tabla_validacion": _html_tabla(p.validacion.assign(
                cumple=p.validacion["cumple"].map({True: "Sí", False: "NO", None: "n/d"}))),
            "guia": _md_html("guia_interpretacion.md"),
            "metodologia": _md_html("metodologia.md"),
        })
    env = Environment(loader=FileSystemLoader(os.path.join(DIR, "plantillas")),
                      autoescape=select_autoescape(enabled_extensions=()))
    html = env.get_template("informe.html").render(**ctx)
    path_html = os.path.join(carpeta, "informe_ambientes.html")
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html)
    path_pdf = os.path.join(carpeta, "informe_ambientes.pdf")
    try:
        import logging
        from weasyprint import HTML
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        HTML(string=html, base_url=carpeta).write_pdf(path_pdf)
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo generar el PDF ({e}). El HTML está disponible.")
        path_pdf = None
    return path_html, path_pdf


def exportar(p, carpeta):
    """GPKG, SHP, GeoTIFF, Excel de estadísticas y ZIP con todo."""
    os.makedirs(carpeta, exist_ok=True)
    base = f"{p.cfg.establecimiento}_{p.cfg.lote}".replace(" ", "_")
    rutas = []

    gpkg = os.path.join(carpeta, f"{base}_ambientes.gpkg")
    p.gdf.to_file(gpkg, layer="ambientes", driver="GPKG")
    rutas.append(gpkg)
    shp_dir = os.path.join(carpeta, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    shp = os.path.join(shp_dir, f"{base}_ambientes.shp")
    p.gdf.rename(columns={"ambiente_id": "amb_id", "ambiente": "ambiente", "potencial": "potenc"}).to_file(
        shp, encoding="utf-8")
    rutas += [os.path.join(shp_dir, f) for f in os.listdir(shp_dir)]

    mag = {k: p.stack.magnitud(k) for k in p.stack.disponibles}
    tif_mag = os.path.join(carpeta, f"{base}_capas_magnitud.tif")
    p.stack.exportar_geotiff(tif_mag, mag)
    tif_norm = os.path.join(carpeta, f"{base}_capas_normalizadas.tif")
    p.stack.exportar_geotiff(tif_norm, p.stack.norm)
    tif_cls = os.path.join(carpeta, f"{base}_clasificacion.tif")
    p.stack.exportar_geotiff(tif_cls, {"ambiente": np.where(p.raster_final > 0, p.raster_final, np.nan)})
    rutas += [tif_mag, tif_norm, tif_cls]

    xlsx = os.path.join(carpeta, f"{base}_estadisticas.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        p.gdf.drop(columns="geometry").to_excel(w, sheet_name="ambientes", index=False)
        tabla_estadisticas_informe(p).to_excel(w, sheet_name="resumen_informe")
        p.est_largo.to_excel(w, sheet_name="estadisticas", index=False)
        pd.DataFrame({k: pd.Series(v) for k, v in {"descripcion": p.textos}.items()}).to_excel(
            w, sheet_name="descripciones")
        p.tabla_img.to_excel(w, sheet_name="imagenes", index=False)
        if p.tabla_rend is not None:
            p.tabla_rend.to_excel(w, sheet_name="rendimiento", index=False)
        p.k_df.to_excel(w, sheet_name="seleccion_k")
        p.explic["tabla"].to_excel(w, sheet_name="importancia_variables")
        p.validacion.to_excel(w, sheet_name="validacion_topologica", index=False)
    rutas.append(xlsx)
    return rutas


def empaquetar(carpeta, nombre_zip):
    path = os.path.join(os.path.dirname(os.path.abspath(carpeta)), nombre_zip)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for raiz, _, archivos in os.walk(carpeta):
            for a in archivos:
                full = os.path.join(raiz, a)
                z.write(full, os.path.relpath(full, carpeta))
    return path
