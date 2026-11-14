"""Traducción de los resultados a lenguaje llano para productores y agrónomos.

Genera los nombres simples de las variables, los epígrafes de los mapas, las
fichas por ambiente (comparación con el promedio del lote en palabras) y las
conclusiones automáticas del resumen.
"""
from dataclasses import dataclass

import numpy as np

from .variables import VARIABLES


@dataclass
class Simple:
    nombre: str
    epigrafe: str
    tipo: str            # relativo | nivel | relieve | altura | anegamiento | estabilidad
    principal: bool      # True = va en el cuerpo del informe; False = anexo técnico
    palabras: tuple = ("mayor", "menor")  # para tipo "nivel"


SIMPLE = {
    "ndvi_rel": Simple("Vigor del cultivo (NDVI)",
                       "Promedio de las fechas analizadas. Verde = más vigor; rojo = menos vigor.",
                       "relativo", True),
    "cire_rel": Simple("Clorofila (índice red-edge)",
                       "Relacionado con el estado nutricional del cultivo. Verde intenso = más clorofila.",
                       "relativo", True),
    "ndmi_rel": Simple("Humedad del cultivo (NDMI)",
                       "Contenido de agua en las hojas. Celeste = más húmedo; naranja = más seco.",
                       "nivel", True, ("más húmedo", "más seco")),
    "rend_rel": Simple("Rendimiento",
                       "Rendimiento de los monitores de cosecha. Verde = rinde más; rojo = rinde menos.",
                       "relativo", True),
    "tpi": Simple("Posición en el relieve",
                  "Altura respecto del entorno inmediato (150 m). Rojo = lomas; azul = bajos.",
                  "relieve", True),
    "dem": Simple("Altimetría",
                  "Altura sobre el nivel del mar. Marrón = partes altas; verde = partes bajas.",
                  "altura", True),
    "anegamiento": Simple("Anegamiento",
                          "Porcentaje de semestres con agua en superficie desde 1990. Azul intenso = se anega seguido.",
                          "anegamiento", True),
    "ndvi_cv": Simple("Estabilidad del vigor entre fechas",
                      "Variación del NDVI entre fechas. Rojo = comportamiento cambiante.", "estabilidad", False),
    "ndmi_cv": Simple("Estabilidad de la humedad entre fechas",
                      "Variación del NDMI entre fechas. Rojo = comportamiento cambiante.", "estabilidad", False),
    "cire_cv": Simple("Estabilidad de la clorofila entre fechas",
                      "Variación del CIre entre fechas. Rojo = comportamiento cambiante.", "estabilidad", False),
    "rend_cv": Simple("Estabilidad del rendimiento entre campañas",
                      "Variación del rendimiento entre campañas. Rojo = rinde distinto según el año.",
                      "estabilidad", False),
}

ORDEN_FICHA = ["rend_rel", "ndvi_rel", "cire_rel", "ndmi_rel", "tpi", "dem", "anegamiento"]
UMBRAL_Z = 0.5


def _stats(p, clave, sel):
    mag = p.stack.magnitud(clave)
    m = p.stack.grilla.mascara & np.isfinite(mag)
    lote = mag[m]
    vals = mag[m & sel]
    if vals.size == 0:
        return None
    sd = lote.std()
    z = (vals.mean() - lote.mean()) / sd if sd > 0 else 0.0
    return vals.mean(), lote.mean(), z


def _frase(p, clave, a, L, z):
    """Devuelve (texto, tendencia) con tendencia en {'mayor', 'menor', 'similar'}."""
    s = SIMPLE[clave]
    if s.tipo == "relativo":
        pct = 100 * (a - L) / abs(L) if L else 0.0
        if abs(pct) < 3:
            return "similar al promedio del lote", "similar"
        return (f"{abs(pct):.0f} % {'mayor' if pct > 0 else 'menor'} que el promedio del lote",
                "mayor" if pct > 0 else "menor")
    if s.tipo == "nivel":
        if abs(z) < UMBRAL_Z:
            return "similar al promedio del lote", "similar"
        return (f"{s.palabras[0] if z > 0 else s.palabras[1]} que el promedio del lote",
                "mayor" if z > 0 else "menor")
    if s.tipo == "relieve":
        if z > UMBRAL_Z:
            return f"loma ({a:+.2f} m respecto del entorno)", "mayor"
        if z < -UMBRAL_Z:
            return f"bajo ({a:+.2f} m respecto del entorno)", "menor"
        return f"media loma ({a:+.2f} m respecto del entorno)", "similar"
    if s.tipo == "altura":
        d = a - L
        if abs(d) < 0.2:
            return "a la altura media del lote", "similar"
        return (f"{abs(d):.1f} m {'por encima' if d > 0 else 'por debajo'} de la altura media del lote",
                "mayor" if d > 0 else "menor")
    if s.tipo == "anegamiento":
        if clave in p.stack.excluidas and a < 1:
            return "sin anegamiento detectado", "similar"
        if a < 5:
            return f"anegamiento raro ({a:.0f} % de los semestres)", "menor" if z < 0 else "similar"
        if a < 20:
            return f"anegamiento ocasional ({a:.0f} % de los semestres)", "similar" if abs(z) < UMBRAL_Z else (
                "mayor" if z > 0 else "menor")
        return f"anegamiento frecuente ({a:.0f} % de los semestres)", "mayor"
    if s.tipo == "estabilidad":
        if abs(z) < UMBRAL_Z:
            return "estabilidad similar al promedio", "similar"
        return ("menos estable entre fechas" if z > 0 else "más estable entre fechas",
                "mayor" if z > 0 else "menor")
    return "", "similar"


def _etiquetas(p, sel, n=3):
    """Rasgos cortos que más distinguen al ambiente (p. ej. 'loma · sin anegamiento')."""
    rasgos = []
    for k in p.stack.disponibles:
        st = _stats(p, k, sel)
        if st is None or k in p.stack.excluidas:
            continue
        _, _, z = st
        if abs(z) >= UMBRAL_Z:
            alto, bajo = VARIABLES[k].descriptor
            rasgos.append((abs(z), alto if z > 0 else bajo))
    vistos, out = set(), []
    for _, palabra in sorted(rasgos, reverse=True):
        if palabra not in vistos:
            vistos.add(palabra)
            out.append(palabra)
    return out[:n]


def nombre_simple(p, clave):
    if clave == "rend_rel" and "rend_media" not in p.stack.capas:
        return "Rendimiento relativo"
    return SIMPLE[clave].nombre if clave in SIMPLE else VARIABLES[clave].nombre


def unidad_simple(p, clave):
    if clave == "rend_rel":
        return p.unidad(clave) if "rend_media" in p.stack.capas else "% del promedio del lote"
    if clave == "anegamiento":
        return "% de semestres con agua"
    if clave == "tpi":
        return "m respecto del entorno"
    return p.unidad(clave)


def fichas(p):
    """Una ficha por ambiente con su descripción en palabras."""
    out = []
    for _, r in p.gdf.sort_values("ambiente_id").iterrows():
        amb = int(r["ambiente_id"])
        sel = p.raster_final == amb
        filas = []
        for k in ORDEN_FICHA:
            if k not in p.stack.disponibles:
                continue
            st = _stats(p, k, sel)
            if st is None:
                continue
            texto, tendencia = _frase(p, k, *st)
            filas.append({"variable": nombre_simple(p, k), "texto": texto, "tendencia": tendencia})
        cvs = [k for k in ("ndvi_cv", "cire_cv", "ndmi_cv", "rend_cv") if k in p.stack.norm]
        if cvs:
            zs = [_stats(p, k, sel)[2] for k in cvs if _stats(p, k, sel)]
            texto, tendencia = _frase(p, "ndvi_cv", 0, 0, float(np.mean(zs)))
            filas.append({"variable": "Estabilidad entre fechas", "texto": texto, "tendencia": tendencia})
        out.append({"id": amb, "nombre": r["ambiente"], "color": r["color"], "sup_ha": r["sup_ha"],
                     "sup_pct": r["sup_pct"], "etiquetas": _etiquetas(p, sel), "filas": filas})
    return out


def conclusiones(p, fichas_):
    c = []
    sup = p.lote.area.iloc[0] / 10_000
    k = len(fichas_)
    c.append(f"El lote ({sup:.0f} ha) se dividió en {k} ambientes de manejo, ordenados del 1 "
             f"(mayor potencial productivo) al {k} (menor potencial).")
    mejor, peor = fichas_[0], fichas_[-1]
    for f, rotulo in ((mejor, "mayor"), (peor, "menor")):
        rasgos = ", ".join(f["etiquetas"]) if f["etiquetas"] else "sin rasgos muy marcados"
        c.append(f"El ambiente {f['nombre']} ({f['sup_ha']:.0f} ha, {f['sup_pct']:.0f} % del lote) "
                 f"es el de {rotulo} potencial: {rasgos}.")
    clave = "rend_rel" if "rend_rel" in p.stack.disponibles else (
        "ndvi_rel" if "ndvi_rel" in p.stack.disponibles else None)
    if clave:
        a = _stats(p, clave, p.raster_final == mejor["id"])
        b = _stats(p, clave, p.raster_final == peor["id"])
        if a and b and b[0]:
            dif = 100 * (a[0] - b[0]) / abs(b[0])
            que = "rendimiento" if clave == "rend_rel" else "vigor del cultivo (NDVI)"
            c.append(f"El {que} en el ambiente {mejor['nombre']} es {dif:.0f} % mayor que en el "
                     f"ambiente {peor['nombre']}.")
    if "anegamiento" in p.stack.disponibles:
        if "anegamiento" in p.stack.excluidas:
            c.append("No se detectaron diferencias de anegamiento dentro del lote en el período analizado.")
        else:
            peores = [(_stats(p, "anegamiento", p.raster_final == f["id"]), f) for f in fichas_]
            frec = [(s[0], f) for s, f in peores if s and s[0] >= 20]
            if frec:
                v, f = max(frec, key=lambda t: t[0])
                c.append(f"El ambiente {f['nombre']} se anega con frecuencia ({v:.0f} % de los semestres): "
                         "conviene considerarlo en la elección de cultivos, fechas y densidades.")
    return c


def como_se_hizo(p):
    k = len(p.nombres)
    fuentes = ["imágenes satelitales Sentinel-2 de las fechas y cultivos indicados"]
    if "rend_rel" in p.stack.disponibles:
        fuentes.append("los mapas de rendimiento")
    return (f"Se combinaron {', '.join(fuentes)}, el relieve del terreno y la historia de anegamiento del "
            "lote desde 1990 (satélites Landsat). Con esa información se agruparon automáticamente las zonas "
            f"del lote que se comportan de manera parecida, dando como resultado {k} ambientes. Cada ambiente "
            "se ordenó según su potencial productivo y se dibujó como un polígono suavizado, sin manchas "
            f"menores a {p.cfg.umm_ha:g} ha, listo para usar en la planificación. El detalle técnico se "
            "encuentra en el anexo.")
