"""Orquestador: mantiene el estado del análisis y expone una función por etapa."""
import os

import numpy as np
import pandas as pd

from . import (entradas, grilla as mod_grilla, sentinel, topografia, anegamiento, rendimiento,
               eda as mod_eda, clustering, explicabilidad, ambientes, vectorizacion,
               visualizacion, informe)
from .variables import Stack, VARIABLES


class Proyecto:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lote = self.crs = self.lote_ee = self.grilla = self.stack = None
        self.tabla_img_entrada = self.tabla_img = self.tabla_rend = None
        self.serie_anegamiento = None
        self.claves = []
        self.datos = self.k_df = self.votos = self.k_sugerido = self.fig_k = None
        self.k = self.km = self.raster_rank = self.raster_final = None
        self.nombres = self.colores = self.potencial = self.criterio = None
        self.explic = self.gdf = self.validacion = self.raster_gen = None
        self.est_largo = self.est_ancho = self.textos = None

    # ------------------------------------------------------------------ util
    def unidad(self, clave):
        if clave == "rend_rel" and "rend_media" in self.stack.capas and self.tabla_rend is not None:
            return f"rendimiento ({self.tabla_rend['columna'].iloc[0]})"
        return VARIABLES[clave].unidad

    def nombre(self, clave):
        if clave == "rend_rel" and "rend_media" in self.stack.capas:
            return "Rendimiento"
        return VARIABLES[clave].nombre

    # ------------------------------------------------------------ 0. entradas
    def cargar_entradas(self):
        c = self.cfg
        for nombre, ruta in (("perímetro del lote", c.lote_path), ("Excel de fechas", c.excel_path)):
            if not ruta:
                raise FileNotFoundError(f"Falta el {nombre}: subilo en el paso 3a y volvé a ejecutar este paso.")
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"No existe el archivo del {nombre}: {ruta}")
        self.lote, self.crs = entradas.leer_lote(c.lote_path)
        self.tabla_img_entrada, self.tabla_rend = entradas.leer_excel(c.excel_path)
        self.grilla = mod_grilla.crear_grilla(self.lote, c.resolucion_m)
        self.stack = Stack(self.grilla, c)
        print(f"Lote: {self.lote.area.iloc[0] / 1e4:.1f} ha · CRS {self.grilla.epsg} · "
              f"grilla {self.grilla.ancho}×{self.grilla.alto} px · "
              f"{len(self.tabla_img_entrada)} fechas · "
              f"{0 if self.tabla_rend is None else len(self.tabla_rend)} mapas de rendimiento")

    def inicializar_ee(self):
        import ee
        try:
            ee.Initialize(project=self.cfg.gee_project)
        except Exception:  # noqa: BLE001
            ee.Authenticate()
            ee.Initialize(project=self.cfg.gee_project)
        self.lote_ee = entradas.lote_a_ee(self.lote)
        print(f"Earth Engine inicializado (proyecto {self.cfg.gee_project}).")

    # -------------------------------------------------------- 1. sentinel-2
    def capas_sentinel(self):
        self.tabla_img = sentinel.seleccionar_fechas(self.tabla_img_entrada, self.lote_ee,
                                                     self.grilla.epsg, self.cfg)
        validas = self.tabla_img[self.tabla_img["estado"].isin(["fecha solicitada", "fecha cercana"])]
        if validas.empty:
            raise RuntimeError("Ninguna fecha tiene imagen limpia en la ventana de búsqueda.")
        img = sentinel.imagen_indices(validas["fecha_usada"].tolist(), self.lote_ee, self.cfg)
        bandas = mod_grilla.descargar(img, self.grilla)
        self.indices_por_fecha = bandas
        self.stack.agregar(sentinel.compuestos_temporales(bandas, self.grilla.mascara))
        excluidas = len(self.tabla_img) - len(validas)
        print(f"Fechas utilizadas: {len(validas)} · excluidas: {excluidas}")
        return self.tabla_img

    # -------------------------------------------------------- 2. topografía
    def capas_topografia(self):
        img = topografia.imagen_topografia(self.lote_ee, self.grilla, self.cfg)
        self.stack.agregar(mod_grilla.descargar(img, self.grilla))
        print("DEM y TPI descargados.")

    # ------------------------------------------------------- 3. anegamiento
    def capa_anegamiento(self, serie=True):
        img = anegamiento.imagen_anegamiento(self.lote_ee, self.cfg)
        self.stack.agregar(mod_grilla.descargar(img, self.grilla))
        if serie:
            self.serie_anegamiento = anegamiento.resumen_semestres(self.lote_ee, self.cfg)
            print(f"Semestres con observaciones: {len(self.serie_anegamiento)}")
        sv = self.stack.capas["semestres_validos"][self.grilla.mascara]
        print(f"Semestres válidos por píxel: mín {np.nanmin(sv):.0f} · mediana {np.nanmedian(sv):.0f}")

    # -------------------------------------------------------- 4. rendimiento
    def capas_rendimiento(self):
        capas, tabla = rendimiento.procesar_rendimiento(self.tabla_rend, self.cfg.rendimiento_dir,
                                                        self.lote, self.grilla, self.cfg)
        self.stack.agregar(capas)
        self.tabla_rend = tabla
        if tabla is not None:
            print(f"Rendimiento: {len(tabla)} campaña(s) interpoladas.")

    # ----------------------------------------------------- 5. normalización
    def normalizar(self):
        disp = self.stack.normalizar_todo()
        self.claves = list(disp)
        print("Variables disponibles:", ", ".join(disp))
        return disp

    # ---------------------------------------------------------------- 6. EDA
    def eda(self):
        return mod_eda.ejecutar(self.stack)

    # ------------------------------------------------------- 7. clustering
    def evaluar_k(self, claves=None):
        self.claves = list(claves or self.claves)
        self.datos = clustering.preparar(self.stack, self.claves, self.cfg)
        self.k_df, self.votos, self.k_sugerido = clustering.evaluar_k(self.datos, self.cfg)
        self.fig_k = clustering.grafico_k(self.k_df, self.votos, self.k_sugerido)
        return self.k_df, self.votos, self.k_sugerido

    def clasificar(self, k=None):
        self.k = int(k or self.k_sugerido)
        self.km, raster = clustering.ajustar(self.datos, self.k, self.cfg)
        self.raster_rank, self.potencial, sd, self.criterio = ambientes.rankear(raster, self.stack)
        self.nombres = ambientes.nombrar(self.raster_rank, self.stack, self.potencial, sd, self.cfg)
        self.colores = ambientes.colores(self.k)
        return self.nombres

    # ---------------------------------------------------- 8. explicabilidad
    def explicar(self):
        etiquetas = self.raster_rank[self.datos.filas, self.datos.cols]
        self.explic = explicabilidad.explicar(self.datos, etiquetas, self.colores, self.nombres, self.cfg)
        return self.explic

    # ----------------------------------------------------- 9. vectorización
    def vectorizar(self):
        self.gdf, self.validacion, self.raster_gen = vectorizacion.vectorizar(
            self.raster_rank, self.grilla, self.lote, self.cfg, self.nombres, self.colores,
            self.potencial)
        self.raster_final = vectorizacion.rasterizar(self.gdf, self.grilla)
        sup = self.gdf.set_index("ambiente_id")["sup_ha"]
        self.est_largo, self.est_ancho = ambientes.estadisticas(self.raster_final, self.stack, self.nombres,
                                                               self.tabla_rend, sup)
        self.textos = ambientes.descripciones(self.raster_final, self.stack, self.nombres, superficies=sup)
        return self.gdf, self.validacion

    # ------------------------------------------------------------ 10. visor
    def visor(self, normalizadas=False):
        return visualizacion.visor(self.stack, self.lote, self.gdf, normalizadas)

    # ---------------------------------------------------------- 11. informe
    def informe(self):
        carpeta = os.path.join(self.cfg.salida_dir, f"{self.cfg.establecimiento}_{self.cfg.lote}".replace(" ", "_"))
        html, pdf = informe.generar(self, carpeta)
        rutas = informe.exportar(self, carpeta)
        zip_path = informe.empaquetar(carpeta, os.path.basename(carpeta) + ".zip")
        print("Informe HTML:", html)
        print("Informe PDF :", pdf)
        print("ZIP         :", zip_path)
        return {"html": html, "pdf": pdf, "zip": zip_path, "archivos": rutas}
