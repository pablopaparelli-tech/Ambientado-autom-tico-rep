"""Parámetros del flujo de ambientación.

Todos los valores tienen un default razonable y se pueden sobrescribir desde
la celda de configuración del notebook.
"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Config:
    # --- Proyecto / identificación ---------------------------------------
    gee_project: str = "ee-pablopaparelli"
    establecimiento: str = "Establecimiento"
    lote: str = "Lote"
    analista: str = ""

    # --- Entradas ----------------------------------------------------------
    lote_path: str = ""            # SHP, ZIP con SHP o GeoJSON del perímetro
    excel_path: str = ""           # Excel con hojas "imagenes" y "rendimiento"
    rendimiento_dir: str = ""      # carpeta donde están los SHP de rendimiento
    salida_dir: str = "salidas"

    # --- Grilla ------------------------------------------------------------
    resolucion_m: float = 10.0

    # --- Sentinel-2 ---------------------------------------------------------
    ventana_dias: int = 10         # búsqueda de fecha limpia ± días
    buffer_nubes_m: float = 20.0   # buffer del lote para el control de nubes
    cs_umbral: float = 0.60        # Cloud Score+ (cs_cdf) mínimo para píxel claro
    scl_invalidas: tuple = (0, 1, 3, 8, 9, 10, 11)  # sin dato, saturado, sombra, nubes, cirros, nieve

    # --- Topografía --------------------------------------------------------
    tpi_radio_m: float = 150.0

    # --- Anegamiento (Landsat) ---------------------------------------------
    l5_inicio: str = "1990-01-01"
    l5_fin: str = "2011-12-31"
    l89_inicio: str = "2013-03-18"
    l89_fin: str = field(default_factory=lambda: date.today().isoformat())
    usar_l9: bool = True
    nubosidad_max_l5: float = 50.0
    nubosidad_max_l89: float = 30.0
    umbral_ndvi: float = 0.25       # agua: ndvi <= umbral
    umbral_ndwi_gao: float = 0.0    # agua: ndwi_gao >= umbral
    umbral_ndwi_xu: float = -0.05   # agua: ndwi_xu  >= umbral
    umbral_ndwi_mc: float = -0.15   # agua: ndwi_mc  >= umbral
    semestres_validos_min: int = 10  # por debajo se rellena con vecinos
    radio_relleno_m: float = 90.0

    # --- Rendimiento -------------------------------------------------------
    interpolacion: str = "kriging"   # "kriging" o "idw"
    max_puntos_kriging: int = 3000

    # --- Normalización / clustering ---------------------------------------
    percentiles_norm: tuple = (2, 98)
    filtro_mediana_px: int = 3
    k_min: int = 2
    k_max: int = 6
    area_min_ambiente_pct: float = 5.0
    semilla: int = 42

    # --- Nombres de ambientes ---------------------------------------------
    umbral_similitud_sd: float = 0.25  # diferencia de potencial (en SD) para "parecidos"

    # --- Vectorización -----------------------------------------------------
    umm_ha: float = 0.5
    factor_suavizado: int = 4          # sobremuestreo de la clasificación para suavizar
    sigma_suavizado_px: float = 1.0    # sigma gaussiano (en píxeles de 10 m)
    tolerancia_simplificacion_m: float = 3.0

    @property
    def pixel_ha(self) -> float:
        return self.resolucion_m ** 2 / 10_000

    @property
    def umm_px(self) -> int:
        return max(1, int(round(self.umm_ha / self.pixel_ha)))
