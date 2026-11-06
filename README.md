# Ambientado automático de lotes agrícolas

Delimitación de ambientes productivos con **Google Earth Engine** y **machine learning**, pensada para
ejecutarse en **Google Colab**, con informe en HTML y PDF.

## Uso rápido

1. Abrí `notebooks/Ambientacion_Colab.ipynb` en Colab.
2. Guardá un token de GitHub de solo lectura en **Secrets (🔑)** con el nombre `GITHUB_TOKEN`.
   No lo pegues en el código.
3. Ejecutá las celdas en orden. Subí el perímetro del lote (SHP en .zip o GeoJSON), el Excel de fechas y
   cultivos (plantilla: `entrada_fechas_rendimiento.xlsx`) y, si hay, los mapas de rendimiento.
4. El notebook se detiene para que el analista revise el EDA, el número de ambientes y el resultado antes
   de generar el informe.

## Capas

| Capa | Fuente | Paleta |
|---|---|---|
| NDVI (promedio y CV temporal) | Sentinel-2 L2A | rojo → verde |
| NDMI (promedio y CV temporal) | Sentinel-2 L2A | naranja → amarillo → cian |
| Clorofila red-edge CIre (promedio y CV temporal) | Sentinel-2 L2A | blanco → verde |
| Altimetría | Copernicus DEM GLO-30 | verde → amarillo → marrón |
| TPI (radio 150 m) | Copernicus DEM GLO-30 | turbo |
| Frecuencia de anegamiento | Landsat 5/8/9 | blanco → azul |
| Rendimiento relativo (opcional) | Monitor de rendimiento | rojo → verde |
| Ambientes | — | verde (mejor) → amarillo → rojo (peor) |

La metodología completa está en [`docs/metodologia.md`](docs/metodologia.md).

## Estructura

```
ambientacion/        paquete Python (una etapa por módulo; flujo.py orquesta)
  plantillas/        plantilla HTML del informe
notebooks/           notebook de Colab
entrada_fechas_rendimiento.xlsx   plantilla Excel de entrada
docs/                metodología
tests/               prueba de punta a punta con datos sintéticos (sin GEE)
```

## Prueba local (sin Earth Engine)

```bash
pip install -r requirements.txt
python -m tests.test_sintetico
```
