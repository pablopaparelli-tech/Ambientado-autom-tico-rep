# Metodología de delimitación de ambientes

## 1. Área de trabajo y grilla común

El perímetro del lote (SHP o GeoJSON) se reproyecta a la zona UTM correspondiente. Todas las capas se
corregistran a una **grilla común de 10 m** alineada a múltiplos de 10 m, coincidente con la grilla de
Sentinel-2 en esa zona UTM. Las capas de resolución más gruesa (bandas red-edge y SWIR de 20 m, DEM de
30 m, Landsat de 30 m) se remuestrean en forma **bilineal**, por tratarse de variables continuas.
El análisis usa los píxeles cuyo centro cae dentro del lote.

## 2. Índices espectrales (Sentinel-2)

- Colección: `COPERNICUS/S2_SR_HARMONIZED` (reflectancia de superficie, nivel L2A).
- **Control de nubes y sombras:** Cloud Score+ (`cs_cdf` ≥ 0,60) y clasificación de escena SCL (se descartan
  sin dato, saturado, sombra de nube, nubes media/alta probabilidad, cirros y nieve). Una fecha se acepta sólo
  si **no hay ningún píxel contaminado** en el lote más un buffer de 20 m. Si la fecha solicitada no cumple,
  se busca la fecha válida más cercana dentro de ±10 días. En el informe se listan la fecha solicitada y la
  utilizada.
- Índices por fecha:
  - NDVI = (B8 − B4) / (B8 + B4)
  - NDMI = (B8 − B11) / (B8 + B11)
  - CIre (clorofila red-edge) = B7 / B5 − 1
- **Normalización por lote y compuestos temporales:** para cada fecha el índice se expresa relativo a la
  media del lote (valor / media del lote). Con el conjunto de fechas se calculan:
  - el **promedio temporal** (en magnitud para los mapas y relativo para el clustering), y
  - el **coeficiente de variación temporal** (CV % = desvío / media del índice relativo), que mide la
    estabilidad del comportamiento de cada sitio entre fechas y campañas, independientemente de la escala
    del índice en cada fecha.

## 3. Topografía

- DEM: Copernicus GLO-30 (`COPERNICUS/DEM/GLO30`), remuestreado en forma bilineal a 10 m.
- **TPI (índice de posición topográfica)** = elevación − elevación media en un círculo de 150 m de radio.
  Valores positivos indican lomas y valores negativos, bajos. La ventana focal usa datos de fuera del lote,
  por lo que no hay efecto borde.

## 4. Frecuencia de anegamiento (Landsat)

- Colecciones Landsat 5 TM (1990–2011), Landsat 8 OLI (2013–actualidad) y Landsat 9 OLI-2
  (2021–actualidad), Collection 2 Level 2.
- Máscara de nubes, nubes dilatadas, cirros y sombras (QA_PIXEL bits 0–4), máscara de saturación
  (QA_RADSAT) y escalado a reflectancia.
- Compuestos **semestrales** (enero–junio y julio–diciembre) por promedio de las observaciones válidas.
  Los semestres sin observaciones se descartan automáticamente.
- Regla de agua: NDVI ≤ 0,25, NDWI (Gao, NIR–SWIR1) ≥ 0, NDWI (Xu, verde–SWIR1) ≥ −0,05 y
  NDWI (McFeeters, verde–NIR) ≥ −0,15.
- **Frecuencia de anegamiento** (%) = semestres con agua / semestres con dato válido en cada píxel.
  Si un píxel tiene menos de 10 semestres válidos, su valor se reemplaza por la media de los píxeles
  vecinos con dato suficiente (radio de 90 m).

## 5. Rendimiento (opcional)

Los mapas de rendimiento limpios (puntos) se interpolan a la grilla de 10 m por **kriging ordinario**
(variograma esférico, vecindario de 48 puntos, IDW como respaldo). Cada campaña se expresa como
rendimiento relativo a la media del lote. Con varias campañas se calculan el promedio y el CV temporal del
rendimiento relativo. Si hay rendimiento, es el criterio para ordenar los ambientes por potencial.

## 6. Normalización y análisis exploratorio

Cada variable se normaliza dentro del lote con un min-max robusto (percentiles 2–98, recortado a 0–1).
El análisis exploratorio incluye estadísticos descriptivos, histogramas, correlación de Spearman y factor
de inflación de la varianza (VIF), para que el analista decida qué variables entran al clustering y evite
redundancias.

## 7. Clustering

- Las capas normalizadas se suavizan con un filtro de mediana de 3×3 píxeles para reducir el ruido de
  píxel aislado.
- **K-means** (k = 2 a 6). Se evalúan cuatro criterios: codo de la inercia, silueta, Calinski-Harabasz y
  Davies-Bouldin. Cada criterio vota un k, considerando sólo las soluciones en que todos los ambientes
  superan el 5 % de la superficie. Se adopta el k más votado (en caso de empate, el de la silueta).
  El analista puede confirmar o modificar k.

## 8. Variables explicativas

Se entrena un **Random Forest** que reproduce la clasificación y se calculan:
- la **importancia por permutación**,
- los **valores SHAP** (contribución media absoluta de cada variable a la asignación de cada ambiente), y
- la prueba de **Kruskal-Wallis** con su tamaño de efecto ε² (proporción de la variabilidad de cada variable
  explicada por los ambientes).

## 9. Orden y nombre de los ambientes

- **Índice de potencial:** rendimiento relativo si se dispone de él. Si no, la media de NDVI y CIre
  normalizados.
- Los ambientes se numeran en forma ascendente: **1 = mayor potencial productivo**.
- Nombres base: Alta, Media, Baja, Muy Baja. Cuando dos o más ambientes consecutivos difieren en menos de
  0,25 desvíos del índice de potencial, comparten el nombre base y se les agrega un descriptor según la
  variable que más los diferencia, por ejemplo *Media (loma)*, *Baja (inundable)* o *Media (inestable)*.
- Paleta del mapa: verde (mejor ambiente) – amarillo – rojo (peor ambiente).

## 10. Vectorización y generalización topológica

1. Filtro de mayoría de 3×3 píxeles.
2. **Unidad mínima de mapeo (UMM, 0,5 ha por defecto):** toda mancha más chica se fusiona con el ambiente
   vecino con el que comparte más borde.
3. **Suavizado:** la probabilidad de pertenencia a cada ambiente se filtra con un núcleo gaussiano y se
   sobremuestrea ×4 (2,5 m). Cada celda toma la clase más probable. Al ser una partición raster, no
   genera huecos ni superposiciones.
4. Vectorización y **simplificación de cobertura** (Visvalingam-Whyatt sobre la cobertura, con
   `shapely.coverage_simplify`): cada borde compartido se simplifica una sola vez para los dos ambientes
   vecinos.
5. **Recorte topológico al lote:** se nodan juntos todos los bordes de ambientes y el perímetro del lote
   sobre una grilla de 1 mm, se poligonizan las caras y cada cara toma el ambiente que la contiene.
6. Fusión de astillas menores a la UMM y **validación**: geometrías válidas, cobertura válida (bordes
   coincidentes), sin superposiciones, sin huecos, sin superficie fuera del lote, superficie total igual a la
   del lote y ningún polígono menor a la UMM.

## 11. Productos

- Informe en HTML y PDF.
- Polígonos de ambientes (GPKG y SHP).
- GeoTIFF de las capas en magnitud, normalizadas y de la clasificación.
- Excel con las estadísticas por ambiente, la tabla de imágenes y la de rendimiento, la selección de k, la
  importancia de variables y la validación topológica.
