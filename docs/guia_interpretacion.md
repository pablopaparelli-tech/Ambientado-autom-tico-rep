# Guía de interpretación de los resultados técnicos

Esta guía explica, en términos simples, las herramientas estadísticas usadas para delimitar los ambientes
y cómo leer sus resultados.

## 1. VIF: ¿hay variables que dicen lo mismo?

El **VIF** (factor de inflación de la varianza) mide cuánto de una variable se puede adivinar a partir de
las otras. Si el NDVI se puede calcular casi exacto con el CIre y el NDMI, el NDVI no aporta información
nueva.

| VIF | Lectura |
|---|---|
| 1 | Totalmente independiente de las demás |
| 1 – 5 | Redundancia baja: está bien |
| 5 – 10 | Redundancia moderada: conviene mirarla |
| más de 10 | Muy redundante: candidata a sacar |

**Por qué importa:** en el agrupamiento cada variable "vota". Si tres variables dicen lo mismo, ese mensaje
vota tres veces y tapa a variables con otra información, como el relieve o el anegamiento. De cada par muy
correlacionado conviene quedarse con la variable más fácil de interpretar agronómicamente.

## 2. Kruskal-Wallis y ε²: ¿los ambientes son realmente distintos?

Para cada variable, la prueba de **Kruskal-Wallis** ordena todos los píxeles del lote de menor a mayor y
mira en qué puestos caen los de cada ambiente. Si el ambiente Alta ocupa siempre los puestos altos y el Baja
los bajos, los ambientes son distintos en esa variable. Al trabajar con puestos y no con valores, no lo
afectan los valores extremos.

- **H:** cuanto más grande, más separados están los ambientes en esa variable.
- **p-valor:** con miles de píxeles casi siempre da significativo; no conviene darle mucho peso.
- **ε² (épsilon cuadrado):** el dato más útil. Es la **proporción de la variabilidad de la variable que
  explican los ambientes**.

| ε² | Lectura |
|---|---|
| menos de 0,06 | Los ambientes casi no difieren en esta variable |
| 0,06 – 0,14 | Diferencia moderada |
| más de 0,14 | Diferencia fuerte: la variable distingue bien los ambientes |

## 3. ¿Cuántos ambientes? Los cuatro criterios

El agrupamiento (k-means) se prueba con 2 a 6 ambientes y cuatro criterios votan cuál es el mejor número.

- **Codo (inercia):** la inercia mide qué tan dispersos están los píxeles dentro de cada ambiente. Siempre
  baja al agregar ambientes. Se busca el punto donde deja de bajar mucho (el "codo"): agregar ambientes
  después de ese punto complica el mapa sin mejorarlo.
- **Silueta (más alto es mejor, de −1 a 1):** compara qué tan parecido es cada píxel a su propio ambiente
  contra el ambiente vecino más cercano. En lotes agrícolas, valores de 0,3 a 0,5 son buenos porque el
  paisaje es continuo.
- **Calinski-Harabasz (más alto es mejor):** separación entre ambientes dividida por la dispersión dentro
  de cada uno. Alto = ambientes compactos y bien separados.
- **Davies-Bouldin (más bajo es mejor):** mide cuánto se superpone cada ambiente con el más parecido.

**Cómo se decide:** primero se descartan las opciones en que algún ambiente ocupa menos del 5 % del lote,
porque un ambiente tan chico no se puede manejar a campo. Luego cada criterio vota y gana el número más
votado (si hay empate, decide la silueta). El analista puede ajustar la elección según su conocimiento del
lote.

## 4. Random Forest: ¿qué variables explican la delimitación?

El agrupamiento forma los ambientes pero no dice *por qué* un sector quedó en uno u otro. Para averiguarlo
se entrena un **Random Forest**, un conjunto de 200 árboles de decisión que aprende a reproducir la
clasificación haciendo preguntas en cadena, por ejemplo: "¿anegamiento mayor a 15 %? → ¿relieve de bajo?
→ ambiente Baja".

- **Exactitud:** se prueba con sectores que el modelo no vio. Si acierta más del 90 %, entendió bien la
  lógica de los ambientes y lo que dice sobre las variables es confiable.
- **Importancia por permutación:** se mezclan al azar los valores de una variable y se mide cuánto cae la
  exactitud. Si cae mucho, esa variable era importante.

## 5. SHAP: ¿cuánto pesa cada variable en cada ambiente?

**SHAP** reparte la decisión del modelo entre las variables, como repartir un premio entre los jugadores de
un equipo según lo que aportó cada uno. Para cada sector responde: *¿cuánto empujó cada variable para que
quede en este ambiente?*

**Cómo leer el gráfico:**
- Cada fila es una variable, ordenadas de más a menos importante.
- El largo total de la barra es la importancia general de la variable.
- Cada color es un ambiente; su tramo indica cuánto se usa esa variable para reconocer ese ambiente.

Por ejemplo, si en la fila "Anegamiento" el tramo rojo (Baja) es largo, el anegamiento es lo que define al
ambiente Baja. SHAP no dice si la variable es alta o baja en el ambiente: para eso se mira la ficha de cada
ambiente o la tabla de estadísticas.

## 6. Orden de lectura recomendado

1. **VIF y correlación:** sacar variables que repiten información.
2. **Codo, silueta, CH y DB:** elegir cuántos ambientes.
3. **Exactitud del Random Forest:** confirmar que la explicación es confiable.
4. **SHAP:** ver qué variables definen cada ambiente.
5. **ε²:** confirmar en qué variables los ambientes son realmente distintos.
6. **Fichas y estadísticas:** ver la dirección (más alto, más bajo, más inundable).
