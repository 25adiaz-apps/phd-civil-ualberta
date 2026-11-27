# Project Phase 3: forma de trabajo

Proyecto de CIV E 789 Sensing Techniques and Data Analytics (Fall 2026): Machine Learning for Anomaly Detection (K-Means Clustering on Healthy Data).
Entrega: martes 8 de diciembre de 2026, 11 PM. Vale el 20 % de la nota (100 puntos). Se califica en forma comparativa (la mejor entrega fija el máximo) y la entrega tardía pierde 10 % por día, hasta tres días.

Este archivo guarda cómo trabajamos en el proyecto. Las instrucciones nuevas que vayan saliendo se agregan aquí.

## Antecedentes

- **Phase 1** (enunciado y entrega en esta carpeta): puente de la Highway 43, continuo de tres vanos (20 m – 40 m – 60 m, apoyos en 0, 20, 60 y 120 m), con 24 acelerómetros cada 5 m. Análisis en el tiempo y en frecuencia de ensayos de impacto y ambientales (descriptores, IRF, FFT, PSD de Welch, FDD) para los escenarios intact, damage1 y damage2.
- **Phase 2** (`../project-phase2/`, con su propio `FORMA_DE_TRABAJO.md`): detección estadística de daño con seis características (f1, f2, f3, MAC1, MAC2, MAC3) y 20 estimaciones por escenario: z-score, Wasserstein, Mahalanobis, combinaciones de 4 características y PCA.
- **Phase 3**: detector de anomalías con k-means entrenado solo con datos sanos (200 registros intact; 50 por cada escenario de daño). Damage1 es el escenario moderado y damage2 el severo.

## Archivos

```
project-phase3/
├── ProjectPhase3_KMeans_Anomaly_Detection_Fall2026.pdf   Enunciado de la Phase 3
├── ProjectPhase1_*.pdf, ProjectPhase2_*.pdf              Enunciados y entregas de las phases anteriores
├── FORMA_DE_TRABAJO.md                                   Este archivo
└── ProjectPhase3_Group6/        Contenido del ZIP que se entrega
    ├── Main.ipynb               Notebook con todo el análisis (Partes A–C y nota sobre el uso de IA)
    ├── phase3_modal_data.json   Datos (intact: 200 registros; damage1 y damage2: 50 cada uno)
    ├── phase3_kmeans_starter.py Código de inicio del profesor
    ├── README.txt               Descripción breve del contenido, en inglés (al final)
    ├── figures/                 Figuras que genera el notebook
    └── results/                 Tablas que genera el notebook (CSV)
```

Se entregan dos archivos: `ProjectPhase3_Group6.zip` (la carpeta de arriba) y `ProjectPhase3_Group6.pdf` (el notebook exportado, con todas las salidas visibles). El PDF se exporta con el `export_pdf.py` de la Phase 2, adaptado a esta carpeta.

## Grupo 6

Integrantes: Amilcar Diaz (`adiazgar`) y Alexander Jordan (`ajordan2`).

| Sensor | S2 | S4 | S6 | S9 | S12 | S15 | S18 | S21 | S24 |
|---|---|---|---|---|---|---|---|---|---|
| Posición (m) | 7.5 | 17.5 | 27.5 | 42.5 | 57.5 | 72.5 | 87.5 | 102.5 | 117.5 |

Cada forma modal es una lista de 24 valores ordenados de S1 a S24, así que el sensor Sk está en el índice k − 1.

## Reglas del enunciado

- **Solo nuestros sensores.** Las formas modales, y por tanto los MAC, usan solo los 9 sensores del grupo 6.
- **Solo datos sanos para todo lo que se ajusta.** El escalador, las formas modales de referencia del MAC, los centroides, el número de clusters y el umbral salen de registros intact. Los registros dañados solo se usan para probar el detector terminado.
- **Tres subconjuntos sanos.** Los 200 registros intact se dividen una sola vez en 100 *fit*, 50 *calibration* y 50 *test*, con semilla = 6 (número de grupo; lo hace el starter code). Se ajusta con *fit*, el umbral se fija con *calibration* y los resultados se reportan con *test*.
- **Dos detectores.** Detector F: f1, f2, f3. Detector FM: f1, f2, f3, MAC1, MAC2, MAC3.
- El clustering se hace con `sklearn.cluster.KMeans` (`n_init` ≥ 20 y `random_state` fijo); el puntaje, el umbral y la evaluación se programan con NumPy y `sklearn.metrics`.
- Todas las explicaciones van en celdas Markdown, no en `print` ni en comentarios del código.
- Todas las figuras con ejes rotulados, título y leyenda cuando hay varias condiciones.
- Las respuestas de la Parte A van con nuestras palabras, en unas 3 a 6 oraciones; A1 lleva además el cálculo a mano.
- Las respuestas de la Parte C se apoyan en números de nuestros resultados.
- Al final del notebook va una nota sobre el uso de IA (para qué se usó y los prompts principales), una por integrante.
- Referencia del enunciado: Capítulo 7 del libro del curso, sección *K-Means Clustering for Anomaly Detection*.

## Cómo trabajamos

- **Poco a poco.** El proyecto no se desarrolla de inicio a fin de una vez. Agrego una sección (o la parte que pidas), explico qué hace el código y qué muestran los resultados, y espero tu revisión antes de seguir.
- **Solo lo que se pide.** Se hace exactamente lo que se pide, sin secciones, tablas, funciones ni sugerencias adicionales. Respuestas cortas.
- **Código lo más simple posible.** Legible para alguien que empieza en Python: pasos comentados, bucles e `if` sencillos, sin trucos.
- **Estructura del notebook.** Se sigue la de la Phase 2: encabezado (título, curso, profesor, grupo, datos, integrantes e índice), sección 0 de Setup, una sección por parte (`## Part X`) y una subsección por tarea (`### X1. ...`), y al final la nota sobre IA.
- **Idioma.** El código (variables y comentarios) va en inglés. El texto explicativo del notebook lo trabajamos primero en español; cuando digas que una parte está lista, la traduzco al inglés conservando tu voz. Tablas, figuras, pies de tabla y figura, rótulos de ejes y ecuaciones van en inglés desde el principio. Las explicaciones en el chat siguen en español.
- **Numeración académica.** Todas las figuras, tablas y ecuaciones se numeran en orden de aparición. El título de cada tabla va arriba ("**Table N.** ..."), el pie de cada figura va abajo ("**Figure N.** ..."), ambos fuera del objeto, y las ecuaciones llevan `\tag{N}`. En el texto se citan como "Table N", "Figure N" y "Eq. N".
- **Fórmulas.** En LaTeX, con el significado de cada símbolo.
- **Semilla fija** para cualquier dato aleatorio (la división de los datos y KMeans).

## Google Drive (trabajo con el compañero)

- El compañero trabaja en Google Colab. Nosotros trabajamos primero en local.
- Carpeta compartida: `CIV E 789/ProjectPhase3_Group6/`, con la misma estructura que `ProjectPhase3_Group6/` en local (sin los `.gitkeep`). En este PC, `CIV E 789` está montada en `G:\.shortcut-targets-by-id\19igiJAqeDixiQzelAH-aNp7HxnJjV69i\CIV E 789`. En Colab, la primera celda del notebook (etiqueta `remove-cell`, no sale en el PDF) monta el Drive y entra a la carpeta del proyecto.
- **Subir:** solo cuando lo pidas, se copia la versión local actual al Drive para que el compañero continúe.
- **Traer:** cuando lo pidas:
  1. Revisar los cambios que el compañero hizo en el Drive.
  2. Traerlos al local.
  3. Verificar que no interfieran con lo nuestro y que estén correctos.
  4. Unificar las versiones.
  5. Subir la versión completa al Drive.

## GitHub

- Cada cambio que se pida se sube a GitHub al terminarlo, con un commit descriptivo y push, manteniendo la misma estructura de carpetas que en local.
- Esos commits incluyen solo los archivos de `sensing-techniques/project-phase3/`. Los cambios pendientes de otras carpetas del repositorio no se incluyen.
