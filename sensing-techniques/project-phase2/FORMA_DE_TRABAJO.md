# Project Phase 2: forma de trabajo

Proyecto de CIV E 789 Sensing Techniques and Data Analytics (Fall 2026): Statistical Damage Detection.
Entrega: jueves 19 de noviembre de 2026, 11 AM. Vale el 20 % de la nota (100 puntos).

Este archivo guarda cómo trabajamos en el proyecto. Las instrucciones nuevas que vayan saliendo se agregan aquí.

## Archivos

```
project-phase2/
├── ProjectPhase2_Statistical_Damage_Detection_Fall2026.pdf   Enunciado
├── FORMA_DE_TRABAJO.md                                        Este archivo
├── export_pdf.py                                              Exporta Main.ipynb a ProjectPhase2_Group6.pdf
├── ProjectPhase2_Group6.pdf                                   PDF del notebook (segundo entregable)
└── ProjectPhase2_Group6/        Contenido del ZIP que se entrega
    ├── Main.ipynb               Notebook con todo el análisis (Partes A–F y nota sobre el uso de IA)
    ├── phase2_modal_parameters.json  Datos (intact, damage1, damage2)
    ├── README.txt               Descripción breve del contenido (en inglés)
    ├── figures/part-a … part-e/ Figuras que genera el notebook, una carpeta por parte
    └── results/                 Tablas que genera el notebook (CSV)
```

Se entregan dos archivos: `ProjectPhase2_Group6.zip` (la carpeta de arriba) y `ProjectPhase2_Group6.pdf` (el notebook exportado, con todas las salidas visibles).

## Grupo 6

Sensores asignados (de `metadata.sensor_assignments.group_6` del JSON) y su posición a lo largo del puente (`metadata.sensor_positions_m`):

| Sensor | S2 | S4 | S6 | S9 | S12 | S15 | S18 | S21 | S24 |
|---|---|---|---|---|---|---|---|---|---|
| Posición (m) | 7.5 | 17.5 | 27.5 | 42.5 | 57.5 | 72.5 | 87.5 | 102.5 | 117.5 |

Cada forma modal es una lista de 24 valores ordenados de S1 a S24, así que el sensor Sk está en el índice k − 1.

## Cómo trabajamos

- **Solo lo que se pide.** Se hace exactamente lo que se pide, nada más: sin secciones, tablas, funciones ni sugerencias adicionales. Respuestas cortas.
- **Estructura simple.** No se crean secciones innecesarias; lo que va junto (p. ej. funciones y el código que las usa) queda en la misma sección.
- **Setup solo extrae datos.** La sección 0 solo carga y extrae los datos del JSON. Lo que cada parte necesite calcular (MAC, medias, desviaciones, etc.) se hace dentro de esa parte.
- **Una sección a la vez.** Agrego las celdas de una sección, explico qué hace el código y qué muestran los resultados, y espero tu revisión antes de pasar a la siguiente.
- **Ejecución.** Antes de cada commit ejecuto `Main.ipynb` completo (`jupyter nbconvert --execute --inplace`), para que en GitHub se vean las tablas y figuras. Si algún cálculo llega a tardar varios minutos, lo aviso antes de ejecutarlo.
- **Idioma.** El texto explicativo del notebook lo escribes en español mientras trabajamos; lo traduzco al inglés cuando lo pidas, conservando tu voz. Tablas, figuras, gráficas, pies de tabla y figura, rótulos de ejes, código y ecuaciones van en inglés desde el principio. Las explicaciones en el chat siguen en español.
- **Numeración académica.** Todas las figuras, tablas y ecuaciones se numeran en orden de aparición, con el formato de los documentos académicos. El título de cada tabla va arriba ("**Table N.** ..."), el pie de cada figura va abajo ("**Figure N.** ..."), ambos fuera del objeto, y las ecuaciones llevan `\tag{N}`.
- **Fórmulas.** Se escriben en LaTeX e indican el significado de cada símbolo.
- **Sensores.** Las formas modales usan solo los 9 sensores del grupo 6 (tabla de arriba).
- **Requisitos del enunciado.** Todos los ejes van rotulados, con leyenda cuando hay varias condiciones y título en cada figura. Las distancias de Wasserstein 1D se calculan con `scipy.stats.wasserstein_distance` y las multivariadas (Partes D y E) con `scipy.stats.wasserstein_distance_nd`.
- **Semilla fija** para cualquier dato aleatorio.

## Google Drive (trabajo con el compañero)

- Carpeta compartida: `CIV E 789/ProjectPhase2_Group6/` ([enlace](https://drive.google.com/drive/folders/18TTWW3c4ZXOr1zfuauUzpvXo1fMnCXYG)). En este PC está montada en `G:\.shortcut-targets-by-id\19igiJAqeDixiQzelAH-aNp7HxnJjV69i\CIV E 789\ProjectPhase2_Group6`.
- Tiene la misma estructura que `ProjectPhase2_Group6/` en local (sin los `.gitkeep`), con `phase2_modal_parameters.json` junto a `Main.ipynb`.
- Trabajamos en local y se sube al Drive solo cuando se pida. Al actualizar:
  1. Revisar los cambios que el compañero hizo en el Drive.
  2. Traerlos al local.
  3. Verificar que no interfieran con lo nuestro y que estén correctos.
  4. Unificar las versiones.
  5. Subir la versión completa al Drive.

## Pendientes por hablar con el compañero

Las partes A y B del compañero son todavía un borrador. La verificación se hace cuando él termine sus partes.

- **ddof (unificado el 11 de noviembre de 2026; avisarle).** Todas las desviaciones estándar se calculan con ddof = 1: A2 (bandas de la Figure 4), A3 (Table 6), B1 (Table 7), B2 (Table 8), C3 (`z_flagged`), D1 y E2. Se actualizaron los números del texto que cambiaron: A3 (0.0087 → 0.0089, 0.0038 → 0.0039), B1 (2.5% → 1.7%, 50% → 48.3%), C3 (5–10 % → 0–10 %, 45–50 % → 40–50 %) y D2 (3.227 / 11.022 → 3.145 / 10.743, 44 % / 33 % → 48 % / 37 %). Excepción a propósito: los ajustes Gaussianos con `stats.norm.fit` de A1 (Figures 1–3) y A3 (Figures 5–7) siguen siendo MLE (ddof = 0), porque el enunciado pide "maximum likelihood estimation"; por eso la σ de las leyendas de las Figures 5–7 es ≈ 2.6 % menor que la de la Table 6.
- **B1, error en el texto:** dice que los MAC detectan el daño "79.2% of the time", pero el promedio de las tasas de los MAC en la Table 7 es 70.0 % con ddof = 1 (40, 50, 55, 85, 95 y 95 %); con el ddof = 0 anterior era 70.8 %. Ya estaba mal antes del cambio de ddof.
- **B3, argumento que ya no se sostiene:** dice que f1 queda por encima de f3 porque f1 detecta 5 % y f3 0 % ("f1 had the second highest correct damage flagged percentage at 5%"). Con ddof = 1 los dos detectan 0 % (Table 7), y B2 no desempata (f1 < f3 en damage1 y f1 > f3 en damage2). Hay que cambiar la redacción, no solo el número.
- **Parte A (revisión del 9 de noviembre de 2026):**
  - Hay dos celdas de Colab. En Colab, la primera (`setup-colab`) falla con un `assert` y detiene "Run all" antes de la segunda (su ruta fija). Dejar una sola.
  - A3 dice que se usó "the reference intact scenario provided", pero el JSON no trae una referencia aparte: el código usa el promedio de las 20 mediciones intactas del grupo 6, como C1. La explicación de por qué el MAC del intacto es menor que 1 (por usar solo 9 sensores) no corresponde al código.
  - A2 y A3 dicen que la σ aumenta con el daño por la degradación del material, pero en A1 la σ de las frecuencias baja con el daño, y C1 explica la dispersión por temperatura, tráfico y errores de identificación.
  - Sobrescribe `data` (el JSON cargado) en sus celdas; hoy no rompe nada, pero es frágil.
  - Cálculo del MAC repetido (`compute_mac` en A3 y `calculate_MAC` en C1) e imports repetidos en cada celda.
  - Cadenas con `\pm` sin `r''`, que dan `SyntaxWarning` en Python 3.12.
  - Notas suyas sin resolver: "Summarize what was done in Part A.", "Please check explanation…", "NEED TO UPDATE AFTER LATER ANALYSIS".
- **Parte B y lo que depende de ella en C–E (revisión del 11 de noviembre de 2026; C3 y D2 ya resueltos):**
  - **E2:** quitar la celda "**PENDING:** Compare with Part B results…" y el aviso "Pending confirmation with the actual Part B results". El texto ya coincide con los números (PC1 normalizado 0.269 en damage1 y 1.107 en damage2: menor que todas las features en damage1, la menor es f1 con 0.352, y mayor solo que las frecuencias en damage2, la mayor es f2 con 0.984).
  - Renumerar las tablas y figuras de C–E cuando se inserten las de B.

## Pendientes de la entrega (revisión del 11 de noviembre de 2026)

- **Figura de E4.** El enunciado pide una figura con dos subplots (uno por escenario de daño) con la distancia de Wasserstein frente al número de PCs, k = 1…6. Hoy solo está la Table 21. Propuesta: una celda nueva después de la Table 21, guardada como `figures/part-e/E4_wasserstein_k_PCs.png` (Figure 18; no hay figuras después, así que no hay que renumerar). El README ya menciona "k components" en `part-e/`.
- **Prefijos en los nombres de archivo de A y B.** Para igualarlos con C–E (`C1_…`, `D1_…`), cambiar el nombre del archivo en 4 celdas del compañero: `mode_X_freq_dist_3sigma.png` → `A1_…`, `mode_shapes_3x3.png` → `A2_…`, `mode_X_mac_distributions.png` y `mac_summary_statistics.csv` → `A3_…`, `univariate_zscore_detection.csv` → `B1_…`, `wasserstein_distance_grouped_bars.png` y `wasserstein_distance_summary.csv` → `B2_…`. Opcional.

## GitHub

- Cada cambio que se pida se sube a GitHub al terminarlo, con un commit descriptivo y push, manteniendo la misma estructura de carpetas que en local.
- Esos commits incluyen solo los archivos de `sensing-techniques/project-phase2/`. Los cambios pendientes de otras carpetas del repositorio no se incluyen.
