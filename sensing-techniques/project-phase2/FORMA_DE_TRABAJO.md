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
- **No recalcular lo de A y B.** Las partes C–E usan los resultados ya obtenidos en A y B (p. ej. `mac_results` de A3, `detection_summary_df` de B1, `w1_dict` de B2); solo se calcula lo que A y B no calcularon.

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

### Errores y contradicciones en A y B (para que el compañero los revise)

Revisión del 9 de noviembre de 2026 (Parte A) y del 11–12 de noviembre de 2026 (Parte B).

1. **A3, referencia del MAC.** El texto dice que se usó "the reference intact scenario provided", pero el JSON no trae una referencia aparte: el código usa el promedio de las 20 mediciones intactas del grupo 6, como C1. Por eso tampoco corresponde al código la explicación de por qué el MAC del intacto es menor que 1 (por usar solo 9 sensores).
2. **A2 y A3, σ con el daño.** Dicen que la σ aumenta con el daño por la degradación del material, pero en A1 la σ de las frecuencias baja con el daño, y C1 explica la dispersión por temperatura, tráfico y errores de identificación.
3. **Celdas de Colab.** Hay dos. En Colab, la primera (`setup-colab`) falla con un `assert` y detiene "Run all" antes de la segunda (su ruta fija). Dejar una sola.
4. **B1, error en el texto.** Dice que los MAC detectan el daño "79.2% of the time", pero el promedio de las tasas de los MAC en la Table 7 es 70.8 % (45, 50, 55, 85, 95 y 95 %). Además, a esa frase le falta "using MAC".
5. **ddof.** A y B usan ddof = 0 (`np.std`, `stats.norm.fit`) y C–E usan ddof = 1 en desviaciones estándar y covarianzas, así que el mismo dato sale con dos valores (p. ej. σ de f1: 0.112 en A y 0.115 en la Table 15). Acordar un solo criterio. El `z_flagged` de C3 usa el mismo criterio que B1 (`x0.std()`, ddof = 0); si se cambia, cambiarlo también ahí.
6. **B1, nivel de confianza.** Dice que |z| > 2 "corresponds to a 95% confidence level"; debería decir "approximately 95%, assuming a normal distribution", porque el 95 % exacto corresponde a 1.96.
7. **Erratas en la Parte B.** "Wassterstein" → "Wasserstein"; "Comparitively" → "Comparatively"; "analysis'" → "analysis" (sin apóstrofo); "have the following observations" → "we have the following observations" (falta el sujeto).
8. **B2, la distancia de Wasserstein usada como z-score.** El texto dice que con un umbral de $Z = 2$ no se detectaría daño en las frecuencias y sí en los MAC, y la Figure 8 tiene esa línea. No es correcto: la distancia de Wasserstein compara las dos distribuciones completas, no clasifica cada medición, así que no tiene un umbral de detección de 2; sirve para medir sensibilidad. Coincide numéricamente con el z-score medio solo porque aquí el daño desplaza sobre todo la media. Hay que quitar la línea de la Figure 8 y las dos frases que hablan de detectar con ese umbral.

### Otros puntos de la Parte A (código)

- Sobrescribe `data` (el JSON cargado) en sus celdas; hoy no rompe nada, pero es frágil.
- Cálculo del MAC repetido (`compute_mac` en A3 y `calculate_MAC` en C1) e imports repetidos en cada celda.
- Cadenas con `\pm` sin `r''`, que dan `SyntaxWarning` en Python 3.12.
- Notas suyas sin resolver: "Summarize what was done in Part A.", "Please check explanation…", "NEED TO UPDATE AFTER LATER ANALYSIS".

### Parte B y lo que depende de ella en C–E (revisión del 11 de noviembre de 2026; C3, D2 y E2 ya resueltos)

- **Decidido el 12 de noviembre de 2026:** E2 normaliza el PC1 con ddof = 1 (se quitó el comentario que pedía igualarlo a la Parte B) y D2 compara el W1,norm de B2 (ddof = 0) con el W de D1 (ddof = 1) tal como está.
- Renumerar las tablas y figuras de C–E cuando se inserten las de B.

## Pendientes de la entrega (revisión del 11 de noviembre de 2026)

- **Notas del compañero que hay que quitar antes de generar los entregables** (encontradas el 12 de noviembre de 2026):
  1. Sección 0 (Setup), celda de texto `25nOxOqlkbTN`, justo después de la primera celda de Colab: "Note: The above code wasnt working for me so I set up a direct path, we can tidy this up after".
  2. Sección 0 (Setup), celda de código `ai5eX0j7kYzb` (su segunda celda de Colab, con la ruta fija `/content/drive/MyDrive/Colab Notebooks/CIV E 789/ProjectPhase2_Group6`), comentarios: "# Using the exact path you located" y "# Change the working directory to your project folder".
  3. A1, celda de código `82SEK_9oVdB_`, comentario: "# Set your starting figure number here based on previous cells".

- **Prefijos en los nombres de archivo de A y B.** Para igualarlos con C–E (`C1_…`, `D1_…`), cambiar el nombre del archivo en 4 celdas del compañero: `mode_X_freq_dist_3sigma.png` → `A1_…`, `mode_shapes_3x3.png` → `A2_…`, `mode_X_mac_distributions.png` y `mac_summary_statistics.csv` → `A3_…`, `univariate_zscore_detection.csv` → `B1_…`, `wasserstein_distance_grouped_bars.png` y `wasserstein_distance_summary.csv` → `B2_…`. Opcional.

## GitHub

- Cada cambio que se pida se sube a GitHub al terminarlo, con un commit descriptivo y push, manteniendo la misma estructura de carpetas que en local.
- Esos commits incluyen solo los archivos de `sensing-techniques/project-phase2/`. Los cambios pendientes de otras carpetas del repositorio no se incluyen.
