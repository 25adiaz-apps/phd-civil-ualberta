# Literature review

Material de apoyo para la revisión de literatura.

```
literature-review/
├── metodos/                  Explicación conceptual de cada método (qué es, cómo funciona, ejemplos con código y gráficas)
│   ├── metodos.ipynb
│   ├── research_map_methods.ipynb   Métodos en el orden del mapa de investigación (drive-by)
│   ├── research_map_methods_simple.ipynb   Los mismos métodos explicados en lenguaje sencillo
│   ├── requirements.txt      Librerías de los notebooks (numpy, scipy, matplotlib, EMD-signal, ssqueezepy)
│   └── prompt_explicar_metodos.md   Instrucciones para el chat de Claude: explicar un método y generar el prompt para agregarlo al notebook
├── revision-papers/          Resumen de cada artículo de la revisión (qué hicieron, cómo y para qué usaron cada método)
│   ├── resumenes_papers.ipynb
│   ├── figuras/              Esquemas propios de cada artículo, en PNG
│   └── requirements.txt      Librerías del notebook (numpy, matplotlib)
└── undestanding-papers/      Un notebook por artículo para entenderlo a fondo (ver su README.md)
```

## Resúmenes de artículos

[`revision-papers/resumenes_papers.ipynb`](revision-papers/resumenes_papers.ipynb) tiene un resumen por artículo (sección de nivel 1 con ancla e índice arriba): referencia, problema, pregunta, configuración y procedimiento paso a paso (citando las figuras del artículo), tabla de métodos y propósito, validación, limitaciones y conclusiones. Las figuras son esquemas propios generados con matplotlib (no copias de las del artículo) y se guardan también en `revision-papers/figuras/`. No explica cómo funciona cada método, sino que enlaza al notebook de métodos con rutas relativas (`../metodos/metodos.ipynb#ancla`).

| Autores | Año | Título corto |
|---|---|---|
| Malekjafarian y OBrien | 2017 | Uso de un vehículo que pasa para estimar las formas modales de un puente |
| Mei et al. | 2021 | Formas modales con vehículos a velocidad de tráfico mediante completado de matrices dispersas |
| Talebi-Kalaleh et al. | 2025 | Monitoreo por crowdsensing con una red dispersa de sensores móviles aleatorios |

**Regla para cada artículo nuevo** en `resumenes_papers.ipynb`:

- (a) Tener al inicio de su encabezado un ancla `<a id="apellido-año"></a>` (por ejemplo, `<a id="mei-2021"></a>`). No cambiar los ids existentes, porque otros documentos los enlazan.
- (b) Agregarse al **Índice** (celda `## Índice`, ancla `#indice`) en el mismo orden en que aparece en el cuaderno, con el formato `N. [Autor (año) — título corto](#apellido-año)`.
- (c) Ir separado del artículo anterior con una celda Markdown que contenga solo `---`.
- (d) Terminar con la línea `[↑ Volver al índice](#indice)`.

## Notebook de métodos

[`metodos/metodos.ipynb`](metodos/metodos.ipynb) organiza los métodos **por función**: cada función es una sección de nivel 1 y cada método una subsección dentro de ella. Al inicio hay una tabla de contenido con todas las funciones y métodos.

| Función | Método |
|---|---|
| 0. Fundamentos: señal, forma modal y amplitud | Concepto base (no es un método) |
| 1. Descomposición de señales | 1.1 EMD (Empirical Mode Decomposition) |
| 2. Envolvente de amplitud / amplitud instantánea | 2.1 Transformada de Hilbert |
| 3. Sensado disperso | 3.1 Mapeo a puntos fijos virtuales |
| | 3.2 Completado de matrices: soft imputing |
| 4. Identificación modal | 4.1 SVD para extraer formas modales |

### Flujo para agregar un método

1. En un chat con Claude, pegar las instrucciones de [`metodos/prompt_explicar_metodos.md`](metodos/prompt_explicar_metodos.md) y pedir la explicación del método.
2. Al terminar, Claude entrega un prompt para Claude Code.
3. Ese prompt se le da a Claude Code, que agrega el método al notebook, lo ejecuta y hace commit y push.

### Convenciones para agregar un método

- Si el método no encaja en ninguna función existente, se crea una nueva sección de función.
- Se actualiza la tabla de contenido del notebook (y la de este README).
- **Nivel de detalle conceptual:** el objetivo es tener claro el concepto, no un tratamiento exhaustivo. Unas 10–12 celdas por método, máximo 3–4 gráficas, sin widgets ni deducciones salvo que se pidan.
- Cada método se explica como una **narrativa corta con orden lógico** (para qué sirve → idea central con su fórmula y gráfica → requisitos o limitación principal → aplicación a puentes → tabla de fortalezas y debilidades), no en bloques separados: en cada paso van juntas la explicación, la fórmula en LaTeX (con el significado de cada símbolo y la descripción en palabras), el código, la gráfica y su interpretación. El detalle está en [`metodos/prompt_explicar_metodos.md`](metodos/prompt_explicar_metodos.md).
- Semilla fija para cualquier dato aleatorio.
- Texto explicativo en español; código (variables y comentarios) en inglés, con datos de ejemplo.
- Antes de hacer commit se ejecuta el notebook completo para que las gráficas queden guardadas:

```bash
pip install -r literature-review/metodos/requirements.txt
jupyter nbconvert --to notebook --execute --inplace literature-review/metodos/metodos.ipynb
```
