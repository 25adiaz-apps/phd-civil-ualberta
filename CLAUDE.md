# CLAUDE.md

Contexto para Claude Code al trabajar en este repositorio.

## Repositorio

- Nombre: phd-civil-ualberta. Antes se llamaba assignment-3-machine-learning; se renombró en octubre de 2026 porque dejó de ser solo del Assignment 3.
- URL: https://github.com/25adiaz-apps/phd-civil-ualberta
- Propósito: repositorio general de mi doctorado en Ingeniería Civil en la University of Alberta: cursos, proyectos, revisión de literatura e investigación.
- Copia local: `C:\Apps\phd-civil-ualberta` (Windows, PowerShell). No usar PowerShell como administrador.

## Entorno de Python

- Python 3.12, entorno virtual en `.venv` en la raíz. Activar con: `.venv\Scripts\activate`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, arquitectura Blackwell (sm_120). Driver 592.27, que soporta como máximo CUDA 13.1.
- PyTorch 2.14.1+cu130 y torchvision 0.29.1+cu130, instalados desde el índice cu130 (fijados en [`requirements-torch.txt`](requirements-torch.txt)):
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
- Nunca instalar torch sin `--index-url`: se instalaría una versión sin soporte de GPU. No usar índices de CUDA mayores que 13.1 mientras no se actualice el driver.
- Verificación:
  `python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.randn(3).cuda())"`
- En código de deep learning, usar `device = "cuda" if torch.cuda.is_available() else "cpu"`. Con 8 GB de memoria, ajustar el tamaño de lote si hay errores de memoria.

## Convenciones generales

Resumen; el detalle está en [`README.md`](README.md) y [`literature-review/README.md`](literature-review/README.md).

- Una carpeta por materia (nombre en inglés, minúsculas y guiones) y, dentro, una subcarpeta por trabajo (p. ej. `machine-learning/assignment-3/`).
- Si un trabajo necesita librerías extra, lleva su propio `requirements.txt` en su carpeta.
- Las herramientas reutilizables entre proyectos van en `sensing-techniques/tools/<herramienta>/` como paquetes instalables (`pip install -e`); los proyectos las importan y guardan sus datos en su propia carpeta (p. ej. `vehicle-bridge-signal-generator`).
- Código (variables y comentarios) en inglés; texto explicativo en español.
- Fórmulas en LaTeX, con el significado de cada símbolo.
- Semilla fija para cualquier dato aleatorio.
- Ejecutar los notebooks antes de cada commit para que las gráficas queden guardadas.
- Commits descriptivos y push al terminar.
- Los datasets grandes no se suben a git.
- Los trabajos existentes se respetan: no se modifican, mueven ni reorganizan sin que yo lo pida explícitamente.

## Comprensión de artículos (`literature-review/undestanding-papers/`)

Objetivo: entender a fondo cada artículo que subo a esa carpeta. Un notebook por artículo. (El nombre de la carpeta, `undestanding`, se mantiene tal cual.)

**Las instrucciones nuevas que dé mientras trabajamos en estos notebooks se agregan a esta sección**, para tenerlas presentes en artículos futuros.

### Flujo al subir un PDF nuevo

1. Leer el artículo completo (todas las páginas, incluidas figuras, tablas y apéndices).
2. Crear la carpeta `AAAA-apellido-descripcion-corta`: año, apellido del primer autor y 3–5 palabras en inglés, todo en minúsculas, con guiones y sin tildes (p. ej. `2021-mei-mode-shapes-sparse-matrix-completion`).
3. Mover el PDF dentro de la carpeta y renombrarlo con el mismo nombre (`<carpeta>.pdf`).
4. Crear `<carpeta>.ipynb` con la estructura de abajo.
5. Ejecutar el notebook (son ejemplos ligeros; sin cálculos largos) para que las gráficas queden guardadas.
6. Agregar el artículo a la tabla de [`literature-review/undestanding-papers/README.md`](literature-review/undestanding-papers/README.md).
7. Commit descriptivo y push.

**Sincronización con GitHub:** todo lo que cambie dentro de `literature-review/undestanding-papers/` se sube a GitHub al terminar cada tarea, con la misma estructura de carpetas que en local. Esto incluye los PDFs recién subidos que aún no tienen notebook (quedan en la raíz de la carpeta hasta procesarlos), los cambios de CLAUDE.md y los README. Los cambios pendientes de otras carpetas del repositorio no se incluyen en esos commits.

### Estructura del notebook (en este orden)

0. **Encabezado:** título, referencia completa (autores, año, revista, DOI), el artículo en una frase e índice con enlaces a las secciones.
   - **Glosario** (justo después del índice, sin número): todos los términos técnicos y siglas que aparecen en el notebook, agrupados por tema (p. ej. monitoreo, dinámica estructural, procesamiento de señales, estadística, índices propios del artículo). Cada término lleva su definición y una breve descripción de qué es (1–2 líneas), con el nombre en inglés si la sigla viene de ahí. Si el término tiene ficha en Conceptos clave, se enlaza.
   - Después, una celda de código con imports, semilla y funciones auxiliares.
1. **Problema:** qué limitación o vacío existe y por qué importa.
2. **Propósito:** qué busca el artículo y su aporte principal.
3. **Conceptos clave:** una subsección por cada método o concepto muy importante del artículo, con una ficha que explique claramente:
   - **Qué es:** definición, significado del nombre o las siglas y de dónde viene.
   - **Para qué sirve.**
   - **Por qué se usó en este artículo.**
   - **Cuándo conviene usarlo.**
   - **Cuándo no conviene:** limitaciones y supuestos que, si no se cumplen, lo invalidan.
   - **Alternativas:** otros métodos que resuelven lo mismo.
4. **Metodología:** cómo aplica el artículo esos conceptos (fórmulas, pasos, código), con un esquema general del enfoque. No repite las fichas; las enlaza.
5. **Procedimiento:** paso a paso lo que hicieron (modelo, datos, configuración, parámetros).
6. **Resultados:** paso a paso lo que obtuvieron, con los valores que reportan.
7. **Conclusiones y limitaciones:** las conclusiones principales del artículo y sus limitaciones (las que declaran los autores y las que se identifiquen al analizarlo, indicando cuáles son cuáles).

### Cómo se explica

- Explicaciones cortas: párrafos breves y viñetas, sin texto extenso.
- Cada concepto o método se **define antes de usarlo**: los términos, en el Glosario; los métodos o conceptos importantes, en su ficha de Conceptos clave. No basta con dar la sigla y los pasos del cálculo.
- En Metodología, Procedimiento y Resultados, cada paso junta: qué se hizo → el concepto usado → fórmula en LaTeX (con el significado de cada símbolo y en palabras) → código con un ejemplo → gráfica o esquema → interpretación breve. Usar todos los recursos que ayuden a entender (fórmulas, código, gráficas, esquemas, tablas).
- Citar las ecuaciones, figuras y tablas del artículo por su número (p. ej. "Ec. 5 del artículo", "Fig. 3 del artículo").
- Las gráficas y esquemas son propios (matplotlib), no copias de las figuras del artículo. Distinguir siempre lo ilustrativo (datos sintéticos) de lo reportado por el artículo (valores tomados de sus tablas o texto).
- Numerar las tablas, ecuaciones y figuras del notebook en formato académico: título de tabla arriba ("**Tabla N.** ..."), pie de figura abajo ("**Figura N.** ...") y fuera de la gráfica, ecuaciones con `\tag{N}`.
- Texto en español; código (variables y comentarios) en inglés.
- Semilla fija para cualquier dato aleatorio.
- Librerías comunes en `literature-review/undestanding-papers/requirements.txt`; si un artículo necesita otras, su carpeta lleva su propio `requirements.txt`.

## Notebook de métodos del mapa

[`literature-review/metodos/research_map_methods.ipynb`](literature-review/metodos/research_map_methods.ipynb) explica los métodos del mapa de investigación (monitoreo indirecto de puentes, drive-by), en el orden del mapa y solo con información de métodos.

- **Numeración jerárquica** según el mapa; cada nivel hereda el número de su padre y los métodos van un nivel por debajo de su paso (p. ej. `2.2.2.1 Wavelet transform (SSWT)`). Niveles: `#` título del notebook, `##` nivel 1, `###` nivel 2, `####` nivel 3, `#####` método. Solo se crean los encabezados necesarios para llegar a cada método; el índice de arriba muestra solo los que existen.

  ```
  1 Bridge response separation
    1.1 Road roughness
    1.2 Vehicle dynamics
  2 Modal identification
    2.1 Natural frequencies
      2.1.1 Signal decomposition
      2.1.2 Bridge components
    2.2 Mode shapes
      2.2.1 Sparse spatial coverage
      2.2.2 Signal decomposition
      2.2.3 Bridge components
      2.2.4 Build the shape along the bridge
  3 Uncontrolled data quality
  4 Damage detection
    4.1 Healthy baseline
      4.1.1 Environmental variability
      4.1.2 Operational variability
    4.2 Continuous monitoring
  ```
- **Separador:** una celda markdown con solo `---` antes de cada encabezado de nivel 1 y 2 (`##`, `###`); no entre las subsecciones internas de un método.
- Cada método termina con **Entradas y salidas** (en negrita, sin número): **Entrada**, **Salida** y **Qué se obtiene o interpreta**.
- Todas las celdas de código llevan la etiqueta `hide-input` en `metadata.tags` y `"jupyter": {"source_hidden": true}` en su metadata; las salidas quedan visibles.
- **No regenerar el notebook:** insertar cada método nuevo en su lugar según el orden del mapa y dejar intacto lo demás.
- PDF sin código: [`literature-review/metodos/export_pdf.ps1`](literature-review/metodos/export_pdf.ps1) (webpdf con playwright y chromium en el `.venv`). El PDF no se sube a git.
