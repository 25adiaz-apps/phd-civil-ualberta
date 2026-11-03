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
7. Commit descriptivo y push, solo con los archivos del artículo.

### Estructura del notebook (en este orden)

0. **Encabezado:** título, referencia completa (autores, año, revista, DOI), el artículo en una frase e índice con enlaces a las secciones. Después, una celda de código con imports, semilla y funciones auxiliares.
1. **Problema:** qué limitación o vacío existe y por qué importa.
2. **Propósito:** qué busca el artículo y su aporte principal.
3. **Metodología:** los conceptos y métodos en que se basa, con un esquema general del enfoque.
4. **Procedimiento:** paso a paso lo que hicieron (modelo, datos, configuración, parámetros).
5. **Resultados:** paso a paso lo que obtuvieron, con los valores que reportan.
6. **Conclusiones y limitaciones:** las conclusiones principales del artículo y sus limitaciones (las que declaran los autores y las que se identifiquen al analizarlo, indicando cuáles son cuáles).

### Cómo se explica

- Explicaciones cortas: párrafos breves y viñetas, sin texto extenso.
- En Metodología, Procedimiento y Resultados, cada paso junta: qué se hizo → el concepto usado → fórmula en LaTeX (con el significado de cada símbolo y en palabras) → código con un ejemplo → gráfica o esquema → interpretación breve. Usar todos los recursos que ayuden a entender (fórmulas, código, gráficas, esquemas, tablas).
- Citar las ecuaciones, figuras y tablas del artículo por su número (p. ej. "Ec. 5 del artículo", "Fig. 3 del artículo").
- Las gráficas y esquemas son propios (matplotlib), no copias de las figuras del artículo. Distinguir siempre lo ilustrativo (datos sintéticos) de lo reportado por el artículo (valores tomados de sus tablas o texto).
- Numerar las tablas, ecuaciones y figuras del notebook en formato académico: título de tabla arriba ("**Tabla N.** ..."), pie de figura abajo ("**Figura N.** ...") y fuera de la gráfica, ecuaciones con `\tag{N}`.
- Texto en español; código (variables y comentarios) en inglés.
- Semilla fija para cualquier dato aleatorio.
- Librerías comunes en `literature-review/undestanding-papers/requirements.txt`; si un artículo necesita otras, su carpeta lleva su propio `requirements.txt`.
