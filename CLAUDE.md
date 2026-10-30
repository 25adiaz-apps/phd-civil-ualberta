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
- Código (variables y comentarios) en inglés; texto explicativo en español.
- Fórmulas en LaTeX, con el significado de cada símbolo.
- Semilla fija para cualquier dato aleatorio.
- Ejecutar los notebooks antes de cada commit para que las gráficas queden guardadas.
- Commits descriptivos y push al terminar.
- Los datasets grandes no se suben a git.
- Los trabajos existentes se respetan: no se modifican, mueven ni reorganizan sin que yo lo pida explícitamente.
