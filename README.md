# phd-civil-ualberta

Repositorio general de mi doctorado en Ingeniería Civil en la University of Alberta: cursos, proyectos, revisión de literatura e investigación.

## Estructura

Cada materia tiene su propia carpeta (nombre de la materia en inglés) y, dentro, una carpeta por cada trabajo.

```
machine-learning/                 Machine Learning for Engineers (CIVE 779)
└── assignment-3/                 Assignment 3: bias/varianza, validación cruzada, SVM
structural-reliability/           Confiabilidad estructural
├── mini-project-2/               Confiabilidad de un tirante de puente atirantado (Monte Carlo)
└── open-question/                Aproximación de primer orden de la tensión de un tirante
sensing-techniques/               Sensing Techniques and Data Analytics (CIV E 789)
└── tools/                        Herramientas reutilizables entre proyectos
    └── vehicle-bridge-signal-generator/  Generador de señales vehículo-puente (drive-by) con sus componentes separadas
literature-review/                Revisión de literatura
├── metodos/                      Notebook con la explicación de los métodos estudiados
└── revision-papers/              Notebook con el resumen de cada artículo revisado
requirements.txt                  Librerías generales para correr los notebooks
requirements-torch.txt            PyTorch con soporte de GPU (CUDA 13.0), opcional para deep learning
CLAUDE.md                         Contexto y convenciones del repositorio para Claude Code
```

### Trabajos

| Materia | Trabajo | Notebook |
|---|---|---|
| Machine Learning | Assignment 3 | [CIVE779MachineLearningforEngineers-Assignment3.ipynb](machine-learning/assignment-3/CIVE779MachineLearningforEngineers-Assignment3.ipynb) |
| Structural Reliability | Mini-project 2 | [mini_project_2_cable_reliability.ipynb](structural-reliability/mini-project-2/mini_project_2_cable_reliability.ipynb) |
| Structural Reliability | Open question | [open_question_cable_tension.ipynb](structural-reliability/open-question/open_question_cable_tension.ipynb) |

### Revisión de literatura

[`literature-review/metodos/metodos.ipynb`](literature-review/metodos/metodos.ipynb) reúne las explicaciones de los métodos que aparecen en los artículos de la revisión, organizados por función (p. ej. *Descomposición de señales → EMD*, *Envolvente de amplitud → Transformada de Hilbert*), con un nivel de detalle conceptual. [`literature-review/revision-papers/resumenes_papers.ipynb`](literature-review/revision-papers/resumenes_papers.ipynb) resume cada artículo de la revisión. Ver [`literature-review/README.md`](literature-review/README.md) para las convenciones.

### Cómo agregar contenido nuevo

- **Nuevo trabajo de una materia existente:** crear `<materia>/<trabajo>/` (p. ej. `machine-learning/assignment-4/`).
- **Nueva materia:** crear `<nombre-de-la-materia-en-ingles>/` con una carpeta por trabajo.
- Nombres de carpetas en minúsculas y con guiones.
- Si un trabajo necesita librerías extra, agregar un `requirements.txt` dentro de su carpeta.
- **Herramienta reutilizable entre proyectos:** va en `sensing-techniques/tools/<herramienta>/` como paquete instalable (`pip install -e`). Los proyectos la importan y guardan sus datos en su propia carpeta. Ver [`vehicle-bridge-signal-generator`](sensing-techniques/tools/vehicle-bridge-signal-generator/README.md).

## Puesta en marcha

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # o abre los .ipynb en VS Code
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Opcional, para deep learning con GPU (instala PyTorch desde el índice de CUDA 13.0; ver [`CLAUDE.md`](CLAUDE.md)):

```powershell
pip install -r requirements-torch.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.randn(3).cuda())"
```
