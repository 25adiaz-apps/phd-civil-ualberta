# Project Phase 2: forma de trabajo

Proyecto de CIV E 789 Sensing Techniques and Data Analytics (Fall 2026): Statistical Damage Detection.
Entrega: jueves 19 de noviembre de 2026, 11 AM. Vale el 20 % de la nota (100 puntos).

Este archivo guarda cómo trabajamos en el proyecto. Las instrucciones nuevas que vayan saliendo se agregan aquí.

## Archivos

```
project-phase2/
├── ProjectPhase2_Statistical_Damage_Detection_Fall2026.pdf   Enunciado
├── phase2_modal_parameters.json                               Datos (intact, damage1, damage2)
├── FORMA_DE_TRABAJO.md                                        Este archivo
└── ProjectPhase2_GroupX/        Contenido del ZIP que se entrega
    ├── Main.ipynb               Notebook con todo el análisis (Partes A–F y nota sobre el uso de IA)
    ├── README.txt               Descripción breve del contenido (en inglés)
    ├── figures/part-a … part-e/ Figuras que genera el notebook, una carpeta por parte
    └── results/                 Tablas que genera el notebook (CSV)
```

Se entregan dos archivos: `ProjectPhase2_GroupX.zip` (la carpeta de arriba) y `ProjectPhase2_GroupX.pdf` (el notebook exportado, con todas las salidas visibles). La `X` se cambia por el número de grupo.

## Cómo trabajamos

- **Una sección a la vez.** Agrego las celdas de una sección, explico qué hace el código y qué muestran los resultados, y espero tu revisión antes de pasar a la siguiente.
- **Ejecución.** No ejecuto el notebook completo; te indico qué celdas cambiaron para que las corras en tu kernel.
- **Idioma.** El texto explicativo se escribe en español mientras trabajamos. El código (variables y comentarios) va en inglés. Al final se traduce al inglés respetando tu forma de escribir, y solo cuando lo pidas.
- **Numeración académica.** Tablas, ecuaciones y figuras se numeran en orden de aparición. El título de cada tabla va arriba ("**Tabla N.** ..."), el pie de cada figura va abajo ("**Figura N.** ..."), ambos fuera del objeto, y las ecuaciones llevan `\tag{N}`.
- **Fórmulas.** Se escriben en LaTeX e indican el significado de cada símbolo.
- **Sensores.** Las formas modales usan solo los 9 sensores asignados al grupo.
- **Requisitos del enunciado.** Todos los ejes van rotulados, con leyenda cuando hay varias condiciones y título en cada figura. Las distancias de Wasserstein 1D se calculan con `scipy.stats.wasserstein_distance` y las multivariadas (Partes D y E) con `scipy.stats.wasserstein_distance_nd`.
- **Semilla fija** para cualquier dato aleatorio.

## GitHub

- Cada cambio que se pida se sube a GitHub al terminarlo, con un commit descriptivo y push, manteniendo la misma estructura de carpetas que en local.
- Esos commits incluyen solo los archivos de `sensing-techniques/project-phase2/`. Los cambios pendientes de otras carpetas del repositorio no se incluyen.
