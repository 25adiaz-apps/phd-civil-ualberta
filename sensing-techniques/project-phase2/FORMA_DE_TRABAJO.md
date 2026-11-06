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
└── ProjectPhase2_Group6/        Contenido del ZIP que se entrega
    ├── Main.ipynb               Notebook con todo el análisis (Partes A–F y nota sobre el uso de IA)
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

- **Una sección a la vez.** Agrego las celdas de una sección, explico qué hace el código y qué muestran los resultados, y espero tu revisión antes de pasar a la siguiente.
- **Ejecución.** Antes de cada commit ejecuto `Main.ipynb` completo (`jupyter nbconvert --execute --inplace`), para que en GitHub se vean las tablas y figuras. Si algún cálculo llega a tardar varios minutos, lo aviso antes de ejecutarlo.
- **Idioma.** Todo lo que va en `Main.ipynb` se escribe en inglés desde el principio: texto, títulos, tablas, pies de figura, rótulos de ejes y código. Las explicaciones en el chat siguen en español.
- **Numeración académica.** Tablas, ecuaciones y figuras se numeran en orden de aparición. El título de cada tabla va arriba ("**Table N.** ..."), el pie de cada figura va abajo ("**Figure N.** ..."), ambos fuera del objeto, y las ecuaciones llevan `\tag{N}`.
- **Fórmulas.** Se escriben en LaTeX e indican el significado de cada símbolo.
- **Sensores.** Las formas modales usan solo los 9 sensores del grupo 6 (tabla de arriba).
- **Requisitos del enunciado.** Todos los ejes van rotulados, con leyenda cuando hay varias condiciones y título en cada figura. Las distancias de Wasserstein 1D se calculan con `scipy.stats.wasserstein_distance` y las multivariadas (Partes D y E) con `scipy.stats.wasserstein_distance_nd`.
- **Semilla fija** para cualquier dato aleatorio.

## GitHub

- Cada cambio que se pida se sube a GitHub al terminarlo, con un commit descriptivo y push, manteniendo la misma estructura de carpetas que en local.
- Esos commits incluyen solo los archivos de `sensing-techniques/project-phase2/`. Los cambios pendientes de otras carpetas del repositorio no se incluyen.
