# Pendientes del paper

Lista de lo que falta revisar o agregar. Se revisa al final.

## Metodología

### 2.1 Datos

- [ ] **Contradicción, temperatura de la losa:** el texto dice que se usaron las temperaturas del pavimento y de la losa "para analizar el efecto de los cambios de temperatura", pero los modelos usan solo la del pavimento. La de la losa solo aparece en la Tabla 1 y en el criterio de mediciones válidas (nota de la Tabla 4). Decidir cómo explicarlo.

### 2.2 Limpieza de datos

- [ ] **Mediciones incompletas:** decir que se quitaron las mediciones a las que les falta alguna de las tres frecuencias (modo no identificado, MAC con la forma de referencia menor que 0.7). Hoy solo está en la nota de la Tabla 4. Esto explica la diferencia entre el total de la Tabla 1 (4821) y la suma de la Tabla 4 (3933 + 320 + 324 = 4577).
- [ ] **No se quitaron valores atípicos:** el monitoreo real tendrá los mismos errores de identificación, así que el modelo y el límite deben verlos.

### 2.3 División de los datos y validación

- [ ] Los conjuntos de prueba no se usaron para ninguna decisión (variables, modelos, hiperparámetros ni límite).
- [ ] Por qué se eligieron las semanas de invierno del 15.12 al 28.12.1997.
- [ ] Por qué bloques de una semana. Frase propuesta, para ponerla después de "…bloques de una semana, como se muestra en la Fig.~\ref{fig:data-split}.":
  > Se eligieron bloques de una semana porque, con bloques de un día, los episodios de helada de más de un día quedarían repartidos entre el entrenamiento y la validación, y con bloques de un mes, casi todas las mediciones bajo $0\,^\circ$C quedarían en solo tres folds. Con bloques de una semana, cada fold tiene entre 101 y 207 mediciones bajo $0\,^\circ$C (Tabla 3).

## Otros

- [ ] Rehacer `z24_model_selection_v2.ipynb` con la nueva división de los datos antes de escribir Resultados.
- [ ] Título (todavía dice "TITTLE HERE"), resumen e introducción.
- [ ] Borrar las secciones de ejemplo de la plantilla que quedan después de Resultados.
- [ ] Verificar las referencias Peeters2001 y Brincker2001 en el `.bib`.
