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

### 2.4 Modelo del comportamiento sano

- [ ] 2.4.1 no explica en el texto $\min(T,0)$ ni $h$ (horas seguidas bajo 0 °C); hoy solo están en la nota de la Tabla 4.
- [ ] **Bilineal en la selección:** con el texto de 2.4.3, el modelo Bilineal es el más simple y podría ganar con la regla de un error estándar, pero en el notebook no fue candidato. Decidir si entra en la comparación o si es solo el punto de partida.
- [ ] Confirmar en el PDF de Peeters y De Roeck (2001) las dos afirmaciones que se les atribuyen (asfalto congelado y modelo con temperaturas de horas anteriores).

### 2.5 Sistema de detección

- [ ] **Uno o dos límites:** en el notebook también se probó un límite para mediciones con helada en las 72 h anteriores y otro para las demás (18.75 h de retraso contra 40.75 h con un límite, pero más falsas alarmas). Decidir si va en el paper.
- [ ] La curva ROC no está definida en el texto.

## Otros

- [ ] Rehacer `z24_model_selection_v2.ipynb` con la nueva división de los datos antes de escribir Resultados.
- [ ] Título (todavía dice "TITTLE HERE"), resumen e introducción.
- [ ] Secciones finales (*back matter*): Funding, Acknowledgment, Disclosures y Data availability (obligatoria). Están comentadas al final del `.tex`; opciones en `GUIA_TEMPLATE.md`.
- [ ] **Accesibilidad de las figuras (regla de Optica):** no distinguir elementos solo por el color. Revisar sobre todo la Fig. 5 (folds por color) y la Fig. 1 (cajas en azul).
- [ ] Verificar las referencias Peeters2001 y Brincker2001 en el `.bib`.
