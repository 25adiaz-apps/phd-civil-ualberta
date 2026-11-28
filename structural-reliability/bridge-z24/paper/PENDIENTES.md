# Pendientes del paper

Lista de lo que falta revisar o agregar. Se revisa al final.

## Metodología

### 2.1 Datos

### 2.2 Limpieza de datos

- [ ] **Mediciones incompletas:** decir que se quitaron las mediciones a las que les falta alguna de las tres frecuencias (modo no identificado, MAC con la forma de referencia menor que 0.7). Hoy solo está en la nota de la Tabla 4. Esto explica la diferencia entre el total de la Tabla 1 (4821) y la suma de la Tabla 4 (3933 + 320 + 324 = 4577).
- [ ] **No se quitaron valores atípicos:** el monitoreo real tendrá los mismos errores de identificación, así que el modelo y el límite deben verlos.

### 2.3 División de los datos y validación

- [ ] Los conjuntos de prueba no se usaron para ninguna decisión (variables, modelos, hiperparámetros, $\lambda$, percentil ni límite).
- [ ] Por qué se eligieron las semanas de invierno del 15.12 al 28.12.1997. Razón del notebook (paso 3.1, Tabla 2): de los tres pares de semanas con helada moderada, es el que deja más mediciones bajo 0 °C en el entrenamiento (643, frente a 573 y 581) y aun así prueba el puente sano bajo 0 °C (82 mediciones).
- [ ] Por qué bloques de una semana. Frase propuesta, para ponerla después de "…bloques de una semana, como se muestra en la Fig.~\ref{fig:data-split}.":
  > Se eligieron bloques de una semana porque, con bloques de un día, los episodios de helada de más de un día quedarían repartidos entre el entrenamiento y la validación, y con bloques de un mes, casi todas las mediciones bajo $0\,^\circ$C quedarían en solo tres folds. Con bloques de una semana, cada fold tiene entre 101 y 207 mediciones bajo $0\,^\circ$C (Tabla 3).

### 2.4 Modelo del comportamiento sano

- [ ] Confirmar en el PDF de Peeters y De Roeck (2001) las dos afirmaciones que se les atribuyen (asfalto congelado y modelo con temperaturas de horas anteriores).
- [ ] **Ecuación de Stefan:** se menciona en 2.4.1 sin referencia.

## Resultados

- [ ] Redactados por Claude a partir de `z24_step_by_step.ipynb` (pasos 5 a 14): revisar la redacción y ajustarla a tu tono.
- [ ] **Extensión:** el documento usa `shortarticle` (formato carta) y Resultados tiene 9 figuras y 6 tablas. Revisar el límite de páginas de la revista y decidir qué figuras o tablas quitar o pasar a material suplementario.
- [ ] Las figuras de Resultados se generan con `make_results_figures.py` (repite los cálculos del notebook; tarda unos 2 minutos).

## Otros

- [ ] Título (todavía dice "TITTLE HERE"), resumen e introducción.
- [ ] Secciones finales (*back matter*): Funding, Acknowledgment, Disclosures y Data availability (obligatoria). Están comentadas al final del `.tex`; opciones en `GUIA_TEMPLATE.md`.
- [ ] **Accesibilidad de las figuras (regla de Optica):** no distinguir elementos solo por el color. Revisar sobre todo la Fig. 5 (folds por color) y la Fig. 1 (cajas en azul).
- [ ] Verificar las referencias Peeters2001 y Brincker2001 en el `.bib`.
