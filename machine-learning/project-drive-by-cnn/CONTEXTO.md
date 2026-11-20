# Contexto del proyecto: CNN drive-by

Registro de lo que estamos haciendo, lo que se ha pedido, lo hecho y lo pendiente. Se actualiza después de cada cambio y se lee antes de cada paso.

## 1. Objetivo

- Entrenar una CNN que reciba la aceleración medida por un teléfono dentro de un carro que recorre carretera → puente → carretera, y devuelva **solo la oscilación del puente** (`bridge_acc`, el puente bajo la rueda), descartando la rugosidad, la dinámica del carro y el ruido, con la menor información posible.
- Con esa oscilación se buscan las **propiedades dinámicas del puente: las frecuencias de todos sus modos (f1, f2, f3, …)**. **No limitar el objetivo a f1.**
- Idea: con lo que se conoce antes y después del puente, la red aprende el efecto de la dinámica del carro (relación entre lo que siente la rueda y lo que mide el teléfono) y lo aplica sobre el puente para obtener la oscilación del puente.
- Datos: 5 puentes (B1–B5) para entrenar, validar y probar; 5 puentes no vistos (G1–G5) para generalización.
- Es el proyecto final del curso de machine learning: mantenerlo simple.
- **Los datos tienen que ser lo más reales posibles** (pedido el 2026-11-16, tras ver que p01/p02 son demasiado simples).

## 2. Reglas de trabajo (pedidas por el usuario)

- El notebook se construye **una sección a la vez**; se espera revisión antes de seguir.
- Código simple, entendible para alguien con poca experiencia en Python. Texto en español, código (variables y comentarios) en inglés.
- Tablas, figuras y ecuaciones numeradas en formato académico (título de tabla arriba, pie de figura abajo, ecuaciones con `\tag`).
- **Después de cada cambio en un notebook:** ejecutarlo completo, exportar su PDF con `export_pdf.py` (sin modificar el script) y hacer commit + push **solo de esta carpeta**.
- **No afirmar como hecho lo que no se ha demostrado.** Lo que se espera de la red (p. ej. qué modos recupera) se escribe como hipótesis y se comprueba con los resultados.
- **No usar la palabra "filtro" para el efecto de la dinámica del carro** (usar "atenúa", "deja pasar"), para no confundirlo con lo que hace la red.
- Si hay que cambiar el dataset, avisar para regenerarlo. **No modificar el generador ni p01 sin aprobación.**
- Respuestas en español, cortas y directas.
- **Mantener el código simple** en todas las secciones (lo ha pedido varias veces).

## 3. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Dataset | `p02-five-bridges-traffic`: puente excitado por tráfico (0.01–0.1 m/s² RMS modal), rugosidad nueva en cada viaje, 100 m antes y después. Se regenera con `python generate_p02.py` (los `.h5` no van a git). |
| Salida (target) | `bridge_acc` **con todas sus frecuencias**, sin filtrar, multiplicado por la máscara del puente. Primera corrida así, para demostrar con resultados qué modos se recuperan. |
| Entrada | **Versión A:** `measured` + `bridge_mask` (1 sobre el puente). La red no recibe el perfil de la carretera. |
| División | Por puente, semilla 42: 140 / 30 / 30 viajes (entrenamiento / validación / prueba). |
| Escala | Cada señal dividida por su desviación estándar, calculada solo con los viajes de entrenamiento (`scale_measured` = 0.764, `scale_target` = 0.090 m/s²). La máscara no se escala. |

## 4. Hecho

- **Dataset p02** creado, generado y documentado (`datasets/p02-five-bridges-traffic/`).
- **`phone_signal_simulation.ipynb`**: cómo el simulador construye la señal del teléfono (puente, rugosidad, carro, descomposición por fuentes, teléfono). Con PDF.
- **`drive_by_cnn.ipynb`**, secciones 1–5 (con PDF):
  1. Librerías y configuración.
  2. Cargar los datos (Tabla 1, Figura 1 esquema).
  3. Mirar un viaje (Figura 2 señales; Figura 3 puente bajo la rueda vs parte del puente en el teléfono).
  4. Entrada y salida, versión A (Tabla 2; 4.1 `measured` + `bridge_mask`; 4.2 target; 4.3 Figura 4).
  5. División y escala (Tabla 3 con viajes por grupo y frecuencias de todos los modos; Ec. 1; Figura 5).
  6. Construir la red: U-Net 1D, `CHANNELS = [16, 32, 64, 64, 64, 64, 64, 64]` (8 niveles, 7 reducciones a la mitad), bloques de 2 convoluciones de 9 muestras + ReLU, relleno 11 000 → 11 008. 1 041 009 parámetros. Ve ±15.3 s alrededor de cada muestra (comprobado con gradientes). Ec. 2 (convolución), Figura 6 (esquema).
  7. Entrenar (primera corrida, versión A, target `bridge_acc` completo): error cuadrático medio solo sobre el puente (Ec. 3); Adam, lr 3e-4, lote 8, 100 épocas (~4 min); se guardan los pesos con menor error de validación (Tabla 4, Figura 7).
     - **Resultado:** el error de validación queda en ~1.04 desde la primera época, solo ~5 % por debajo de una red que devuelve 0 (1.10), y no mejora. Desde la época ~40 la red memoriza el entrenamiento (entrenamiento baja a ~0.77, validación sube a ~1.15). Pruebas fuera del notebook: BatchNorm y otra tasa de aprendizaje dan lo mismo; con target "parte del puente en el teléfono" la validación baja a ~0.60.
     - Posible explicación (por comprobar en la sección 8): la rugosidad desconocida sobre el puente domina al puente en el teléfono (−9 a −27 dB cerca de f1); con un error muestra por muestra, lo óptimo es devolver casi 0.
  8. Evaluar en prueba (Tablas 5 y 6, Ec. 4, Figuras 8 y 9).
     - **Resultado:** el error de la red es igual al de una red que devuelve 0 en los 5 puentes; la salida mide 3–6 % del target. El pico más alto de la salida está en 1.7–2.0 Hz en todos los puentes (rango del cuerpo del carro, no del puente). Modo 1: como mucho 4 % de la amplitud; modos altos: ≤ 2 %. **La red no obtiene ningún modo del puente en la primera corrida.**
- **Análisis hechos** (en la conversación, no en el notebook):
  - Relación puente/rugosidad dentro del carro cerca de f1: −9 (B5) a −27 dB (B1); puente/ruido: +3 (B1) a +27 dB (B5).
  - Transmisibilidad del carro T(f) (rueda → teléfono), verificada contra el simulador: no hay un corte fijo; cae a 0.1 entre 5.8 y 19 Hz según el carro (mediana 12.6 Hz). T a 20 Hz ≈ 0.02, a 27 Hz ≈ 0.006 (medianas).
  - Promediar espectros de `measured` de muchos viajes no hace aparecer los modos (la rugosidad no baja al promediar); sí ayuda promediar las salidas de la red.
  - `bridge_acc` (punto fijo) difiere de la aceleración que siente la rueda (d²u/dt²): correlación 0.87–0.91. Se decidió mantener `bridge_acc`.

## 5. Pendiente

### Dataset realista p03 (pedido el 2026-11-16; generado)

**Simplificaciones pedidas por el usuario (se mantienen; van al final como limitaciones o investigación futura):** quarter car; sin motor, rueda ni soporte del teléfono; solo ruido blanco del sensor y filtro anti-alias. No volver a proponer half car ni perturbaciones.

Por qué p01/p02 no sirven:
- f1 está escrita a mano en cada puente y sigue casi exactamente la tabla f1(L) del generador; los G son "vecinos" de los B con L y f1 cambiados juntos. Con eso basta una regresión del largo, no una CNN.
- Solo hay 5 puentes, así que la red puede reconocerlos y memorizarlos.
- Es una decisión del dataset, no del generador: por defecto, el generador ya varía f1 entre 0.7 y 1.3 veces el valor de la tabla.

p03 (`datasets/p03-random-bridges-traffic/`, solo yaml nuevo, generador sin cambios):
- Todo igual a p02, salvo los puentes: 300 puentes al azar × 5 viajes = 1500 viajes.
- Cada puente sortea L (15–60 m), f1 = f1_tabla(L) × factor (0.7–1.3), masa (4000–20 000 kg/m) y amortiguamiento de cada modo (0.5–3 %).
- Clase de rugosidad A/B/C sorteada por viaje.
- `partitions` [0.70, 0.15, 0.15]: 210 / 45 / 45 puentes (1050 / 225 / 225 viajes), y ningún puente ni carro aparece en dos grupos. Los puentes de prueba son nuevos, así que reemplazan a G1–G5.
- **Prueba de salida de frecuencias en p03** (fuera del notebook): misma red y error que en p02, 100 épocas, evaluada en los 45 puentes de prueba.
  - Error relativo mediano en f1: red 16.5 %; red sin el puente en la entrada 16.5 % (idéntico); referencia que solo usa el largo exacto y la tabla 17.7 %; promedio 38 %. Los modos 2–5 dan lo mismo.
  - **La red sigue sin usar la vibración del puente.** Solo aprende el atajo del largo (vía el ancho de la máscara), cuyo error es la dispersión del factor 0.7–1.3.
  - Causa probable: en p03 el puente está muy por debajo de la rugosidad en el teléfono (mediana −35 dB, rango 5–95 % de −53 a −18 dB, nunca por encima) e incluso por debajo del ruido del sensor (mediana −4.7 dB). Son relaciones de la señal completa sobre el puente, de los metadatos.
- **Comprobación sin red en p03** (fuera del notebook; pedida por el usuario).
  - Para cada uno de los 300 puentes se combinan sus 5 viajes y se busca el pico más alto entre 1 y 12 Hz. Cuenta como acierto si el pico cae a ±10 % de f1.
  - (A) Espectro medio sobre el puente: 14.0 % de aciertos; sin el puente en la señal, 12.3 %; solo con la parte del puente, 100 % (el método funciona si el puente se ve).
  - (B) Cociente entre el espectro sobre el puente y el de la carretera, con la misma ventana: 7.7 % con 5 viajes y 8.0 % con 1 viaje. El azar da 6.1 %.
  - Cerca de f1 (±10 %), el puente está a una mediana de −21.5 dB de lo demás (rango 5–95 % de −42 a −2 dB). Solo en el 3.8 % de los viajes queda por encima.
  - **Conclusión:** con estas amplitudes, ni 1 ni 5 viajes traen información visible de f1. Ninguna red la puede sacar de ahí.
- **¿Es realista la vibración del puente del generador (0.01–0.1 m/s² RMS por modo, en el vientre del modo)?** Se comparó con mediciones reales del repositorio (fuera del notebook). Se calculó el RMS en una banda alrededor de cada modo, tomando el sensor que más se mueve.
  - **OPARA (openBridgeLAB, Bautzen, m/s²):** de 1e-5 a 1e-4 m/s² por modo (de día, mediana); el máximo de todas las mediciones es 6e-4. Por recordar, sin confirmar: es un puente de investigación sin tráfico, así que representa el piso sin tráfico.
  - **Z24 (EMS, con tráfico encima):** las unidades no están documentadas (m/s² o g). En ventanas de 5 s (como un cruce), modo 1 (3.98 Hz): mediana 2.2e-4, p90 1.1e-3, p99 3.8e-3 y p99.9 8.0e-3, en unidades del archivo.
    - Si son m/s², hasta el 99.9 % de las ventanas está por debajo de 0.01, el mínimo del generador.
    - Si son g, la mediana es 0.002 m/s² y el p99 es 0.037 m/s².
    - Los sensores verticales del EMS no están en el centro del vano de 30 m, así que el valor en el vientre puede ser algo mayor.
  - **Conclusión:** el rango del generador no es pesimista; está en el extremo alto de lo medido o por encima. Las señales de p03 son realistas, o incluso favorables al puente. El hallazgo se mantiene: en un viaje con un carro y un teléfono, el puente casi no se ve.

### Prueba sin rugosidad: p04 (pedida por el usuario el 2026-11-16; hecha fuera del notebook)

- **p04** (`datasets/p04-random-bridges-no-roughness/`) es p03 con `roughness.enabled: false`: mismos 300 puentes, misma división y semilla. Lo comprobé: las f1 y las particiones son idénticas, y la rugosidad vale 0.
- **Modelo:** el mismo del notebook (U-Net, `measured` + máscara → `bridge_acc` × máscara, Ec. 3, Adam 3e-4, lote 8, 100 épocas, mejores pesos de validación). Evaluado en los 45 puentes de prueba (225 viajes).
- **Entrenamiento:** la mejor validación es 0.921 (época 24), frente a 1.090 de una salida 0. Desde ahí memoriza.
- **Prueba en el tiempo:** error 0.754 frente a 0.841 de una salida 0; correlación mediana salida–target 0.26.
- **Pico en f_n** (pico más alto en [0.75, 1.25]·f_n que cae a ±10 % de f_n), con salida / target / `measured`:

  | Modo | Salida | Target | `measured` |
  |---|---|---|---|
  | 1 | 86.7 % | 88.9 % | 87.6 % |
  | 2 | 59.6 % | 90.2 % | 62.7 % |
  | 3 | 17.4 % | 88.2 % | 44.6 % |
  | 4 | 13.3 % | 85.7 % | 37.1 % |
  | 5 | 16.0 % | 88.0 % | 44.0 % |

  Amplitud de la salida respecto del target: 0.74, 0.34, 0.06, 0.02, 0.02.
- **Lectura:**
  - Sin rugosidad, f1 (y en parte f2) ya se ve directamente en `measured`, sin red.
  - La red no mejora sobre el espectro directo de `measured` en f1 y f2, y en los modos 3–5 pierde información frente a él (apaga los modos altos).
  - El error muestra por muestra sigue siendo malo.
- Pendiente de decidir con el usuario cómo seguir.

### Filtro inverso y varios viajes (preguntas del usuario, 2026-11-19/20; sin código nuevo)

- **Artículos:** Shirzad-Ghaleroudkhani y Gül 2020 (Sensors, laboratorio) y 2021 (Smart Cities, dos puentes reales de Edmonton), PDFs en la raíz del proyecto. Filtro F = Ȳ/X̄ (espectro fuera del puente, ADFT con ventanas Hamming de 3 s), aplicado al espectro sobre el puente. Suponen igual velocidad y rugosidad fuera y sobre el puente (en 2021, emparejadas por rango de velocidad y SRC ±5 %). Solo f1; resultados en ejemplos, sin tasa de acierto ni control.
- **Implementable sin cambiar el generador** (cada viaje tiene carretera antes y después con el mismo carro, velocidad y clase de rugosidad). La prueba sin red (B) de p03 ya es casi ese método: 8.0 % (1 viaje) y 7.7 % (5 viajes, promedio de log-cocientes) frente a 6.1 % de azar. Falta solo la receta exacta (ADFT + escala Ȳ); ofrecido, no hecho.
- **Varios viajes:** con 5 viajes no hubo mejora sobre 1. Estimación (no probada): el puente pesa ~0.7 % de la potencia cerca de f1 (−21.5 dB), así que harían falta del orden de miles de viajes por puente. Probarlo requiere generar más viajes por puente (no cambia el generador). Ofrecido, pendiente de decisión.
- **Pregunta del usuario:** si ambas oscilaciones están en el espectro, ¿por qué importa que la rugosidad pese más? Respuesta: el espectro de la rugosidad en un tramo corto no es una curva lisa, varía al azar ~100 % entre frecuencias vecinas; el puente (0.7 %) queda dentro de esa variación, así que ninguna red ni método puede distinguirlo sin información de la rugosidad del propio tramo del puente. Evidencia: control sin puente (12.3 % vs 14.0 %) y p04 (sin rugosidad, f1 visible en `measured` en 87.6 %).
- Figura pedida por el usuario (fuera del proyecto, script `plot_problem.py` en el scratchpad): viaje 728 de p03 (puente 173, f1 = 2.23 Hz, L = 44.9 m, v = 9.5 m/s, puente a −21.5 dB, la mediana). (a) espectro sobre el puente: medido = medido sin puente; la parte del puente queda ~2 órdenes por debajo. (b) mismo viaje en p04: el pico de f1 domina. (c) los 5 viajes del puente: picos en lugares distintos, máximo cerca de 1–2 Hz (frecuencia del carro 1.31 Hz), ninguno en f1.

Límite del generador: solo modela vigas simplemente apoyadas (f_n = n²·f1). Se anota como limitación.

### Perfil de la carretera como entrada (decidido: versión A)

Se continúa con la **versión A** (`measured` + `bridge_mask`, sin perfil de la carretera). Opciones que se discutieron y quedaron fuera por ahora:

1. Perfil suavizado (ondas > 5 m): rechazado por el usuario.
2. Curvas verticales reales de diseño: realistas, pero su contenido está por debajo de ~0.2 Hz y no excitan al carro en las frecuencias del puente.
3. Ondulaciones artificiales (1–10 Hz, unos mm): no son un diseño vial real; requieren cambiar el generador y crear p03.
4. Rugosidad de ondas largas medida con perfilómetro, con error.
5. Versión B (`measured` + perfil completo antes y después + máscara): posible comparación posterior, solo si el usuario lo pide.

### Secciones del notebook por hacer

9. Comparar con un método sin red.
10. Probar en puentes no vistos (G1–G5).
11. Conclusiones.

### Prueba: la red devuelve directamente las frecuencias (pedida por el usuario; hecha fuera del notebook)

- Mismos datos (p02) y entradas (versión A). Red: la bajada de la U-Net (8 niveles) + promedio en el tiempo + capa lineal → 5 frecuencias. Error: error relativo al cuadrado, solo en los modos que existen. 100 épocas.
- **Resultado:** frecuencias muy cercanas en prueba (B1–B5) y también en los no vistos (G2–G4 casi exactas; G1 7.6 vs 7; G5 f1 1.1–1.2 vs 1.4).
- **Control:** quitando la parte del puente de `measured` (con las componentes del archivo), las predicciones son **idénticas**. La red no usa la vibración del puente: deduce las frecuencias de otras pistas. En este dataset f1 depende casi solo del largo del puente (tabla nominal f1(L) del generador, mismo factor para B y G), y el largo se puede deducir del ancho de la máscara (L / v).
- Para que esta prueba tenga sentido, el dataset debe romper la relación largo–frecuencia: p. ej. muchos puentes con `f1_factor` aleatorio (opción que ya tiene el generador, sin modificarlo). Pendiente de decidir con el usuario si se agrega al notebook y cómo seguir.

### Decisión abierta tras la primera corrida

Cómo seguir, dado que en la primera corrida la red no obtiene ningún modo (pendiente de decidir con el usuario). Opciones propuestas:

1. Prueba sin rugosidad (ya estaba pendiente): dataset nuevo con `roughness.enabled: false` (solo un yaml, sin cambiar el generador). Dice si el planteamiento funciona cuando la rugosidad no tapa al puente.
2. Error que compare espectros sobre el puente (no muestra por muestra), acorde al objetivo de obtener frecuencias.
3. Versión B (perfil completo antes y después).

### Pruebas posteriores

- Si en la primera corrida el error en las frecuencias altas es grande: probar la red con datos sin rugosidad, para ver si obtiene todos los modos.

### Riesgos a vigilar

- Con solo 5 puentes, la red puede memorizarlos (pistas: ancho de la máscara = largo del puente / velocidad; clase de rugosidad distinta por puente). La prueba decisiva es G1–G5.
- G5 (f1 = 1.4 Hz) está fuera del rango de f1 de entrenamiento (1.6–8 Hz).
