# Contexto del proyecto: CNN drive-by

Registro de lo que estamos haciendo, lo que se ha pedido, lo hecho y lo pendiente. Se actualiza después de cada cambio y se lee antes de cada paso.

## 1. Objetivo

- Entrenar una CNN que reciba la aceleración medida por un teléfono dentro de un carro que recorre carretera → puente → carretera, y devuelva **solo la oscilación del puente** (`bridge_acc`, el puente bajo la rueda), descartando la rugosidad, la dinámica del carro y el ruido, con la menor información posible.
- Con esa oscilación se buscan las **propiedades dinámicas del puente: las frecuencias de todos sus modos (f1, f2, f3, …)**. **No limitar el objetivo a f1.**
- Idea: con lo que se conoce antes y después del puente, la red aprende el efecto de la dinámica del carro (relación entre lo que siente la rueda y lo que mide el teléfono) y lo aplica sobre el puente para obtener la oscilación del puente.
- Datos: 5 puentes (B1–B5) para entrenar, validar y probar; 5 puentes no vistos (G1–G5) para generalización.
- Es el proyecto final del curso de machine learning: mantenerlo simple.

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
- **Análisis hechos** (en la conversación, no en el notebook):
  - Relación puente/rugosidad dentro del carro cerca de f1: −9 (B5) a −27 dB (B1); puente/ruido: +3 (B1) a +27 dB (B5).
  - Transmisibilidad del carro T(f) (rueda → teléfono), verificada contra el simulador: no hay un corte fijo; cae a 0.1 entre 5.8 y 19 Hz según el carro (mediana 12.6 Hz). T a 20 Hz ≈ 0.02, a 27 Hz ≈ 0.006 (medianas).
  - Promediar espectros de `measured` de muchos viajes no hace aparecer los modos (la rugosidad no baja al promediar); sí ayuda promediar las salidas de la red.
  - `bridge_acc` (punto fijo) difiere de la aceleración que siente la rueda (d²u/dt²): correlación 0.87–0.91. Se decidió mantener `bridge_acc`.

## 5. Pendiente

### Perfil de la carretera como entrada (decidido: versión A)

Se continúa con la **versión A** (`measured` + `bridge_mask`, sin perfil de la carretera). Opciones que se discutieron y quedaron fuera por ahora:

1. Perfil suavizado (ondas > 5 m): rechazado por el usuario.
2. Curvas verticales reales de diseño: realistas, pero su contenido está por debajo de ~0.2 Hz y no excitan al carro en las frecuencias del puente.
3. Ondulaciones artificiales (1–10 Hz, unos mm): no son un diseño vial real; requieren cambiar el generador y crear p03.
4. Rugosidad de ondas largas medida con perfilómetro, con error.
5. Versión B (`measured` + perfil completo antes y después + máscara): posible comparación posterior, solo si el usuario lo pide.

### Secciones del notebook por hacer

8. Evaluar en prueba: comparar el espectro de la salida con el de `bridge_acc` **en cada modo de la Tabla 3**, no solo f1; error por modo; también promediando los viajes de cada puente.
9. Comparar con un método sin red.
10. Probar en puentes no vistos (G1–G5).
11. Conclusiones.

### Decisión abierta tras la primera corrida

Cómo seguir, dado que la red no aprende `bridge_acc` muestra por muestra (pendiente de decidir con el usuario, después de ver la sección 8).

### Pruebas posteriores

- Si en la primera corrida el error en las frecuencias altas es grande: probar la red con datos sin rugosidad, para ver si obtiene todos los modos.

### Riesgos a vigilar

- Con solo 5 puentes, la red puede memorizarlos (pistas: ancho de la máscara = largo del puente / velocidad; clase de rugosidad distinta por puente). La prueba decisiva es G1–G5.
- G5 (f1 = 1.4 Hz) está fuera del rango de f1 de entrenamiento (1.6–8 Hz).
