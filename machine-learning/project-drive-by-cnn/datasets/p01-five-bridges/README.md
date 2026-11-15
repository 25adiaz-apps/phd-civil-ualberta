# p01-five-bridges

Primer paquete de datos para entrenar una CNN de monitoreo indirecto (drive-by).

- **Entrada de la CNN:** señal medida por el teléfono dentro del carro durante todo el viaje, y perfil de la carretera antes y después del puente.
- **Salida de la CNN:** perfil de la carretera y del puente bajo la rueda en cada instante del viaje.

Generado con [`vehicle-bridge-signal-generator`](../../../../sensing-techniques/tools/vehicle-bridge-signal-generator/). Todo se define en los YAML de configuración; la herramienta no tiene nada fijado para este paquete.

## 1. Fuentes de vibración

| Se incluye | No se incluye |
|---|---|
| Vibración propia del puente (vibración ambiental previa) | Motor |
| Rugosidad del pavimento (ISO 8608) | Soporte del teléfono |
| Dinámica del vehículo (cuarto de carro) | Excentricidad de rueda |
| Ruido blanco del teléfono | Ruido rosa, sesgo, deriva, *jitter*, cuantización |

- Interacción **desacoplada** y el carro **no carga** al puente: el puente vibra solo por su vibración previa; el carro la recorre y la mide.
- Amplitudes **reales** (físicas), sin amplificar el puente.
- Se mantiene el filtro anti-alias del teléfono (no es fuente de vibración).
- Frecuencia de muestreo de salida: **200 Hz**.

## 2. Puentes

Viga simplemente apoyada. Frecuencias y formas modales fijas por puente. Cada puente tiene su clase de rugosidad ISO 8608.

**Paquete principal** (200 viajes por puente, 1000 en total):

| Puente | L [m] | f1 [Hz] | masa [kg/m] | ζ | Clase |
|---|---|---|---|---|---|
| B1 | 15 | 8.0 | 5 000 | 2.0 % | A |
| B2 | 25 | 4.5 | 8 000 | 1.5 % | B |
| B3 | 35 | 3.0 | 12 000 | 1.2 % | C |
| B4 | 45 | 2.3 | 15 000 | 1.0 % | B |
| B5 | 60 | 1.6 | 20 000 | 0.8 % | A |

**Paquete de generalización** (puentes no vistos; 50 viajes por puente, 250 en total): un puente "vecino" de cada uno, con L, f1 y masa cambiados entre 10 y 20 %, la misma clase y el mismo ζ.

| Puente | L [m] | f1 [Hz] | masa [kg/m] | ζ | Clase |
|---|---|---|---|---|---|
| G1 | 17.0 | 7.0 | 5 800 | 2.0 % | A |
| G2 | 21.5 | 5.2 | 7 000 | 1.5 % | B |
| G3 | 40.0 | 2.6 | 13 800 | 1.2 % | C |
| G4 | 39.0 | 2.7 | 13 000 | 1.0 % | B |
| G5 | 68.0 | 1.4 | 17 000 | 0.8 % | A |

## 3. Lo que cambia en cada viaje

| Parámetro | Valor |
|---|---|
| Puente | uno de los 5 (200 viajes cada uno) |
| Velocidad | uniforme 5–25 m/s |
| Vehículo (cuarto de carro) | nuevo en cada viaje: carrocería 250–500 kg, rueda 30–100 kg, rebote 1–2 Hz, salto de rueda 8–15 Hz, ζ suspensión 0.2–0.4 |
| Vibración previa del puente | ruido blanco modal, aceleración modal RMS log-uniforme 0.001–0.01 m/s² (amplitud y fase distintas en cada viaje) |
| Ruido del teléfono | densidad uniforme 100–400 µg/√Hz |
| Rugosidad | según el set (sección 4) |

- Viajes con despegue de rueda: se descartan y se vuelven a sortear. Ocurre casi solo en carretera clase B y C a alta velocidad; en B3 (clase C) se volvieron a sortear 106 de 200 viajes (set fijo) y 70 de 200 (set variable), así que su velocidad media baja a 12.6 y 14.0 m/s (15 m/s en los demás).
- Sin división entrenamiento/validación/prueba (se hace al entrenar).

## 4. Sets

| Set | Rugosidad |
|---|---|
| `set-fixed-roughness` | un perfil fijo por puente: todos los viajes sobre un puente pisan la misma carretera |
| `set-variable-roughness` | perfil nuevo en cada viaje (el carro pasa por otra línea del carril), con la clase del puente |

Cada set tiene:

- tramos de **100 m** y de **200 m** antes y después del puente. Son **los mismos viajes**: el de 100 m es un recorte del de 200 m;
- dos versiones de largo: **padded** (largo fijo, rellenado con ceros, con máscara de validez) y **ragged** (largo variable);
- paquete **principal** (1000 viajes) y de **generalización** (250 viajes).

## 5. Contenido de cada viaje

| Dato | Descripción |
|---|---|
| `measured` | aceleración vertical medida por el teléfono [m/s²] (entrada) |
| `profile/position` | posición de la rueda en cada medición [m] (0 = inicio del puente) |
| `profile/road_displacement` | altura de la rugosidad bajo la rueda [m] |
| `profile/road_acceleration` | aceleración que la rugosidad impone a la rueda, v²·r''(x) [m/s²] |
| `profile/bridge_displacement` | desplazamiento del puente bajo la rueda (solo su oscilación; 0 fuera del puente) [m] |
| `profile/bridge_acceleration` | aceleración del punto del puente bajo la rueda [m/s²] |
| `segment` | 0 antes del puente, 1 sobre el puente, 2 después |
| `components/*` | descomposición exacta de `measured` por fuente (puente por modo, rugosidad, ruido) |
| `metadata/*` | parámetros de cada viaje (puente, vehículo, velocidad, ruido, clase, ...) |

## 6. Estructura de la carpeta

Generado con la herramienta 1.1.0 (commit `c65869b`); regenerar con `python generate_p01.py` (unos 8 minutos con 14 procesos).

```
p01-five-bridges/
├── README.md                 este plan
├── p01_base.yaml             configuración común
├── generate_p01.py           regenera todos los archivos
├── set-fixed-roughness/
│   ├── main.yaml, generalization.yaml
│   ├── set-fixed-roughness.ipynb    cómo se generó el set, con gráficas
│   └── data/                        no se sube a git (~860 MB por set)
│       ├── main_200m_ragged.h5      simulado (1000 viajes)
│       ├── main_200m_padded.h5      mismo, rellenado a 19000 muestras (95 s)
│       ├── main_100m_ragged.h5      recorte a 100 m antes y después
│       ├── main_100m_padded.h5      recorte, rellenado a 11000 muestras (55 s)
│       └── generalization_*.h5      igual, 250 viajes
└── set-variable-roughness/
    └── (igual)
```
