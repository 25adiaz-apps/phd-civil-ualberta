# p02-five-bridges-traffic

Igual que el set `set-variable-roughness` de [`p01-five-bridges`](../p01-five-bridges/README.md) (mismos puentes B1–B5 y G1–G5, mismos vehículos, ruido y rugosidad variable por viaje), con dos cambios:

| Cambio | p01 | p02 |
|---|---|---|
| Vibración del puente (aceleración modal RMS, log-uniforme) | 0.001–0.01 m/s² (sin tráfico) | **0.01–0.1 m/s² (con tráfico)** |
| Carretera antes y después del puente | 200 m (y recorte a 100 m) | **100 m** |

Motivo: en p01 el puente quedaba 35–57 dB por debajo de la rugosidad dentro del carro, y por debajo del ruido del teléfono en B1–B3. Con tráfico, el puente sube unos 20 dB y queda por encima del ruido en todos los puentes.

**Tabla 1.** Relación puente/fuente dentro del carro, sobre el puente (mediana de los viajes, paquete principal).

| Puente | Señal completa vs rugosidad | Cerca de f1 vs rugosidad | Cerca de f1 vs ruido |
|---|---|---|---|
| B1 | −40 dB | −27 dB | +3 dB |
| B2 | −41 dB | −24 dB | +11 dB |
| B3 | −41 dB | −26 dB | +16 dB |
| B4 | −31 dB | −19 dB | +23 dB |
| B5 | −21 dB | −9 dB | +27 dB |

"Cerca de f1": potencia en f1 ± 0.25 Hz, con ventana de Hann sobre el tramo del puente.

## Archivos

Regenerar con `python generate_p02.py` (unos 3 minutos con 14 procesos).

```
p02-five-bridges-traffic/
├── README.md
├── main.yaml, generalization.yaml   configuración (heredan de p01)
├── generate_p02.py
└── data/                            no se sube a git
    ├── main_ragged.h5               simulado (1000 viajes, largo variable)
    ├── main_padded.h5               mismo, rellenado a 11000 muestras (55 s)
    └── generalization_*.h5          igual, 250 viajes
```
