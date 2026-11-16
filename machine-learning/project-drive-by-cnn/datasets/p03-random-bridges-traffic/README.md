# p03-random-bridges-traffic

Igual que [p02-five-bridges-traffic](../p02-five-bridges-traffic/README.md), salvo los puentes.

Se mantiene todo lo de p02:
- quarter car;
- sin motor, rueda ni soporte del teléfono;
- ruido blanco del sensor y filtro anti-alias;
- puente excitado por tráfico (0.01–0.1 m/s² RMS modal);
- rugosidad nueva en cada viaje;
- 100 m de carretera antes y después del puente.

En vez de 5 puentes fijos hay **300 puentes al azar**, con 5 viajes cada uno (1500 viajes). Cada puente sortea por separado:

| Propiedad | Rango |
|---|---|
| Largo L | 15–60 m |
| f1 | f1 de la tabla según L × factor 0.7–1.3 (1.3–10.4 Hz) |
| Masa | 4 000–20 000 kg/m |
| Amortiguamiento de cada modo | 0.5–3 % |

La clase de rugosidad (A, B o C) se sortea en cada viaje.

**División:** 210 / 45 / 45 puentes para entrenamiento, validación y prueba (1050 / 225 / 225 viajes). Viene en la columna `metadata/partition`. Ningún puente ni carro aparece en dos grupos, así que los puentes de prueba son nuevos para la red.

**Límite:** el generador solo modela vigas simplemente apoyadas, donde f_n = n²·f1.

**Archivos** (no van a git): `data/main_ragged.h5` y `data/main_padded.h5`, este último rellenado a 11 000 muestras. Se regeneran con:

```
python generate_p03.py
```
