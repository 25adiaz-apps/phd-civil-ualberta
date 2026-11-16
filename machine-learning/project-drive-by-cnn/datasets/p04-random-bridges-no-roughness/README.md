# p04-random-bridges-no-roughness

Igual que [p03-random-bridges-traffic](../p03-random-bridges-traffic/README.md): los mismos 300 puentes, la misma división y la misma semilla. La única diferencia es que **no hay rugosidad** (`roughness.enabled: false`).

El teléfono mide solo dos cosas: el puente (a través del carro) y el ruido del sensor.

Sirve para probar el planteamiento cuando la rugosidad no tapa al puente.

**Archivos** (no van a git): `data/main_ragged.h5` y `data/main_padded.h5`, este último rellenado a 11 000 muestras. Se regeneran con:

```
python generate_p04.py
```
