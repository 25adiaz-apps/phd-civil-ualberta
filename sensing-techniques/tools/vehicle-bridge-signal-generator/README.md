# vehicle-bridge-signal-generator

Generador de señales vehículo-puente (*drive-by*) por simulación física. Simula la aceleración vertical que registra un smartphone dentro de un carro que cruza un puente y entrega, además de la señal medida, **cada componente por separado** (cada modo del puente, rugosidad, motor, rueda y ruido) como verdad de referencia. La separación es exacta: `measured = suma de las componentes`.

Es una herramienta reutilizable: aquí solo se generan y documentan las señales. El procesamiento (separación con deep learning, EMD, identificación de frecuencias...) se hace en las carpetas de cada proyecto, que importan este paquete.

La metodología completa, con ecuaciones, verificaciones y gráficas, está en [`notebooks/metodologia_generacion_senales.ipynb`](notebooks/metodologia_generacion_senales.ipynb).

## Contenido

- [Instalación](#instalación)
- [Uso desde Python](#uso-desde-python)
- [Uso desde la línea de comandos](#uso-desde-la-línea-de-comandos)
- [Presets](#presets)
- [Configuración](#configuración)
- [Modelo físico en breve](#modelo-físico-en-breve)
- [Componentes (etiquetas)](#componentes-etiquetas)
- [Formato del dataset](#formato-del-dataset)
- [Reproducibilidad](#reproducibilidad)
- [Verificación](#verificación)
- [Supuestos](#supuestos)
- [Decisiones de diseño y limitaciones](#decisiones-de-diseño-y-limitaciones)
- [Estructura](#estructura)
- [Referencias](#referencias)

## Instalación

Desde la raíz del repositorio, con el entorno virtual activo (`.venv\Scripts\activate`):

```powershell
pip install -e sensing-techniques/tools/vehicle-bridge-signal-generator
# opcional: pruebas, cuaderno y prueba en vivo contra OpenSees
pip install -e "sensing-techniques/tools/vehicle-bridge-signal-generator[dev]"
```

Dependencias: numpy, scipy, h5py, pyyaml y matplotlib (más pytest, openseespy, jupyter y pandas en `[dev]`). Se recomienda la instalación editable (`-e`): los presets se leen de la carpeta `presets/` de la herramienta.

## Uso desde Python

**Escenario fijo**: todos los parámetros fijos, señal determinista para una semilla.

```python
import vehicle_bridge_signal as vbs

res = vbs.simulate("typical_car_smartphone", seed=0)    # preset, ruta a un YAML o dict
res.measured                     # (2048,) m/s², señal del smartphone a 200 Hz
res.components["bridge_mode_1"]  # cada componente, misma forma
res.on_bridge                    # máscara booleana: algún eje sobre el puente
res.contact["roughness"]         # (n_ejes, 2048) aceleración del punto de contacto
res.metadata                     # frecuencias del puente y del vehículo, mu, relaciones señal-ruido...
res.identity_error()             # |measured - suma(componentes)| / RMS  (exacto)
```

**Estudios paramétricos**: cambiar un parámetro y mantener los demás, incluidas las realizaciones aleatorias. Cada fuente tiene su propio generador aleatorio, así que el perfil de rugosidad es el mismo a cualquier velocidad.

```python
base = vbs.load_preset("typical_car_smartphone")
for v in (5, 10, 15, 20, 25):
    res = vbs.simulate(vbs.with_overrides(base, {"crossing.speed": v}), seed=0)
```

**Muestreo aleatorio y datasets**:

```python
scenario, seed = vbs.sample_scenario("default_dataset", seed=3)   # un escenario fijo sorteado
res = vbs.simulate(scenario, seed)

path = vbs.generate_dataset("default_dataset", n_samples=10_000,
                            out_path="machine-learning/<proyecto>/data/vehicle_bridge_v1.h5",
                            seed=0, workers=8)
with vbs.load_dataset(path) as ds:                 # lectura perezosa (h5py)
    x = ds.measured[0:64]                          # float32 (64, 2048)
    y = ds.stack_components(range(64))             # (64, n_componentes sin 'noise', 2048)
    meta = ds.metadata                             # dict de columnas; ds.metadata_frame() -> pandas
    split = vbs.group_split(ds)                    # train/val/test sin compartir puentes ni vehículos
```

`regenerate_sample(path, i, return_internal=True)` vuelve a simular la muestra `i` en float64, con todas las señales internas a 1000 Hz.

## Uso desde la línea de comandos

`vbsig` es equivalente a `python -m vehicle_bridge_signal`, y `python -m vehicle_bridge_signal.generate` equivale a `vbsig generate`.

```powershell
# dataset (la ruta la decide el proyecto; la herramienta no guarda datos dentro de tools/)
python -m vehicle_bridge_signal.generate --config default_dataset --n-samples 10000 --out <proyecto>/data/vehicle_bridge_v1.h5 --workers 8
# vista previa de N muestras sin generar el dataset
python -m vehicle_bridge_signal.generate --config default_dataset --preview 4 --plot preview.png
# un escenario fijo: resumen, gráficas de componentes y espectros, y arreglos en .npz
vbsig simulate --config typical_car_smartphone --seed 0 --set crossing.speed=10 --plot cruce.png --save cruce.npz
vbsig simulate --config default_dataset --sample --seed 3      # sortea un escenario y lo simula
vbsig presets                                                  # lista de presets
vbsig check --config mi_config.yaml                            # valida y muestra los parámetros aleatorios
vbsig info <proyecto>/data/vehicle_bridge_v1.h5                # resumen de un dataset
```

`--set clave.anidada=valor` cambia cualquier parámetro (el valor se lee como YAML) y se puede repetir. `--config` acepta el nombre de un preset o la ruta a un YAML.

## Presets

| Preset | Tipo | Para qué sirve |
|---|---|---|
| `typical_car_smartphone` | fijo | Caso de referencia: carro de 450 kg, puente de 30 m (f1 = 3.63 Hz), 15 m/s, clase B, motor de 4 cilindros a 1500 rpm, sensor de 150 µg/√Hz. Base de los demás presets fijos. |
| `bridge_only_smooth` | fijo | Pavimento liso, sin motor ni ruido: la medición es solo el camino del puente. |
| `rough_road` | fijo | Pavimento clase D: la rugosidad enmascara al puente. |
| `high_speed` | fijo | 25 m/s: cruce corto y más excitación por rugosidad. |
| `engine_dominant` | fijo | Clase A, motor en ralentí (800 rpm, encendido en 26.7 Hz) con 0.1 m/s² RMS. |
| `low_noise_sensor` | fijo | Sensor de 100 µg/√Hz (cercano al iPhone XR). |
| `half_car_typical` | fijo | Half car (rebote + cabeceo, dos ejes): máscaras y contacto por eje. |
| `yang2004_benchmark` | fijo | Benchmark de Yang et al. (2004), usado en las pruebas. |
| `default_dataset` | aleatorio | 10 000 muestras con los rangos de la sección 6; 500 puentes y 200 vehículos reutilizados en 3 particiones disjuntas (70/15/15 %). |
| `smoke` | aleatorio | 50 muestras de `default_dataset` para pruebas rápidas (segundos). |
| `snr_controlled_dataset` | aleatorio | Como `default_dataset`, pero con relaciones puente/perturbación controladas (puente 5–10 veces sobre el ruido). |
| `ood_class_d` | aleatorio | Fuera de distribución: pavimento clase D (1000 muestras). |
| `ood_high_speed` | aleatorio | Fuera de distribución: 25–30 m/s. |
| `ood_noisy_sensor` | aleatorio | Fuera de distribución: 400–800 µg/√Hz [SUPUESTO]. |

Cada preset lleva su descripción (`vbsig presets`). Un preset puede heredar de otro con `extends: <preset>` y cambiar solo lo necesario.

## Configuración

Todos los parámetros, con su unidad y su fuente, están documentados en [`src/vehicle_bridge_signal/defaults.yaml`](src/vehicle_bridge_signal/defaults.yaml). Cualquier configuración se fusiona sobre esos valores por defecto, así que basta con escribir lo que cambia. Las secciones son `simulation`, `crossing`, `bridge`, `vehicle`, `roughness`, `engine`, `wheel`, `ambient`, `mount`, `sensor`, `snr_control` y `dataset`.

Un parámetro puede ser un valor fijo o una distribución (se muestrea de forma independiente en cada muestra):

```yaml
crossing:
  speed: {dist: uniform, low: 5.0, high: 25.0}          # m/s
engine:
  body_accel_rms: {dist: loguniform, low: 0.005, high: 0.1}
roughness:
  class: {dist: choice, values: [A, B, C], weights: [0.3, 0.4, 0.3]}
bridge:
  span: {dist: normal, mean: 30.0, std: 5.0, low: 15.0}  # normal truncada
```

- Cada perturbación se activa o desactiva por separado (`roughness.enabled`, `engine.enabled`, `wheel.enabled`, `ambient.enabled`, `mount.enabled` y, en el sensor, `white_noise`, `pink_noise`, `bias`, `drift`, `anti_alias`, `jitter`, `quantization`). Los interruptores también aceptan `{dist: choice, values: [true, false]}`.
- `simulate()` exige que estén fijos solo los parámetros que usa (los de un modelo de vehículo inactivo o de una perturbación apagada pueden seguir siendo distribuciones).
- Las claves que definen la forma de los arreglos (`simulation.*`, `vehicle.model`, `bridge.max_modes`, `bridge.n_label_modes`, `dataset.*`...) deben ser fijas.
- La validación ocurre al cargar: una clave mal escrita, un rango imposible (`low >= high`), $f_{s,\text{sim}}$ que no es múltiplo de $f_{s,\text{out}}$, una rugosidad o un motor por encima de la frecuencia de Nyquist de la simulación, etc., producen un error con un mensaje claro (y una sugerencia si la clave se parece a otra).
- Los números en notación científica (`3.3e9`) se leen como números (PyYAML los leería como texto).

## Modelo físico en breve

- **Puente:** viga de Euler-Bernoulli simplemente apoyada, superposición modal, $\phi_n = \sin(n\pi x/L)$, $f_n = n^2 f_1$, $M_n = \bar m L/2$, $EI = \bar m(2L^2f_1/\pi)^2$. Modos con $f_n \le \min(0.45f_{s,\text{out}}, 40)$ Hz (máximo 6). Amortiguamiento distinto por modo.
- **Vehículo:** `quarter_car` (por defecto, 2 GDL), `sprung_mass` (1 GDL) o `half_car` (rebote, cabeceo y dos ejes). Las rigideces salen de las frecuencias objetivo de rebote y de salto de rueda (Gillespie, 1992), y las frecuencias propias reales se guardan en los metadatos.
- **Contacto:** $u_c = w(x_a,t) + r(x_a)$, con velocidad analítica $\dot u_c = \sum_n(\phi_n\dot q_n + v\phi_n'q_n) + v\,r'(x_a)$.
- **Interacción:** `coupled` (por defecto), con Newmark de aceleración promedio sobre el sistema puente-vehículo de matrices variables; o `decoupled` (carga móvil y luego el vehículo). Se guarda $\mu = m_v/(\bar m L)$.
- **Rugosidad ISO 8608:** suma de armónicos con fases aleatorias y derivadas exactas, convención $n_0 = 0.1$ ciclos/m (A = 16, B = 64, C = 256, D = 1024 ×10⁻⁶ m³), periodo de 2048 m independiente de la velocidad, y suavizado por media móvil de 0.25 m (huella de la llanta).
- **Motor:** suma de órdenes ($N_{\text{cyl}}/2$, $N_{\text{cyl}}$, 0.5) con la fase integrada de un rpm suave; la amplitud se fija por la RMS resultante en la carrocería.
- **Otras perturbaciones** (apagadas por defecto): excentricidad de la rueda a $f_w = v/(\pi D)$, vibración ambiental del puente (fuerzas modales de ruido blanco con estado inicial estacionario) y soporte del teléfono no rígido.
- **Smartphone:** simulación a 1000 Hz, luego soporte, ruido (blanco con $\sigma = ND\cdot g\sqrt{f_s/2}$, y opcionalmente rosa, sesgo y deriva), anti-alias Butterworth de orden 4 en $0.4f_{s,\text{out}}$ con `sosfiltfilt`, diezmado a 200 Hz, *jitter* opcional, y cuantización y saturación (16 bits en ±2 g, con la gravedad incluida en el rango).
- **Ventana:** 2048 muestras a 200 Hz (10.24 s), con 4 s de precalentamiento antes (la suspensión ya está excitada al entrar) y 0.5 s después. El cruce completo y sus márgenes (1 s antes y 1 s después) siempre caben: si no, se vuelve a muestrear la velocidad, y nunca se recorta el cruce.

## Componentes (etiquetas)

Por muestra, en m/s², a $f_{s,\text{out}}$:

| Clave | Contenido |
|---|---|
| `bridge_mode_1` … `bridge_mode_K` | Respuesta de la carrocería debida a cada uno de los primeros $K$ modos ($K = 3$ por defecto, `bridge.n_label_modes`). Incluye la parte cuasiestática y la dinámica. Vale cero para modos que no existen en ese puente. |
| `bridge_higher_modes` | Suma de los modos mayores que $K$. |
| `roughness` | Respuesta debida a la rugosidad. |
| `engine` | Respuesta debida al motor. |
| `wheel` | Respuesta debida a la excentricidad de la rueda (cero si está apagada). |
| `noise` | `measured` menos la suma de las demás: ruido del sensor, cuantización, *jitter*, sesgo y deriva. Vale exactamente cero si todas esas etapas están apagadas. |
| `measured` | La señal del smartphone (entrada de la red). |

Reglas:
- Por linealidad, la aceleración se separa exactamente por camino: $\ddot y_s = \sum_n\mathcal G[\phi_n q_n] + \mathcal G[r] + \mathcal G[e] + \mathcal G_e[F_e]$. También vale en el modo acoplado, porque los $q_n$ ya contienen la retroalimentación. Cada muestra guarda `decomposition_residual` (diferencia entre la suma de los caminos y la solución acoplada, ~1e-14).
- Las componentes pasan por las mismas etapas lineales que la medición (soporte, anti-alias y diezmado), pero no por el ruido, el *jitter* ni la cuantización.
- Las componentes del puente valen exactamente cero antes de que el primer eje entre. Después de la salida contienen la vibración libre del vehículo que el puente excitó.
- Las mismas claves (con `total` en lugar de `noise`) se guardan para la aceleración del punto de contacto, por eje. Esa señal no contiene la frecuencia del vehículo (Yang et al., 2018).

**Modos de amplitud** (`simulation.amplitude_mode`):
- `physical` (por defecto): todo sale de los parámetros físicos. Con los rangos de la sección 6 (autos de 280 a 600 kg, $\mu$ de 0.03 % a 1 %), la componente del puente queda en mediana unos **50 dB por debajo de la rugosidad y 20 dB por debajo del ruido del sensor**. Es el resultado físico, no un error, y el cuaderno lo muestra.
- `snr_controlled`: el sensor conserva su densidad de ruido y su cuantización físicas; las entradas que excitan al puente (peso y vibración ambiental) se escalan para que el puente quede `snr_control.targets_db.sensor_noise` dB sobre el ruido blanco, y la rugosidad, el motor y la rueda se escalan respecto al puente. Se mide sobre el puente y a $f_{s,\text{out}}$. La descomposición sigue siendo exacta, pero la amplitud absoluta del puente deja de ser física (factor `scale_bridge_inputs`, típicamente de 100 a 1000).

## Formato del dataset

HDF5 (`h5py`), `float32`, un fragmento por muestra, compresión gzip (configurable). `schema_version = 1.0.0`.

| Ruta | Forma | Contenido |
|---|---|---|
| `/time` | (T,) | tiempo [s], t = 0 al inicio de la ventana |
| `/measured` | (N, T) | señal medida [m/s²] |
| `/components/<clave>` | (N, T) | cada componente de la tabla anterior |
| `/contact/<clave>` | (N, A, T) | aceleración del punto de contacto por eje (A = 1, o 2 en `half_car`) |
| `/bridge/midspan_displacement`, `/bridge/midspan_acceleration` | (N, T) | respuesta del puente al centro (anti-alias y diezmado, sin ruido) |
| `/on_bridge_mask` | (N, T) uint8 | 1 mientras algún eje está sobre el puente |
| `/on_bridge_mask_axles` | (N, A, T) uint8 | lo mismo, por eje |
| `/metadata/<campo>` | (N,) o (N, M) | tabla por muestra (los arreglos por modo se rellenan con NaN hasta `max_modes`) |

**Atributos del archivo:** `schema_version`, `config_yaml` (configuración completa resuelta), `base_seed`, `git_commit`, `git_dirty`, `package_version`, `created_utc`, `component_keys`, `contact_keys`, `fs_out`, `n_samples_out`, `n_axles`, `n_label_modes`, `vehicle_model`, `interaction`, `amplitude_mode`, `sampled_parameters`, `command`, versiones de Python, numpy y h5py.

**Metadatos por muestra (principales):**
- identificación: `sample_index`, `sample_seed`, `bridge_id`, `vehicle_id`, `partition`;
- parámetros sorteados: `param.<ruta>`, uno por cada parámetro aleatorio;
- puente: `span`, `f1`, `mass_per_length`, `flexural_rigidity`, `n_modes`, `bridge_frequencies`, `bridge_damping_ratios`;
- vehículo: `vehicle_model`, `vehicle_mass`, `vehicle_frequencies`, `vehicle_damping_ratios`, `vehicle_*` (rigideces, masas...), `mass_ratio`;
- cruce: `speed`, `entry_time`, `exit_time`, `crossing_duration`, `driving_frequency` = v/(2L);
- rugosidad: `roughness_class`, `roughness_gd_n0`, `roughness_waviness`, `roughness_smoothing_m`;
- motor: `engine_n_cylinders`, `engine_rpm_mean`, `engine_rpm_min/max`, `engine_firing_frequency`, `engine_body_accel_rms`; `wheel_frequency`;
- sensor: `noise_density_ug`, `n_saturated`;
- interruptores: `*_enabled`;
- relaciones sobre el puente: `rms_<componente>_on`, `ratio_db_bridge_to_<roughness|engine|wheel|noise|white_noise>`, `rms_bridge_from_<entrada>_on`, `scale_*`;
- calidad: `min_contact_force_ratio`, `wheel_lift_off`, `decomposition_residual`, `warnings`.

Las rutas y claves se mantendrán dentro de `schema_version` 1.x. Un cambio incompatible subirá la versión mayor, y el lector la verifica al abrir el archivo.

**Datos de ejemplo:** [`examples/example_smoke.h5`](examples/example_smoke.h5) tiene 8 muestras del preset `smoke` para revisar el formato sin instalar nada (generado con `vbsig generate --config smoke --n-samples 8 --set dataset.n_bridges=4 --set dataset.n_vehicles=4 --out examples/example_smoke.h5 --allow-inside-tool`). Los demás datasets (`*.h5`) están en el `.gitignore`.

## Reproducibilidad

- Una semilla base. La muestra `i` usa `SeedSequence(base_seed, spawn_key=(2, i))` (el hijo `i` de `SeedSequence(base_seed).spawn`). Los puentes y los vehículos usan `(0, b)` y `(1, v)`.
- Cada parámetro se sortea con su propio generador (semilla del dueño más un hash de la ruta del parámetro), así que cambiar un rango no altera los demás sorteos.
- Dentro de una simulación, cada fuente (cruce, rugosidad, motor, rueda, vibración ambiental, ruido blanco, ruido rosa, deriva, *jitter*) tiene un flujo aleatorio independiente.
- La misma configuración y la misma semilla dan los mismos arreglos bit a bit, con 1 o con N procesos (verificado en las pruebas). La muestra `i` de un dataset es exactamente `simulate(scenario_i, seed_i)`.
- La generación escribe en `<salida>.h5.partial` y renombra al final: si se interrumpe o falla una muestra, no queda un archivo que parezca completo, y el error indica el índice y la semilla de la muestra.

## Verificación

```powershell
cd sensing-techniques/tools/vehicle-bridge-signal-generator
python -m pytest                                   # 134 pruebas, 2-5 min (la de OpenSees se omite sin OVBI_DIR)
$env:OVBI_DIR = "<clon de OpenSees-VBI-Simulation>"; python -m pytest -m opensees   # prueba en vivo opcional
```

| # | Validación | Resultado |
|---|---|---|
| 1 | $f_n = n^2 f_1$, $\omega_n$ con $EI$ y $\bar m$ | exacto |
| 2 | Deflexión estática → $PL^3/(48EI)$; signo negativo bajo carga hacia abajo | el error coincide con la cola de $\sum n^{-4}$ (2e-6 con 41 modos) |
| 3 | Benchmark de Yang et al. (2004) frente a OpenSees-VBI-Simulation (commit `abbe91f`) | $f_{b1} = 2.084$ Hz, $f_v = 3.249$ Hz; $R^2$ = 0.99999 (deflexión) y 0.99997 (vehículo), en los modos acoplado y desacoplado; también en vivo con openseespy |
| 4 | Acoplado frente a desacoplado | con $\mu = 1$ % difieren 2.2 % en desplazamiento, igual que los dos modos del framework de referencia (2.2 %); la diferencia crece con $\mu$ |
| 5 | Contenido espectral (Yang et al., 2004) | la respuesta sobre el puente se reproduce con $R^2 > 0.999$ usando solo $f_v$, $nv/L$ y $f_{bn} \pm nv/(2L)$; a 5 m/s dominan las frecuencias del puente desplazadas y a 10 m/s aparece $f_v$ |
| 6 | PSD de la rugosidad y clase | Welch sigue a $G_d(n)$; clases A–D recuperadas, $w$ = 2.00 |
| 7 | Identidad de la descomposición y superposición por entrada | error < 1e-6 de la RMS (también en float32); suma de simulaciones por entrada = simulación completa (1e-10) |
| 8 | Ruido | RMS = $ND\,g\sqrt{f_s/2}$ ±5 %; PSD de salida = $(ND\,g)^2$ ±10 % |
| 9 | Aliasing | un tono de 150 Hz aparece en 50 Hz a 200 Hz sin filtro; con filtro se atenúa en $\lvert H(150)\rvert^2$ |
| 10 | Reproducibilidad | bit a bit con 1 y con N procesos |
| 11 | Interruptores | cada perturbación apagada da una componente exactamente cero |

Además: régimen permanente analítico de los caminos de rugosidad y motor (con la FRF del vehículo), solución cerrada de la carga móvil (Frýba), FRF frente a simulación en el tiempo, derivadas exactas del perfil, equivalencia chirp-z = suma directa = IFFT, estacionariedad de la vibración ambiental, cuantización, soporte, *jitter*, validación de configuraciones erróneas, muestreo y CLI.

## Supuestos

Los valores marcados [SUPUESTO] no tienen una fuente confirmada; son configurables y se discuten en la sección 10 del cuaderno.

| Supuesto | Valor por defecto |
|---|---|
| Factor sobre $f_1$ nominal | U[0.7, 1.3] |
| Amortiguamiento del puente | U[0.5 %, 3 %] por modo |
| Amortiguamiento de la suspensión | ζ U[0.2, 0.4] |
| Masa de `sprung_mass` | 250–500 kg (rango de la masa suspendida) |
| Half car | masa 500–1000 kg, distancia entre ejes 2.4–3.0 m, CG al 40–50 %, índice dinámico 1, teléfono en el CG, motor sobre el eje delantero |
| Límite inferior de la clase A | 8e-6 m³ (simétrico en escala logarítmica alrededor de 16e-6) |
| Suavizado por la huella de la llanta | media móvil de 0.25 m, activado (igual que el framework de referencia) |
| Motor | rpm 700–3000; fluctuación del 5 % con τ = 2 s; órdenes relativos 1.0 / 0.3 / 0.1; RMS en la carrocería 0.005–0.1 m/s² log-uniforme |
| Rueda | D 0.60–0.75 m; excentricidad 0.1–0.5 mm |
| Vibración ambiental | 0.001–0.01 m/s² de aceleración modal |
| Soporte del teléfono | 15–40 Hz, ζ 0.05–0.20 |
| Sensor | ruido rosa 300 µg/√Hz a 1 Hz; sesgo ±0.05 m/s²; deriva 0.001 m/s²/√s; *jitter* 0.2 ms |
| Márgenes del cruce | 1 s antes de la entrada y 1 s después de la salida |
| Ubicación del cruce en la ventana | aleatoria (datasets) o centrada (presets fijos) |
| Objetivos de `snr_controlled` | rugosidad −20 a 0 dB, motor y rueda −10 a 10 dB, ruido 14–20 dB (5–10 veces, CIV E 789) |
| Fuera de distribución | ruido de 400–800 µg/√Hz |
| Numéricos | $f_{s,\text{sim}}$ = 1000 Hz, 4 s de precalentamiento, 0.5 s de cola |

## Decisiones de diseño y limitaciones

**Compromisos entre los usos previstos:**
- *Deep learning* (separación): longitud fija, `float32`, identidad exacta de la suma y máscaras `on_bridge`. Con amplitudes físicas el puente es casi irrecuperable (−50 dB respecto a la rugosidad), así que existe `snr_controlled`, que da relaciones controladas a costa de la amplitud física absoluta del puente. `default_dataset` usa particiones disjuntas de puentes y vehículos (70/15/15 %): la división sin fuga no descarta muestras, a cambio de que puentes y vehículos de distintas particiones nunca se combinen. Un proyecto que no quiera particiones puede generar con `--set dataset.partitions=null` y usar `group_split(method="groups")`, que descarta las muestras cuyo puente y cuyo vehículo caen en conjuntos distintos.
- *Métodos clásicos* (EMD, filtros, filtro inverso): se guardan la frecuencia de muestreo y las frecuencias verdaderas del puente y del vehículo. El cruce se ubica al azar dentro de la ventana en los datasets (para que una red no aprenda la posición) y centrado en los presets fijos.
- *Estudios paramétricos*: el escenario fijo con `with_overrides`/`--set` mantiene las realizaciones aleatorias, y el perfil de rugosidad no depende de la velocidad.

**Limitaciones conocidas:**
- **Despegue de rueda:** con los rangos de la sección 6, cerca del 13 % de las muestras de `default_dataset` superan la carga estática con la fuerza dinámica de la llanta (vías clase C a alta velocidad con llantas rígidas). El modelo lineal no representa el despegue; esas muestras quedan marcadas (`wheel_lift_off`, `min_contact_force_ratio`) para filtrarlas si se desea.
- Las frecuencias desplazadas $f_{bn} \pm n v/(2L)$ están separadas por $1/T_{\text{cruce}}$: no se resuelven como dos picos en un solo cruce.
- El anti-alias (Butterworth de orden 4 en $0.4 f_{s,\text{out}}$) no es una pared perfecta: un tono a 1.5 veces el corte solo se atenúa unos 30 dB.
- Newmark de aceleración promedio tiene un error de periodo de $(\omega\Delta t)^2/12$: 0.5 % a 40 Hz con $f_{s,\text{sim}}$ = 1000 Hz (usar 2000 Hz si hace falta más).
- Modelo plano, contacto puntual, vehículo lineal, velocidad constante, una sola luz simplemente apoyada y vibración ambiental como ruido blanco modal.
- Rendimiento: ~0.4 s por muestra y proceso. Con 8 procesos, 10 000 muestras tardan unos 10 minutos y ocupan unos 0.8 GB.

## Estructura

```
sensing-techniques/tools/vehicle-bridge-signal-generator/
├── README.md
├── prompt_vehicle_bridge_signal_generator.md   especificación original
├── pyproject.toml                               paquete instalable (pip install -e)
├── requirements.txt
├── presets/                                     configuraciones YAML listas para usar
├── src/vehicle_bridge_signal/
│   ├── defaults.yaml        todos los parámetros, con unidades y fuentes
│   ├── config.py            carga, herencia, validación
│   ├── bridge.py            viga modal
│   ├── vehicle.py           sprung_mass, quarter_car, half_car
│   ├── roughness.py         ISO 8608
│   ├── engine.py            órdenes del motor
│   ├── disturbances.py      rueda y vibración ambiental
│   ├── vbi.py               Newmark acoplado y desacoplado
│   ├── sensor.py            cadena de medición
│   ├── simulate.py          simulación de un cruce y descomposición
│   ├── sampling.py          muestreo aleatorio y semillas
│   ├── dataset.py           HDF5, lectura, división por grupos
│   ├── spectral.py          Welch y frecuencias esperadas
│   ├── plotting.py          gráficas
│   ├── cli.py, generate.py  línea de comandos
├── tests/                   pytest; reference/ guarda la salida de OpenSees-VBI
├── notebooks/metodologia_generacion_senales.ipynb
└── examples/example_smoke.h5
```

## Referencias

- Talebi-Kalaleh, M., & Mei, Q. (2026). *An Open-Source Framework for Coupled Vehicle-Bridge Interaction Analysis Using OpenSees*. arXiv:2605.09175. https://github.com/MTalebi/OpenSees-VBI-Simulation
- Yang, Y. B., Lin, C. W., & Yau, J. D. (2004). Extracting bridge frequencies from the dynamic response of a passing vehicle. *Journal of Sound and Vibration*. doi:10.1016/S0022-460X(03)00378-X
- Yang, Y. B., Zhang, B., Qian, Y., & Wu, Y. (2018). Contact-point response for modal identification of bridges by a moving test vehicle. *IJSSD*, 18(5), 1850073.
- Gillespie, T. D. (1992). *Fundamentals of Vehicle Dynamics*. SAE.
- ISO 8608:2016. *Mechanical vibration — Road surface profiles — Reporting of measured data*.
- Agostinacchio, M., Ciampa, D., & Olita, S. (2014). The vibrations induced by surface irregularities in road pavements – a Matlab approach. *European Transport Research Review*.
- Loprencipe, G., & Zoccali, P. (2017). Use of generated artificial road profiles in road roughness evaluation. *Journal of Modern Transportation*.
- Malekjafarian, A., & OBrien, E. J. (2017). *Journal of Sound and Vibration*; Mei, Q., Gül, M., & Shirzad-Ghaleroudkhani, N. (2021). *CACAIE*; Shirzad-Ghaleroudkhani, N., Mei, Q., & Gül, M. (2020). *J. Bridge Eng.*; Eshkevari, S. S., et al. (2020). *MSSP*; Cantero, D., et al. (2024), benchmark NuBe.
- Elhattab, A., Uddin, N., & OBrien, E. (2019). *Sensors*, 19(14), 3143; Komarizadehasl, S., et al. (2024). *Frontiers of Structural and Civil Engineering*.
- Burdzik, R. *Identification of Vibrations in Automotive Vehicles*, cap. 11.
- Stoller, D., Ewert, S., & Dixon, S. (2018). Wave-U-Net. ISMIR 2018. arXiv:1806.03185
- Material de clase de CIV E 789 (Prof. Talebi-Kalaleh): Lectures 2, 3 y 4.
