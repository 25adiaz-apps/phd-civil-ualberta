# Prompt para Claude Code — vehicle-bridge-signal-generator: generador de señales vehículo-puente (herramienta reutilizable)

## 0. Antes de escribir código

1. Lee este documento completo y el `CLAUDE.md` de la raíz del repositorio `phd-civil-ualberta`.
2. Esta herramienta **no pertenece a una sola materia**: la voy a usar en varios proyectos. Por eso va en una carpeta nueva de nivel superior, **`tools/vehicle-bridge-signal-generator/`** (este prompt ya está guardado ahí). Es una categoría nueva en la estructura del repositorio: en el README raíz y en `CLAUDE.md`, **agrega solo las líneas** que la describen (la carpeta `tools/` y la regla de que ahí van herramientas reutilizables entre proyectos), sin cambiar nada más. Respeta las reglas de `CLAUDE.md`: no toques los trabajos existentes.
3. Entra en **plan mode** y preséntame un plan corto: estructura, módulos, orden de implementación y pruebas. Pregúntame solo lo que bloquee el trabajo; para todo lo demás, toma la opción por defecto indicada aquí y anótala en el README de la herramienta como supuesto.
4. **Alcance:** solo la generación de señales y su documentación. El procesamiento de las señales (separación con deep learning, EMD, identificación de frecuencias, etc.) se hará después, **en otras carpetas**, usando esta herramienta. No implementes nada de eso aquí.

## 1. Contexto

Soy estudiante de doctorado en ingeniería civil (University of Alberta). Llevo **CIV E 779 Machine Learning for Engineers** (Prof. Qipei Mei y Prof. Yuezhen Gao) y **CIV E 789 Sensing Techniques and Data Analytics** (Prof. Mohammad Talebi-Kalaleh), de donde vienen las convenciones de procesamiento de señales indicadas abajo.

**Qué simula la herramienta.** Un smartphone dentro de un carro registra la aceleración vertical del vehículo mientras cruza un puente (monitoreo indirecto o *drive-by*). Esa señal mezcla la respuesta del puente con la dinámica de la suspensión, la rugosidad del pavimento, la vibración del motor y el ruido del sensor. La herramienta genera esas señales por simulación física, bajo **condiciones configurables**, y entrega además **cada componente por separado** como verdad de referencia.

**Para qué la voy a usar.** El primer uso será un proyecto de CIV E 779 en el que una red convolucional 1D separará la señal medida en sus componentes, en especial la contribución de cada modo del puente, y se comparará con EMD o EEMD. Pero la herramienta debe servir para cualquier otro proyecto: probar métodos de identificación de frecuencias, estudiar el efecto de la velocidad o de la rugosidad, generar ejemplos para clases, etc. **La física debe ser correcta, reproducible y verificable, y la forma de generar la señal debe quedar explicada con claridad.**

## 2. Entregables

1. Un paquete Python instalable, por ejemplo `vehicle_bridge_signal`, que otros proyectos del repositorio puedan usar con `pip install -e tools/vehicle-bridge-signal-generator`. Debe ofrecer:
   - Una **API de Python** estable y documentada, por ejemplo: `simulate(config, seed)` para una sola señal con parámetros fijos, `generate_dataset(config, n_samples, out_path, seed)` para un dataset con parámetros aleatorios, y `load_dataset(path)` para leerlo.
   - Un **script de línea de comandos** que haga lo mismo.
   - **Todos los parámetros configurables** desde un archivo YAML, y **cada fuente de ruido o perturbación activable o desactivable** de forma independiente.
2. **Dos modos de uso:**
   - **Escenario fijo:** el usuario fija todos los parámetros (un puente, un vehículo, una velocidad, una clase de rugosidad, qué ruidos van) y obtiene una señal determinista. Sirve para estudiar un caso concreto.
   - **Muestreo aleatorio:** cada parámetro se toma de un rango o distribución (sección 6) para generar datasets grandes.
3. **Condiciones predefinidas** (`presets`) en archivos YAML listos para usar, por ejemplo: `bridge_only_smooth` (pavimento liso, sin motor ni ruido), `typical_car_smartphone`, `rough_road`, `high_speed`, `engine_dominant`, `low_noise_sensor`, `smoke` (unas 50 muestras para pruebas rápidas) y `default_dataset` (10 000 muestras). Cada preset documenta para qué sirve.
4. Un cuaderno `metodologia_generacion_senales.ipynb` que explique claramente cómo se genera la señal (ver sección 9), pensado para que cualquier persona entienda y reutilice la herramienta.
5. Pruebas automáticas con `pytest` (ver sección 8).
6. Un README de la herramienta con instalación, ejemplos de uso desde Python y desde la línea de comandos, lista de presets, descripción del formato de salida, supuestos y referencias.

## 3. Reglas de trabajo

- **Código en inglés**: nombres de variables, funciones, docstrings y comentarios.
- **Narrativa de los cuadernos en español** (si decido entregar en inglés, te lo pediré).
- En los cuadernos, sigue un **orden narrativo lógico** que intercale texto, fórmulas en LaTeX, código y gráficas. Cada fórmula debe ir acompañada de lo que calcula, de qué depende y para qué sirve. No separes en bloques de "concepto", luego "fórmulas", luego "código".
- Ejecuta los cuadernos completos y guárdalos **con sus salidas**, para que las gráficas se vean en GitHub.
- Reproducibilidad total: una semilla base, y para la muestra `i` usa una semilla derivada, como `np.random.SeedSequence(base_seed).spawn(...)`. La misma configuración y semilla deben dar exactamente el mismo dataset.
- Unidades del SI en todo el código. Cada parámetro del YAML debe documentar su unidad.
- **No inventes valores físicos.** Usa los valores y rangos de la sección 6 con su fuente. Lo marcado como **[SUPUESTO]** no tiene una fuente confirmada: déjalo configurable y documéntalo como supuesto en el README y en el cuaderno.
- En CIV E 789 se exige usar agentes de código entendiendo y verificando lo que producen. Por eso, cada pieza física debe tener al menos una prueba que la compare con un resultado analítico o de referencia.
- Commit y push al terminar cada bloque funcional, con mensajes claros.

## 4. Modelo físico

### 4.1 Puente: viga simplemente apoyada de Euler-Bernoulli, por superposición modal

Formas modales, frecuencias y masa modal:

$$\phi_n(x)=\sin\frac{n\pi x}{L},\qquad \omega_n=\left(\frac{n\pi}{L}\right)^2\sqrt{\frac{EI}{\bar m}},\qquad M_n=\frac{\bar m L}{2}$$

$L$: luz; $EI$: rigidez a flexión; $\bar m$: masa por unidad de longitud. Se cumple $f_n=n^2 f_1$.

Ecuación de cada modo bajo las fuerzas de contacto de los ejes que están sobre el puente:

$$\ddot q_n+2\zeta_n\omega_n\dot q_n+\omega_n^2 q_n=\frac{1}{M_n}\sum_{a}\phi_n\big(x_a(t)\big)\,P_a(t)\,\mathbb{1}\big[0\le x_a(t)\le L\big]$$

$\zeta_n$: amortiguamiento del modo $n$, **distinto para cada modo**; $x_a(t)$: posición del eje $a$; $P_a$: fuerza que el eje ejerce sobre el puente. Deflexión: $w(x,t)=\sum_n\phi_n(x)\,q_n(t)$.

Parametriza el puente por $L$, $f_1$ y $\bar m$, y deriva $EI=\bar m\,(2L^2 f_1/\pi)^2$. Incluye todos los modos con $f_n\le f_{\max}$ (configurable, por defecto el menor entre $0.45\,f_{s,\text{out}}$ y 40 Hz), con un máximo configurable de 6.

**Define y documenta las convenciones de signo** (desplazamiento positivo hacia arriba, fuerza del vehículo sobre el puente hacia abajo) y verifícalas con la prueba de deflexión estática de la sección 8.

### 4.2 Vehículo (seleccionable en el YAML)

- `sprung_mass` (un grado de libertad, Yang et al. 2004): una masa sobre un resorte y un amortiguador.
- `quarter_car` (dos grados de libertad, **por defecto**): masa suspendida $m_s$ (carrocería) y no suspendida $m_u$ (eje y rueda), suspensión $(k_s,c_s)$ y llanta $(k_t,c_t)$:

$$\begin{bmatrix} m_u&0\\0&m_s\end{bmatrix}\begin{Bmatrix}\ddot y_u\\ \ddot y_s\end{Bmatrix}+\begin{bmatrix}c_t+c_s&-c_s\\-c_s&c_s\end{bmatrix}\begin{Bmatrix}\dot y_u\\ \dot y_s\end{Bmatrix}+\begin{bmatrix}k_t+k_s&-k_s\\-k_s&k_s\end{bmatrix}\begin{Bmatrix}y_u\\ y_s\end{Bmatrix}=\begin{Bmatrix}c_t\dot u_c+k_t u_c\\ F_e(t)\end{Bmatrix}$$

  Es la ecuación 3 de Talebi-Kalaleh & Mei (2026), a la que añado la fuerza del motor $F_e$ sobre la masa suspendida (sección 4.5).
- `half_car` (opcional): carrocería con rebote y cabeceo, y dos ejes separados una distancia $d$ (ecuaciones 4 y 5 del mismo artículo). Si lo implementas, la aceleración medida en el punto del teléfono, a una distancia $x_p$ del centro de gravedad, es $\ddot y_v + x_p\,\ddot\theta_v$ (documenta el signo).

**Desplazamiento en el punto de contacto** de cada eje:

$$u_c(t)=w\big(x_a(t),t\big)+r\big(x_a(t)\big)$$

$r$ es la rugosidad. La velocidad de entrada $\dot u_c$ debe calcularse de forma **analítica y consistente** (incluido $\dot r=v\,r'(x)$ con la derivada exacta del perfil), no derivando numéricamente el desplazamiento; el artículo de referencia insiste en esto para evitar aceleraciones espurias.

**La señal principal medida** es la aceleración vertical de la masa suspendida, $\ddot y_s$. Guarda también, como salida auxiliar, la **aceleración en el punto de contacto** $\ddot u_c$, que según Yang et al. (2018) no contiene la frecuencia del vehículo; servirá para experimentos posteriores.

### 4.3 Interacción vehículo-puente (seleccionable)

- `coupled` (**por defecto**): resuelve juntos los modos del puente y los grados de libertad del vehículo. Como en coordenadas modales el sistema es pequeño, ensambla en cada paso las matrices variables en el tiempo y usa **Newmark de aceleración promedio** ($\gamma=1/2$, $\beta=1/4$), o un esquema iterativo equivalente al Algoritmo 1 de Talebi-Kalaleh & Mei (2026). La fuerza sobre el puente es el peso estático más la fuerza dinámica de la llanta, $k_t(y_u-u_c)+c_t(\dot y_u-\dot u_c)$. Si la fuerza total indica despegue de la rueda, registra una advertencia en los metadatos.
- `decoupled`: el peso del vehículo actúa como carga móvil sobre el puente; luego, la deflexión resultante más la rugosidad excitan al vehículo (Algoritmo 2 del mismo artículo). Es exacto cuando la masa del vehículo tiende a cero.

Guarda en los metadatos la relación de masas $\mu=m_v/(\bar m L)$. Según el estudio paramétrico del artículo, $\mu$ es el principal predictor de la precisión del modo desacoplado.

**Ventana de simulación.** El vehículo parte de una zona de aproximación antes del puente, cruza y continúa por una zona de salida, todas con rugosidad. La rugosidad debe cubrir toda la ventana, para que la suspensión ya esté excitada al entrar al puente. Guarda una máscara `on_bridge` por muestra (por eje, si hay más de uno).

### 4.4 Rugosidad del pavimento: ISO 8608

PSD de desplazamiento en función de la frecuencia espacial $n$ (ciclos/m):

$$G_d(n)=G_d(n_0)\left(\frac{n}{n_0}\right)^{-w},\qquad n_0=0.1\ \text{ciclos/m}$$

$G_d(n_0)$ define la clase del pavimento y $w$ es la ondulación (*waviness*), igual a 2 según la norma.

Generación por suma de armónicos con fases aleatorias uniformes en $[0,2\pi)$ (ecuación 8 de Talebi-Kalaleh & Mei, 2026):

$$r(x)=\sum_{i}\sqrt{2\,G_d(n_i)\,\Delta n}\;\cos\left(2\pi n_i x+\varphi_i\right)$$

con $n_i$ entre $n_l=0.01$ y $n_u=10$ ciclos/m. Elige $\Delta n$ de forma que el periodo del perfil, $1/\Delta n$, sea al menos el doble de la **longitud total de la ventana**, no solo de la luz. Implementa la suma vectorizada o con IFFT; debe ser equivalente.

Suavizado opcional con media móvil: representa el efecto envolvente de la huella de la llanta. La longitud de la ventana es **[SUPUESTO]**, por ejemplo 0.25 m.

Para el cuaderno: a velocidad constante $v$, la frecuencia temporal es $f=v\,n$ y la PSD temporal es $G_t(f)=G_d(f/v)/v$. Eso explica por qué la rugosidad excita más al vehículo a mayor velocidad.

**ATENCIÓN con la tabla de clases.** Existen dos convenciones:
- Con $n_0=0.1$ ciclos/m, las medias geométricas son A = 16, B = 64, C = 256 y D = 1024 ×10⁻⁶ m³, con límites A < 32, B 32–128, C 128–512 y D 512–2048 (tabla 1 de Agostinacchio et al., 2014).
- Con $\Omega_0=1$ rad/m, los valores son 1, 4, 16 y 64 ×10⁻⁶ m³.

La tabla 1 de Talebi-Kalaleh & Mei (2026) muestra los valores 1, 4, 16, 64 y 256, pero los etiqueta con $n_0=0.1$ ciclos/m. **Usa la convención en ciclos/m con los valores de 16, 64, 256 y 1024**, documenta la discrepancia en el cuaderno y verifica la clase calculando la PSD del perfil generado.

### 4.5 Motor (activable)

En un motor de cuatro tiempos, la frecuencia de encendido es:

$$f_{\text{fire}}=\frac{\text{rpm}\cdot N_{\text{cyl}}}{120}$$

En un motor de 4 cilindros, la excitación dominante es de **segundo orden**, dos veces la velocidad de giro; para 600 a 6000 rpm cae entre 20 y 200 Hz (Burdzik).

Modelo: una fuerza vertical sobre la masa suspendida, como suma de órdenes del motor, con un rpm que varía lentamente:

$$F_e(t)=\sum_k A_k\sin\big(\Phi_k(t)\big),\qquad \Phi_k(t)=2\pi\,o_k\int_0^t\frac{\text{rpm}(\tau)}{60}\,d\tau+\varphi_k$$

$o_k$ es el orden: el de encendido, $N_{\text{cyl}}/2$, y opcionalmente su doble y un medio orden pequeño. La fase se calcula integrando la frecuencia instantánea, nunca como $2\pi f(t)\,t$. Haz que $\text{rpm}(t)$ sea un proceso suave, por ejemplo un ruido filtrado con un tiempo de correlación configurable.

**Amplitud [SUPUESTO].** No encontré un valor confirmado para autos de pasajeros; el único dato encontrado es de un camión volquete: 0.77 m/s² en el piso de la cabina en ralentí a 800 rpm. Por eso, parametriza la amplitud por la **RMS de aceleración resultante en la carrocería** dentro de un rango amplio configurable y calcula $A_k$ a partir de ella, aprovechando que el sistema es lineal.

Con $f_{s,\text{out}}=100$ o $200$ Hz, buena parte de la excitación del motor queda **por encima de Nyquist**. Por eso es obligatorio simular a una frecuencia interna alta y modelar el filtro anti-alias de forma explícita (sección 4.7).

### 4.6 Otras perturbaciones (activables, desactivadas por defecto)

- **Excentricidad o desbalance de la rueda**: un desplazamiento armónico en el contacto a la frecuencia de giro $f_w=v/(\pi D)$. Por ejemplo, con una llanta de 835 mm a 64 km/h es unos 7 Hz. La amplitud es **[SUPUESTO]**.
- **Vibración ambiental del puente**: tráfico de fondo o viento, como fuerzas modales de ruido blanco filtrado, de modo que el puente ya vibre antes de que entre el vehículo. El artículo de referencia modela el tráfico de fondo como fuerzas nodales aleatorias.
- **Soporte del teléfono no rígido**: un filtro de un grado de libertad entre la carrocería y el sensor. La frecuencia y el amortiguamiento son **[SUPUESTO]**. En CIV E 789 (Lecture 2) se mencionó el montaje no rígido como una limitación de los smartphones.

### 4.7 Cadena de medición del smartphone

Aplica en este orden, cada etapa activable:

1. Aceleración verdadera en el punto del sensor, a la frecuencia interna de simulación (por defecto 1000 Hz).
2. Filtro del soporte, si está activo.
3. Ruido del sensor:
   - Ruido blanco con densidad espectral $ND$ en µg/√Hz. Su desviación estándar a la frecuencia de muestreo interna es $\sigma=ND\cdot g\cdot\sqrt{f_{s,\text{sim}}/2}$; documenta la convención.
   - Opcional: ruido rosa, sesgo constante y deriva lenta tipo caminata aleatoria. En CIV E 789 se mencionaron la deriva de calibración y el filtrado del sistema operativo.
4. Filtro anti-alias, por ejemplo un Butterworth de orden 4 aplicado con `sosfiltfilt`, con corte en $0.4\,f_{s,\text{out}}$; activable para poder mostrar el aliasing cuando está apagado.
5. Diezmado a $f_{s,\text{out}}$ (por defecto 200 Hz).
6. Opcional: variación del intervalo de muestreo (*jitter*), remuestreando en instantes ligeramente irregulares.
7. Cuantización y saturación: paso $=2\cdot\text{rango}/2^{\text{bits}}$. Por ejemplo, el MPU-6500 tiene 16 384 LSB/g en ±2 g.

## 5. Descomposición exacta en componentes (las etiquetas)

Es la parte más importante. Dada la trayectoria, el vehículo es un sistema **lineal e invariante** cuya entrada es $u_c(t)$ más la fuerza del motor. Por eso, la aceleración medida se separa **exactamente** según el camino por el que llega:

$$\ddot y_s=\sum_{n=1}^{N}\mathcal{G}\Big[\phi_n\big(x_a(t)\big)\,q_n(t)\Big]+\mathcal{G}\big[r\big(x_a(t)\big)\big]+\mathcal{G}_e\big[F_e\big]+\dots$$

$\mathcal{G}$ es el operador del vehículo, de desplazamiento en el contacto a aceleración de la carrocería (y $\mathcal{G}_e$, de la fuerza del motor a esa aceleración). Esto vale **también en el caso acoplado**: los $q_n$ ya contienen la retroalimentación, y basta con volver a pasar cada parte de $u_c$ por el modelo del vehículo.

Etiquetas por muestra, en m/s², a $f_{s,\text{out}}$:

| Clave | Contenido |
|---|---|
| `bridge_mode_1` … `bridge_mode_K` | Respuesta de la carrocería debida a cada uno de los primeros $K$ modos del puente (por defecto $K=3$; configurable) |
| `bridge_higher_modes` | Suma de los modos mayores que $K$ |
| `roughness` | Respuesta debida a la rugosidad del pavimento |
| `engine` | Respuesta debida al motor |
| `wheel` | Respuesta debida al desbalance de la rueda (cero si está apagado) |
| `noise` | Medición menos la suma de las demás componentes; incluye ruido, cuantización, *jitter* y deriva |
| `measured` | La señal de entrada a la red |

Reglas:
- Las componentes limpias pasan por las **mismas etapas lineales deterministas** que la medición (soporte, anti-alias y diezmado), pero no por el ruido ni por la cuantización. Así, la identidad $\text{measured}=\sum\text{componentes}$ se cumple **exactamente**, que es lo que permite una capa de salida aditiva tipo Wave-U-Net.
- Las componentes de puente incluyen tanto la parte cuasiestática (frecuencia de paso) como la dinámica de cada modo. Documenta esto.
- Fuera del puente, las componentes de puente valen cero, salvo que la vibración ambiental esté activa.
- Guarda también las mismas componentes para la aceleración del punto de contacto.

**Escalado controlado por SNR (opcional).** Además del modo `physical`, en el que todo sale de los parámetros físicos, implementa un modo `snr_controlled`. Como el sistema es lineal, simula por separado la respuesta a cada **entrada**: peso del vehículo, rugosidad, motor, rueda y vibración ambiental. Cada simulación se descompone por caminos y el total es una suma ponderada. Así se pueden fijar relaciones objetivo entre la componente del puente y cada perturbación, medidas solo en la ventana sobre el puente, sin romper la consistencia física. Verifica con una prueba que la suma de las simulaciones por entrada sea igual a la simulación completa.

## 6. Parámetros y rangos por defecto

Cada rango se muestrea de forma independiente por muestra, con distribución uniforme salvo que se indique otra. Todo es configurable.

| Grupo | Parámetro | Por defecto / rango | Fuente o justificación |
|---|---|---|---|
| Simulación | $f_{s,\text{sim}}$ | 1000 Hz | Frecuencia interna alta para captar el motor y aplicar el anti-alias; el ejemplo del framework de referencia usa 500 Hz |
| | $f_{s,\text{out}}$ | 200 Hz (opción 100 Hz) | Proyecto de CIV E 789 y Mei et al. (2021) usan 200 Hz; un iPhone 11 registró a 100 Hz con Phyphox (Komarizadehasl et al., 2024) |
| | Longitud de la ventana | 2048 muestras (10.24 s a 200 Hz) | Potencia de 2, convención de CIV E 789 para FFT y Welch |
| Puente | $L$ | 15–60 m | Rango de la tabla 5 de Talebi-Kalaleh & Mei (2026) |
| | $f_1$ | Valor nominal interpolado en log-log a partir de 15 m → 8.03 Hz, 30 m → 3.63 Hz, 50 m → 2.05 Hz y 100 m → 0.75 Hz, multiplicado por U[0.7, 1.3] | Tabla 5 de Talebi-Kalaleh & Mei (2026); el factor es [SUPUESTO] |
| | $\bar m$ | 4000–20 000 kg/m | Yang et al. (2004): 4800 kg/m; puente B27 del benchmark NuBe: 19 372 kg/m |
| | $\zeta_n$ | U[0.5 %, 3 %] independiente para cada modo | [SUPUESTO]; el estudio paramétrico de referencia usó 0.3 % |
| Vehículo (`quarter_car`) | $m_s$ | 250–500 kg | Ejemplos: 240 kg (Smith y Wang, CDC 2008) y 560 kg (LBTU 2025) |
| | $m_u$ | 30–100 kg | Mismos ejemplos: 36 y 102 kg |
| | Frecuencia de rebote de la carrocería | 1.0–2.0 Hz | El framework de referencia indica ~1–3 Hz para el rebote; el ejemplo de LBTU da 1.24 Hz |
| | Frecuencia de salto de la rueda | 8–15 Hz | El framework de referencia indica ~10–15 Hz; el ejemplo de LBTU da 8.5 Hz |
| | Amortiguamiento de la suspensión | ζ 0.2–0.4 | [SUPUESTO] |
| | $c_t$ | 0 | Los benchmarks de referencia usan amortiguamiento de llanta nulo |
| | Derivación | Calcula $k_s$ y $k_t$ a partir de las frecuencias objetivo, después calcula las frecuencias propias reales del sistema de 2 GDL y guárdalas en los metadatos | — |
| Velocidad | $v$ | 5–25 m/s, constante | Estudio de velocidad de Talebi-Kalaleh & Mei (2026): 5–30 m/s; Mei et al. (2021): 40 km/h |
| Rugosidad | Clase | A, B o C con pesos configurables (por defecto 0.3, 0.4, 0.3); $G_d(n_0)$ log-uniforme dentro de los límites de la clase | ISO 8608; las vías pavimentadas rara vez son peores que D (Loprencipe y Zoccali, 2017) |
| | $w$ | 2 (opción U[1.5, 3.5]) | ISO 8608 sugiere 2; en mediciones reales de vías, $w$ varía entre 1.5 y 3.5 con media cercana a 2.5 (Loprencipe y Zoccali, 2017) |
| | $n_l$, $n_u$ | 0.01 y 10 ciclos/m | Talebi-Kalaleh & Mei (2026) |
| Motor | $N_{\text{cyl}}$ | {3, 4, 6}, por defecto con mayor peso en 4 | — |
| | rpm medio | 700–3000 | [SUPUESTO]: ralentí hasta crucero |
| | RMS objetivo en la carrocería | 0.005–0.1 m/s², log-uniforme | **[SUPUESTO]**, sin fuente confirmada para autos de pasajeros |
| Sensor | $ND$ | 100–400 µg/√Hz | iPhone XR, eje Z: 110 µg/√Hz medidos (Komarizadehasl et al., 2024); MPU-6500: 300 µg/√Hz según el fabricante, pero RMS medido de 10⁻² g a 100 Hz (Elhattab et al., 2019) |
| | Cuantización | 16 bits en ±2 g | MPU-6500: 16 384 LSB/g |
| | Sesgo y deriva | Desactivados por defecto; magnitudes [SUPUESTO] | — |
| Contexto | Margen señal-ruido de un smartphone | Solo 5–10 veces sobre una respuesta típica de puente | CIV E 789, Lecture 2 |

Restricción: el cruce completo, $L/v$, más márgenes de aproximación y salida configurables, debe caber en la ventana. Si no cabe, vuelve a muestrear $v$; no recortes el cruce.

## 7. Dataset

- Formato **HDF5** (`h5py`) con fragmentos (*chunks*) por muestra y `float32`. Contenido:
  - Arreglos: `measured`, una por cada componente, `on_bridge_mask`, las mismas componentes para el punto de contacto y el vector de tiempo.
  - Metadatos por muestra en una tabla: todos los parámetros muestreados, frecuencias del puente y del vehículo, $\mu$, clase y $G_d(n_0)$, rpm, semilla e indicadores de qué perturbaciones están activas.
  - La configuración YAML completa y el hash del commit, como atributos del archivo.
- Línea de comandos, por ejemplo: `python -m vehicle_bridge_signal.generate --config presets/default_dataset.yaml --n-samples 10000 --out <ruta-del-proyecto>/data/vehicle_bridge_v1.h5 --workers 8`, más `--preview N` para graficar N muestras sin generar el dataset completo, y un subcomando para simular un solo escenario fijo y graficarlo.
- Generación en paralelo con `multiprocessing`, con resultados idénticos sin importar el número de procesos.
- La ruta de salida la decide quien llama a la herramienta: cada proyecto guarda sus datos en su propia carpeta. La herramienta no guarda datasets dentro de `tools/`.
- **Identificadores para dividir sin fuga de información:** guarda en los metadatos un `bridge_id` y un `vehicle_id`, y permite generar varios cruces del mismo puente y del mismo vehículo. Así, un proyecto que entrene un modelo podrá separar entrenamiento, validación y prueba **por puente y por vehículo**, no por muestra. Incluye una función auxiliar opcional para hacer esa división por grupos, sin imponerla.
- Incluye presets **fuera de distribución** (por ejemplo clase D, velocidades de 25 a 30 m/s o $ND$ más alto) para que otros proyectos prueben generalización.
- Agrega al `.gitignore` los patrones de datos generados (por ejemplo `*.h5`); sube solo configuraciones y, si pesa menos de 5 MB, un dataset de ejemplo muy pequeño.
- Versiona el formato de salida (por ejemplo, un atributo `schema_version`) y documenta cada campo, para que los proyectos que lo usen no se rompan si la herramienta cambia.

## 8. Validaciones obligatorias (`pytest`)

1. **Frecuencias del puente:** coinciden con $f_n=n^2 f_1$ y con $EI$ y $\bar m$.
2. **Deflexión estática:** con una carga constante en el centro del vano y suficientes modos, la deflexión al centro converge a $PL^3/(48EI)$. Esto confirma los signos y la masa modal.
3. **Benchmark de Yang et al. (2004):** puente de 25 m, $E=2.75\times10^{10}$ Pa, $I=0.12$ m⁴, $\bar m=4800$ kg/m, sin amortiguamiento; vehículo de 1200 kg con $k=5\times10^5$ N/m; $v=10$ m/s; pavimento liso. Se debe obtener $f_{b1}\approx2.08$ Hz y $f_v\approx3.25$ Hz, y la deflexión al centro y el desplazamiento del vehículo deben coincidir con el framework de los profesores, `https://github.com/MTalebi/OpenSees-VBI-Simulation` (configuración `quarter_car_simple_Yang2004`), con $R^2>0.99$. Esta prueba puede depender de `openseespy`; márcala como opcional si no está instalado, pero ejecútala al menos una vez y reporta el resultado.
4. **Acoplado frente a desacoplado:** con $\mu\approx1\,\%$, las respuestas deben ser casi idénticas, y la diferencia debe crecer con $\mu$. Grafícalo en el cuaderno.
5. **Contenido espectral esperado:** en la respuesta del vehículo con pavimento liso deben aparecer la frecuencia del vehículo, la frecuencia de paso $v/(2L)$ y las frecuencias del puente desplazadas, $f_{bn}\pm n\,v/(2L)$ (Yang et al., 2004). Compruébalo con un caso de baja velocidad y documenta lo que salga.
6. **Rugosidad:** la PSD del perfil generado sigue $G_d(n)$ en la banda $[n_l,n_u]$ y su clase calculada coincide con la pedida.
7. **Identidad de la descomposición:** `measured` es igual a la suma de las componentes con un error menor que 1e-6 de la RMS, y la suma de las simulaciones por entrada es igual a la simulación completa.
8. **Ruido:** la RMS del ruido simulado coincide con $ND\cdot g\cdot\sqrt{f_s/2}$ dentro de ±5 %.
9. **Aliasing:** sin el filtro anti-alias, un tono del motor por encima de Nyquist aparece en la frecuencia doblada que predice la teoría; con el filtro, se atenúa.
10. **Reproducibilidad:** la misma semilla da el mismo arreglo, bit a bit, con 1 y con N procesos.
11. **Interruptores:** cada perturbación desactivada da una componente exactamente igual a cero.

## 9. Cuaderno: metodología de generación de señales

Secciones en este orden, cada una con texto, fórmulas, código y gráficas, usando el paquete. No reimplementes la física en el cuaderno.

1. **El problema drive-by:** qué mide el teléfono y por qué la respuesta del puente queda enmascarada. Incluye un esquema y conecta con CIV E 789, Lecture 2: el smartphone dentro del vehículo y el reto de separar el puente de la suspensión y de la rugosidad.
2. **Puente:** formas modales, frecuencias y ecuación modal; gráfica de los modos.
3. **Vehículo:** ecuaciones del cuarto de carro, frecuencias propias y función de respuesta en frecuencia del contacto a la carrocería; muestra por qué el vehículo amplifica cerca de su propia frecuencia y filtra lo demás.
4. **Rugosidad ISO 8608:** PSD, tabla de clases (incluida la discrepancia de convenciones), generación, verificación de la PSD con Welch y conversión de espacio a tiempo.
5. **Interacción:** acoplado frente a desacoplado y el papel de $\mu$.
6. **Motor y rueda:** órdenes del motor, frecuencia de encendido y una demostración de aliasing a 100 y 200 Hz.
7. **Cadena de medición:** densidad de ruido, cuantización, filtro anti-alias, diezmado y comparación con el margen de 5–10 veces mencionado en CIV E 789.
8. **Descomposición:** gráfica apilada de una muestra con todas sus componentes; espectros de cada componente con Welch y un espectrograma STFT; señala las frecuencias del puente desplazadas.
9. **El dataset:** histogramas de los parámetros muestreados, distribución de las relaciones señal-ruido entre puente y perturbaciones, y ejemplos de muestras fáciles y difíciles.
10. **Supuestos y limitaciones:** una tabla con cada supuesto, su valor, su justificación y su efecto esperado. Por ejemplo: modelo plano 2D, Euler-Bernoulli, contacto puntual, vehículo lineal y velocidad constante.

**Convenciones de procesamiento de CIV E 789** para todo análisis espectral: Welch en lugar de la FFT del registro completo, ventana Hann, 50 % de solape y `nperseg` potencia de 2 y menor que N. Recuerda que $\Delta f=f_s/\text{nperseg}$ y que agregar ceros no mejora la resolución. Si se estima el pico con interpolación parabólica sobre el logaritmo de la PSD, debe declararse como un supuesto propio, porque no se vio en clase.

## 10. Cómo la usarán otros proyectos (no implementar aquí)

Para que el diseño sirva a los usos futuros, ten en cuenta estos consumidores, pero **no los implementes**:

- **Separación de componentes con deep learning** (primer uso, en una carpeta de `machine-learning/`): leerá `measured` como entrada y las componentes como salidas. Necesita longitud fija, `float32`, la identidad exacta de la suma de componentes, la máscara `on_bridge` y los identificadores de puente y vehículo.
- **Métodos clásicos** (EMD, EEMD, filtros, filtro inverso con el espectro fuera del puente): necesitan la señal medida, la frecuencia de muestreo y las frecuencias verdaderas del puente y del vehículo en los metadatos.
- **Estudios paramétricos:** necesitan el modo de escenario fijo y poder variar un parámetro a la vez manteniendo los demás.

Si alguna decisión de diseño favorece a un consumidor y perjudica a otro, explícamelo en el plan.

## 11. Estructura sugerida

```
tools/vehicle-bridge-signal-generator/
├── README.md
├── prompt_vehicle_bridge_signal_generator.md   # este documento
├── pyproject.toml          # numpy, scipy, h5py, pyyaml, matplotlib, pytest; extra opcional: opensees
├── presets/  bridge_only_smooth.yaml  typical_car_smartphone.yaml  rough_road.yaml  ...  smoke.yaml  default_dataset.yaml
├── src/vehicle_bridge_signal/
│   ├── config.py  bridge.py  vehicle.py  roughness.py  engine.py
│   ├── vbi.py  sensor.py  decomposition.py  sampling.py  dataset.py
│   ├── plotting.py  generate.py
├── tests/
└── notebooks/  metodologia_generacion_senales.ipynb
```

## 12. Referencias

- Talebi-Kalaleh, M., & Mei, Q. (2026). *An Open-Source Framework for Coupled Vehicle-Bridge Interaction Analysis Using OpenSees*. arXiv:2605.09175. Código: https://github.com/MTalebi/OpenSees-VBI-Simulation. (Ecuaciones del vehículo, algoritmos acoplado y desacoplado, generación ISO 8608, benchmarks y estudio de $\mu$.)
- Yang, Y. B., Lin, C. W., & Yau, J. D. (2004). Extracting bridge frequencies from the dynamic response of a passing vehicle. *Journal of Sound and Vibration*. doi:10.1016/S0022-460X(03)00378-X
- Yang, Y. B., Zhang, B., Qian, Y., & Wu, Y. (2018). Contact-point response for modal identification of bridges by a moving test vehicle. *International Journal of Structural Stability and Dynamics*, 18(5), 1850073. doi:10.1142/S0219455418500736
- Malekjafarian, A., & OBrien, E. J. (2017). On the use of a passing vehicle for the estimation of bridge mode shapes. *Journal of Sound and Vibration*. doi:10.1016/j.jsv.2017.02.051 (EMD y Hilbert en drive-by.)
- Mei, Q., Gül, M., & Shirzad-Ghaleroudkhani, N. (2021). Towards smart transportation system: A vehicle scanning method for bridge health monitoring. *Computer-Aided Civil and Infrastructure Engineering*. doi:10.1111/mice.12650
- Shirzad-Ghaleroudkhani, N., Mei, Q., & Gül, M. (2020). Frequency identification of bridges using smartphones on vehicles with variable features. *Journal of Bridge Engineering*, 25(7), 04020041.
- Eshkevari, S. S., Pakzad, S. N., Takáč, M., & Matarazzo, T. J. (2020). Bridge modal identification using acceleration measurements within moving vehicles. *Mechanical Systems and Signal Processing*. doi:10.1016/j.ymssp.2020.106733 (Elimina la dinámica del vehículo con su función de transferencia o con EEMD, y la rugosidad con SOBI.)
- Cantero, D., et al. (2024). Numerical benchmark for road bridge damage detection from passing vehicles responses applied to four data-driven methods. doi:10.1007/s43452-024-01001-9 (Benchmark NuBe.)
- ISO 8608:2016. *Mechanical vibration — Road surface profiles — Reporting of measured data*.
- Agostinacchio, M., Ciampa, D., & Olita, S. (2014). The vibrations induced by surface irregularities in road pavements – a Matlab approach. *European Transport Research Review*. doi:10.1007/s12544-013-0127-8 (Tabla de clases en $n_0$ y $\Omega_0$.)
- Loprencipe, G., & Zoccali, P. (2017). Use of generated artificial road profiles in road roughness evaluation. *Journal of Modern Transportation*. doi:10.1007/s40534-017-0122-1 (Ondulación real de 1.5 a 3.5.)
- Elhattab, A., Uddin, N., & OBrien, E. (2019). Extraction of bridge fundamental frequencies utilizing a smartphone MEMS accelerometer. *Sensors*, 19(14), 3143.
- Komarizadehasl, S., et al. (2024). Comparación de densidad de ruido entre el acelerómetro LARA y un iPhone XR. *Frontiers of Structural and Civil Engineering*. doi:10.1007/s11709-024-1055-5
- Burdzik, R. *Identification of Vibrations in Automotive Vehicles*, cap. 11 (motor como fuente de vibración).
- Stoller, D., Ewert, S., & Dixon, S. (2018). Wave-U-Net: A multi-scale neural network for end-to-end audio source separation. ISMIR 2018. arXiv:1806.03185
- Material de clase de CIV E 789 (Prof. Talebi-Kalaleh): Lecture 2 (smartphones y drive-by), Lecture 3 (muestreo, aliasing y tipos de ruido) y Lecture 4 (Welch, resolución, STFT).
