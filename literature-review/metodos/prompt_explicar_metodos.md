# Instrucciones para explicar métodos del literature review

Cuando te pida que me expliques un método, sigue este flujo.

## Parte 1: Explicación en el chat

- Explícame una cosa a la vez, por partes. No me des todo de golpe; yo hago preguntas de seguimiento.
- Responde solo lo que pregunto, de forma directa, sin proponer siguientes pasos.
- Fórmulas renderizadas en LaTeX (nunca en bloque de código). Debajo de cada fórmula: qué significa cada símbolo y una descripción de la fórmula en palabras.
- Acompaña los conceptos con gráficas generadas con datos de ejemplo. Si te pido ver el efecto de los parámetros, hazla interactiva.
- Código en Python, con variables y comentarios en inglés.
- Si el método aparece en alguno de los artículos de mi revisión, explícalo en el contexto de ese artículo y verifica los datos contra el texto, no de memoria.
- Si no sabes algo o el artículo no lo dice, dilo.

## Parte 2: Prompt para Claude Code

Cuando te diga que ya terminamos con el método, dame un prompt completo para Claude Code que agregue ese método al notebook `literature-review/metodos/metodos.ipynb` de mi repositorio "phd-civil-ualberta" y haga commit y push a GitHub.

### Nivel de detalle
Nivel de detalle: conceptual. El objetivo es tener claro el concepto del método, no un tratamiento exhaustivo. Unas 10–12 celdas, máximo 3–4 gráficas, sin widgets ni deducciones salvo que yo lo pida.

### Ubicación
- Indicar en qué **sección de función** va el método (por ejemplo: Descomposición de señales, Envolvente de amplitud, Identificación modal, Separación de la respuesta del puente, Reconstrucción con datos dispersos, Cuantificación de incertidumbre). Si no encaja en una existente, indicar que se cree una nueva.
- Indicar que se actualice la tabla de contenido.

### Estructura: narrativa con orden lógico, no bloques separados
El método se explica como una secuencia lógica de pasos, del problema a la aplicación. **No** separar en bloques "concepto / fórmulas / código / gráficas". Cada paso de la explicación combina, en el punto donde hace falta, las herramientas del notebook:
- una celda Markdown corta que plantea la idea;
- la fórmula en LaTeX justo donde aparece, con el significado de cada símbolo y la descripción en palabras;
- la celda de código que la implementa;
- la gráfica que la muestra;
- una celda Markdown corta que interpreta lo que se ve en la gráfica.

Orden lógico (adaptar a cada método):
1. **Para qué sirve:** dos o tres líneas sobre qué resuelve el método y por qué las herramientas más simples no bastan.
2. **Idea central** con su fórmula y su gráfica.
3. **Requisitos o limitación principal** (qué debe cumplir la señal o en qué caso falla), con su gráfica si ayuda.
4. **Aplicación a puentes**, y en qué artículos de mi revisión aparece y cómo lo usan.
5. **Tabla de fortalezas y debilidades.**

### Herramientas del notebook a usar
- Encabezados jerárquicos y anclas para la tabla de contenido.
- LaTeX en línea y en bloque.
- Tablas Markdown para comparaciones y resúmenes.
- Recuadros de nota o advertencia para requisitos y errores comunes (blockquote con ícono o `<div>` HTML con borde).
- Bloques plegables (`<details><summary>`) solo si pido deducciones.
- Funciones auxiliares definidas una vez y reutilizadas en las celdas siguientes, para no repetir código.
- Gráficas con subplots cuando se comparan casos lado a lado.
- Widgets interactivos (`ipywidgets.interact`) solo si los pido. Como GitHub no renderiza widgets, cada widget debe ir acompañado de una gráfica estática equivalente que muestre varios valores del parámetro.

### Contenido obligatorio
- Todo autocontenido, sin depender de esta conversación.
- Incluir lo que yo haya aclarado o corregido durante la explicación.
- Texto explicativo en español; código (variables y comentarios) en inglés.
- Semilla fija para cualquier dato aleatorio.
- Ejecutar el notebook antes del commit para guardar las gráficas, actualizar `requirements.txt` si se usan librerías nuevas, y commit descriptivo (por ejemplo: "Add EMD to signal decomposition section").

Entrega el prompt dentro de un bloque de código para que lo pueda copiar.
