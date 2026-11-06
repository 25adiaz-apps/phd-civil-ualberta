"""
Exporta uno o varios notebooks de Jupyter a un solo PDF controlando tamaño de
página, márgenes, tamaño de letra, etc.

Edita la sección CONFIGURACIÓN de abajo y ejecuta:
    python export_pdf.py
(o el botón ▶ "Run Python File" de VS Code)

Flujo: nbconvert (notebook -> HTML con CSS de impresión) -> Chrome/Edge headless (HTML -> PDF).
No requiere LaTeX.
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Notebook(s) a exportar:
#   None = abrir una ventana para elegirlo al ejecutar el script
#   una ruta (relativa a la carpeta de este script, o absoluta), ej. "mi_notebook.ipynb"
#   o una lista de rutas: se unen en ese orden en un solo PDF, cada una desde página nueva
NOTEBOOK = "ProjectPhase2_Group6/Main.ipynb"

# PDF de salida. None = mismo nombre que el (primer) notebook
OUTPUT = "ProjectPhase2_Group6.pdf"

# --- Página ---
PAGE_SIZE = "A4"        # "Letter", "A4", "Legal", "A3" o medidas: "210mm 297mm"
LANDSCAPE = False           # True = horizontal
MARGIN = "2cm"              # "2cm" | "arriba+abajo lados": "2.5cm 2cm" | "arriba der abajo izq": "2cm 1.5cm 2cm 1.5cm"
PAGE_NUMBERS = True        # "1 / 20" al pie de cada página

# --- Tipografía ---
FONT_SIZE = "12pt"          # texto markdown
H1_SIZE = "20pt"            # título H1 (H2, H3, H4 se escalan a partir de este)
CODE_FONT_SIZE = "11pt"      # código de entrada
OUTPUT_FONT_SIZE = "11pt"    # salidas de texto (print, errores...)
TABLE_FONT_SIZE = "10pt"     # tablas / DataFrames
FONT_FAMILY = "sans-serif, 'Calibri', 'Segoe UI', Arial, sans-serif"
CODE_FONT_FAMILY = "Consolas, 'Courier New', monospace"
LINE_HEIGHT = "1.4"         # interlineado del texto

# --- Contenido ---
# Etiquetas de celda que deciden qué se exporta. Se ponen en Jupyter desde la barra
# lateral derecha (Common Tools > Cell Tags) o en VS Code con el menú "..." de la celda.
# Una celda puede llevar varias; basta con que tenga una de la lista.
HIDE_INPUT_TAGS = ("hide-input", "remove-input")     # oculta el código, deja la salida
HIDE_OUTPUT_TAGS = ("hide-output", "remove-output")  # oculta la salida, deja el código
HIDE_CELL_TAGS = ("hide-cell", "remove-cell")        # no exporta la celda en absoluto
COLLAPSE_SECTION_TAGS = ("collapse-section",)        # en la celda de un título: exporta solo
                            # el título y omite todo lo que hay debajo hasta el siguiente título
                            # del mismo nivel o superior (como una sección minimizada).
                            # También vale minimizar la sección en JupyterLab y guardar: JupyterLab
                            # lo deja en el archivo; VS Code no guarda ese estado.
NO_INPUT = False            # True = ocultar el código, mostrar solo resultados
PROMPT_STYLE = "none"     # Cómo mostrar "In [1]:" / "Out [1]:"
                            #   "left"   = columna a la izquierda (estilo Jupyter, desplaza todo)
                            #   "margin" = etiqueta dentro del margen izquierdo; el contenido usa todo el ancho
                            #   "above"  = etiqueta pequeña encima de cada bloque de código
                            #   "none"   = sin etiquetas
IMAGE_WIDTH = "90%"        # ancho máximo de gráficas/imágenes, ej. "80%"
IMAGE_MAX_HEIGHT = "78vh"   # alto maximo de una grafica, para que no ocupe mas de
                            # una pagina y arrastre la leyenda a la siguiente
SPLIT_FIGURE_ROWS = True    # True = una gráfica de varios subplots se corta por las bandas
                            # blancas que separan sus filas: las filas que caben se quedan en
                            # la página y el resto pasa a la siguiente, sin hueco entre ellas
                            # ni espacio muerto al pie. Una gráfica de un solo eje no tiene
                            # esas bandas y viaja entera.
                            # False = cada gráfica se mueve entera a la página siguiente
                            # cuando no cabe, dejando el hueco.
SPLIT_MIN_BAND_PX = 6       # alto mínimo, en píxeles, de la banda blanca para tomarla como
                            # separación entre filas
SPLIT_MIN_PART = 0.12       # alto mínimo de cada trozo, como fracción de la imagen; por
                            # debajo de esto el corte se descarta
KEEP_CAPTION_WITH_FIGURE = True   # True = una grafica y su leyenda nunca quedan en
                            # paginas distintas
IMAGE_ALIGN = "center"      # alineación de gráficas/imágenes: "left", "center" o "right"
AVOID_CELL_BREAKS = False   # True = no cortar celdas de código entre páginas
TRUNCATE_WIDE_TABLES = True # True = una tabla de salida más ancha que la página oculta en el
                            # PDF las columnas del medio y pone una columna "…" en su lugar,
                            # dejando las primeras y las últimas. Cuántas se ocultan se calcula
                            # al exportar, midiendo la tabla. El notebook no se modifica.
EXTRA_CSS = ""              # CSS adicional para ajustes finos

# --- Otros ---
BROWSER = None              # None = autodetectar Chrome/Edge; o ruta al .exe
WAIT_MS = 10000             # ms de espera para que se rendericen fórmulas (MathJax)
KEEP_HTML = False           # True = guardar también el HTML intermedio

# =============================================================================

import base64
import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import nbformat
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import TagRemovePreprocessor
    from traitlets.config import Config
except ImportError:
    # Instala nbconvert en el mismo Python que ejecuta este script
    print(f"Instalando nbconvert en {sys.executable} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "nbconvert"], check=True)
    import nbformat
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import TagRemovePreprocessor
    from traitlets.config import Config

BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "chrome", "google-chrome", "chromium", "chromium-browser", "msedge",
]


def find_browser(explicit=None):
    for cand in ([explicit] if explicit else BROWSER_CANDIDATES):
        if Path(cand).is_file():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    sys.exit("No se encontró Chrome/Edge. Indica la ruta en BROWSER.")


# nbconvert maqueta cada celda como tabla (prompt | contenido); estos estilos la pasan a bloque
_BLOCK_LAYOUT = """
    .jp-InputArea, .jp-OutputArea-child {
        display: block !important; position: relative; overflow: visible !important;
    }
    .jp-Cell-inputWrapper, .jp-Cell-outputWrapper, .jp-OutputArea { overflow: visible !important; }
    .jp-InputArea-editor, .jp-OutputArea-output, .jp-RenderedMarkdown {
        display: block !important; width: auto !important; margin-left: 0 !important;
    }
    .jp-InputPrompt, .jp-OutputPrompt {
        display: block !important; width: auto !important; min-width: 0 !important;
        font-size: 7pt !important; white-space: nowrap;
    }
    .jp-MarkdownCell .jp-InputPrompt, .jp-OutputPrompt:empty { display: none !important; }
"""

PROMPT_CSS = {
    "margin": _BLOCK_LAYOUT + """
    .jp-Notebook { padding-left: var(--prompt-gutter) !important; }
    .jp-InputPrompt, .jp-OutputPrompt {
        position: absolute; left: auto !important; right: 100%; top: 0;
        padding: 0.5em 4px 0 0 !important; text-align: right !important;
    }
    """,
    "above": _BLOCK_LAYOUT + """
    .jp-InputPrompt, .jp-OutputPrompt {
        position: static !important;
        text-align: left !important; padding: 2px 0 1px 0 !important;
    }
    """,
    "none": """
    .jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
    .jp-InputArea-editor, .jp-OutputArea-output, .jp-RenderedMarkdown { margin-left: 0 !important; }
    """,
}


def build_css():
    orientation = " landscape" if LANDSCAPE else ""
    page_numbers = """
        @bottom-center {
            content: counter(page) " / " counter(pages);
            font-family: %s; font-size: 9pt; color: #555;
        }""" % FONT_FAMILY if PAGE_NUMBERS else ""

    # En modo "margin" Chrome recorta lo que sale del área de la página, así que se
    # le quita un "gutter" al margen izquierdo y las etiquetas van en ese espacio.
    gutter = "1.2cm"
    margin_left = ""
    if PROMPT_STYLE == "margin":
        parts = MARGIN.split()
        left = parts[{1: 0, 2: 1, 3: 1, 4: 3}[len(parts)]]
        margin_left = f"margin-left: max(calc({left} - {gutter}), 0.3cm);"

    return f"""
    @page {{
        size: {PAGE_SIZE}{orientation};
        margin: {MARGIN};
        {margin_left}{page_numbers}
    }}
    :root {{ --prompt-gutter: {gutter}; }}

    :root {{
        --jp-content-font-size1: {FONT_SIZE};
        --jp-code-font-size: {CODE_FONT_SIZE};
        --jp-content-font-family: {FONT_FAMILY};
        --jp-code-font-family: {CODE_FONT_FAMILY};
        --jp-content-line-height: {LINE_HEIGHT};
        --jp-code-line-height: 1.3;
    }}

    html, body {{
        margin: 0 !important;
        padding: 0 !important;
        background: white !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}
    body, .jp-Notebook {{ padding: 0 !important; overflow: visible !important; }}
    .jp-Cell {{ padding: 2px 0 !important; }}

    /* Texto (markdown) */
    .jp-RenderedMarkdown, .jp-RenderedHTMLCommon {{
        font-family: {FONT_FAMILY} !important;
        font-size: {FONT_SIZE} !important;
        line-height: {LINE_HEIGHT} !important;
    }}
    /* Fórmulas: MathJax las agranda (~117 %) para igualar la altura de la x de su fuente;
       se dejan al mismo tamaño que el texto que las rodea. */
    .MathJax_CHTML, .MathJax_Display .MathJax_CHTML {{ font-size: 100% !important; }}
    .jp-RenderedHTMLCommon h1 {{ font-size: {H1_SIZE} !important; }}
    .jp-RenderedHTMLCommon h2 {{ font-size: calc({H1_SIZE} * 0.82) !important; }}
    .jp-RenderedHTMLCommon h3 {{ font-size: calc({H1_SIZE} * 0.68) !important; }}
    .jp-RenderedHTMLCommon h4 {{ font-size: calc({H1_SIZE} * 0.58) !important; }}
    .jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2,
    .jp-RenderedHTMLCommon h3, .jp-RenderedHTMLCommon h4 {{
        break-after: avoid; page-break-after: avoid;
    }}

    /* Código de entrada */
    .jp-InputArea-editor, .highlight pre, .CodeMirror {{
        font-family: {CODE_FONT_FAMILY} !important;
        font-size: {CODE_FONT_SIZE} !important;
    }}
    .jp-InputArea-editor pre {{
        margin: 0 !important;
        white-space: pre-wrap !important;
        word-break: break-word;
    }}

    /* Salidas de texto */
    .jp-OutputArea-output pre {{
        font-family: {CODE_FONT_FAMILY};
        font-size: {OUTPUT_FONT_SIZE} !important;
        white-space: pre-wrap !important;
        word-break: break-word;
    }}

    /* Tablas (DataFrames) */
    .jp-RenderedHTMLCommon table {{ font-size: {TABLE_FONT_SIZE} !important; }}
    .jp-OutputArea-output {{ overflow: visible !important; max-width: 100%; }}

    /* Permitir que salidas largas (texto, tablas) se repartan entre páginas.
       nbconvert pone break-inside: avoid-page y las manda completas a la siguiente. */
    .jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-output,
    .jp-OutputArea-output pre, .jp-Cell-outputWrapper {{
        break-inside: auto !important; page-break-inside: auto !important;
        overflow: visible !important;
    }}
    .jp-OutputArea-output pre {{ orphans: 2; widows: 2; }}

    /* nbconvert maqueta cada celda como tabla (prompt | contenido) y Chrome no parte
       filas de tabla que caben en una página: la celda entera salta a la siguiente.
       Con flex/bloque, código y markdown también se reparten entre páginas. */
    .jp-Cell, .jp-Cell-inputWrapper, .jp-InputArea, .jp-InputArea-editor,
    .jp-RenderedMarkdown, .jp-CodeMirrorEditor, .highlight, .highlight pre {{
        break-inside: auto !important; page-break-inside: auto !important;
        overflow: visible !important;
    }}
    /* Ojo: flex tampoco sirve (Chrome no parte filas flex); se usa bloque y el
       prompt se posiciona a la izquierda en absoluto. */
    /* .jp-OutputArea es display:table en el tema de JupyterLab, y Chrome no respeta
       break-inside dentro de una tabla: sin esto el bloque imagen+leyenda se parte igual. */
    .jp-InputArea, .jp-OutputArea, .jp-OutputArea-child {{
        display: block !important; position: relative;
    }}
    .jp-InputPrompt, .jp-OutputPrompt {{
        display: block !important; position: absolute; left: 0; top: 0;
        width: var(--jp-cell-prompt-width) !important; min-width: 0 !important;
        box-sizing: border-box;
    }}
    .jp-InputArea-editor, .jp-RenderedMarkdown, .jp-OutputArea-output {{
        display: block !important; width: auto !important;
        margin-left: var(--jp-cell-prompt-width) !important;
    }}
    .jp-RenderedMarkdown p, .jp-RenderedMarkdown li, .highlight pre {{ orphans: 2; widows: 2; }}
    .jp-RenderedHTMLCommon tr {{ break-inside: avoid; }}

    /* Imágenes / gráficas */
    .jp-OutputArea-output img, .jp-RenderedImage img, svg {{
        max-width: {IMAGE_WIDTH} !important;
        max-height: {IMAGE_MAX_HEIGHT} !important;
        height: auto !important;
        object-fit: contain;
    }}

    /* Una grafica y su leyenda viajan juntas. El bloque lo arma CAPTION_GROUPING_JS en el
       HTML, porque son salidas hermanas de la misma celda y CSS no puede agruparlas. */
    .epdf-figure-block {{
        display: block !important;
        break-inside: avoid !important; page-break-inside: avoid !important;
    }}
    /* La leyenda de tabla va encima. No se envuelve, para que una tabla larga siga
       repartiendose entre paginas; solo se le pide que no se despegue de ella. */
    .epdf-caption-above {{
        break-after: avoid !important; page-break-after: avoid !important;
    }}

    /* Gráfica troceada por filas (SPLIT_FIGURE_ROWS). El bloque sí puede partirse, cada
       trozo no, y el último viaja pegado a la leyenda. Sin separación entre trozos: se
       anula el espacio entre líneas para que no asome ningún hueco. */
    .epdf-figure-block.epdf-split {{
        break-inside: auto !important; page-break-inside: auto !important;
    }}
    .epdf-splittable {{
        break-inside: auto !important; page-break-inside: auto !important;
    }}
    .epdf-strips {{
        break-inside: auto !important; page-break-inside: auto !important;
        font-size: 0 !important; line-height: 0 !important;
    }}
    .epdf-strip {{
        display: block !important;
        margin-top: 0 !important; margin-bottom: 0 !important;
        break-inside: avoid !important; page-break-inside: avoid !important;
    }}
    .epdf-tail {{
        break-inside: avoid !important; page-break-inside: avoid !important;
    }}
    .epdf-tail > .jp-OutputArea-child {{
        font-size: {FONT_SIZE} !important; line-height: {LINE_HEIGHT} !important;
        text-align: left !important;
    }}
    .jp-RenderedImage, .jp-OutputArea-output img {{
        break-inside: avoid; page-break-inside: avoid;
    }}
    .jp-RenderedImage, .jp-RenderedSVG {{ text-align: {IMAGE_ALIGN} !important; }}
    .jp-RenderedImage img, .jp-RenderedSVG svg {{
        display: block;
        margin-left: {"auto" if IMAGE_ALIGN in ("center", "right") else "0"} !important;
        margin-right: {"auto" if IMAGE_ALIGN in ("center", "left") else "0"} !important;
    }}
    {".jp-CodeCell { break-inside: avoid; page-break-inside: avoid; }" if AVOID_CELL_BREAKS else ""}
    {PROMPT_CSS.get(PROMPT_STYLE, "")}
    {EXTRA_CSS}
    """


CAPTION_GROUPING_JS = r"""
<script>
/* Agrupa cada gráfica con su leyenda para que Chrome no las reparta entre páginas.
   nbconvert deja la imagen y el "Figure N." como dos .jp-OutputArea-child hermanos.
   Si la gráfica venía troceada por filas, el bloque sí puede partirse: lo que no se
   separa es cada fila, y la última viaja pegada a la leyenda. */
(function () {
  var ES_LEYENDA = /^\s*(Figure|Table)\s+\d+\./;

  function leyenda(el) {
    var md = el.querySelector('.jp-RenderedMarkdown, .jp-RenderedHTMLCommon');
    if (!md) return null;
    var m = (md.textContent || '').match(ES_LEYENDA);
    return m ? m[1] : null;
  }
  function tieneImagen(el) { return !!el.querySelector('img, svg'); }

  /* .jp-RenderedImage lleva break-inside: avoid para que una imagen suelta no se
     parta; con la gráfica troceada eso impediría repartir las filas, así que a los
     contenedores de trozos se les levanta la prohibición. */
  document.querySelectorAll('.epdf-strips').forEach(function (c) {
    for (var p = c.parentElement; p && !p.classList.contains('jp-OutputArea');
         p = p.parentElement) {
      p.classList.add('epdf-splittable');
    }
  });

  document.querySelectorAll('.jp-OutputArea').forEach(function (area) {
    var hijos = Array.prototype.slice.call(area.children);
    var usados = [];
    hijos.forEach(function (hijo, i) {
      var tipo = leyenda(hijo);
      if (!tipo || usados.indexOf(hijo) !== -1) return;

      if (tipo === 'Table') {
        hijo.classList.add('epdf-caption-above');
        return;
      }

      var desde = -1;
      for (var j = i - 1; j >= 0; j--) {
        if (usados.indexOf(hijos[j]) !== -1 || leyenda(hijos[j])) break;
        if (tieneImagen(hijos[j])) { desde = j; break; }
      }
      if (desde < 0) return;

      var grupo = hijos.slice(desde, i + 1);
      var caja = document.createElement('div');
      caja.className = 'epdf-figure-block';
      grupo[0].parentNode.insertBefore(caja, grupo[0]);
      grupo.forEach(function (el) { caja.appendChild(el); usados.push(el); });

      var tiras = caja.querySelectorAll('.epdf-strip');
      if (tiras.length > 1) {
        caja.classList.add('epdf-split');
        var ultima = tiras[tiras.length - 1];
        var cola = document.createElement('div');
        cola.className = 'epdf-tail';
        ultima.parentNode.insertBefore(cola, ultima);
        cola.appendChild(ultima);
        cola.appendChild(hijo);
      }
    });
  });
})();
</script>
"""


# Tamaños de página con nombre, en mm (ancho, alto) en vertical
PAGE_SIZES_MM = {"letter": (215.9, 279.4), "a4": (210, 297), "legal": (215.9, 355.6), "a3": (297, 420)}


def content_width_css():
    """Ancho útil de la página (ancho de página menos márgenes laterales) como expresión CSS."""
    parts = PAGE_SIZE.split()
    if len(parts) == 1:
        w, h = (f"{v}mm" for v in PAGE_SIZES_MM[PAGE_SIZE.lower()])
    else:
        w, h = parts[0], parts[1]
    if LANDSCAPE:
        w, h = h, w
    m = MARGIN.split()
    right = m[{1: 0, 2: 1, 3: 1, 4: 1}[len(m)]]
    left = m[{1: 0, 2: 1, 3: 1, 4: 3}[len(m)]]
    return f"calc({w} - {left} - {right})"


TRUNCATE_TABLES_JS = r"""
<script>
/* Tablas más anchas que la página: oculta columnas del medio y pone "…" en su lugar.
   Se mide con el cuerpo al ancho útil de la página, que es el que tendrá al imprimir. */
(function () {
  document.body.style.width = '__CONTENT_WIDTH__';
  document.body.style.boxSizing = 'border-box';

  function filas(t) { return Array.prototype.slice.call(t.rows); }

  document.querySelectorAll('.jp-OutputArea-output table').forEach(function (t) {
    /* El contenedor de la salida se estira con la tabla: el límite es el borde derecho
       del cuerpo, que tiene el ancho útil de la página. */
    var limite = document.body.getBoundingClientRect().right;
    function cabe() { return t.getBoundingClientRect().right <= limite + 0.5; }
    if (cabe()) return;

    var rows = filas(t);
    var n = rows[0].cells.length;
    /* Solo tablas simples: todas las filas con las mismas celdas y sin celdas combinadas */
    var simple = rows.every(function (r) {
      return r.cells.length === n && Array.prototype.every.call(r.cells, function (c) {
        return c.colSpan === 1 && c.rowSpan === 1;
      });
    });
    if (!simple) return;

    /* Columnas de índice: las <th> iniciales de la primera fila del cuerpo */
    var cuerpo = t.tBodies[0] && t.tBodies[0].rows[0];
    var nIdx = 0;
    while (cuerpo && nIdx < n && cuerpo.cells[nIdx].tagName === 'TH') nIdx++;
    var nDatos = n - nIdx;
    if (nDatos < 3) return;

    var puntos = [];
    function quitarPuntos() { puntos.forEach(function (c) { c.remove(); }); puntos = []; }

    for (var ocultas = 1; ocultas <= nDatos - 2; ocultas++) {
      var izq = Math.ceil((nDatos - ocultas) / 2);
      var desde = nIdx + izq, hasta = desde + ocultas;   /* columnas [desde, hasta) */
      quitarPuntos();
      rows.forEach(function (r) {
        for (var j = nIdx; j < n; j++) {
          r.cells[j].style.display = (j >= desde && j < hasta) ? 'none' : '';
        }
        var ref = r.cells[desde];
        var p = document.createElement(ref.tagName === 'TH' ? 'th' : 'td');
        /* En una fila cuyas celdas ocultas están vacías no hace falta indicar nada */
        var vacias = true;
        for (var k = desde; k < hasta; k++) {
          if (r.cells[k].textContent.trim()) { vacias = false; break; }
        }
        p.textContent = vacias ? '' : '…';
        p.style.textAlign = 'center';
        r.insertBefore(p, ref);
        puntos.push(p);
      });
      if (cabe()) break;
    }
  });
})();
</script>
"""


def _puntos_de_corte(im):
    """Dónde separar una imagen en filas de subplots.

    Devuelve las coordenadas y de las bandas horizontales de píxeles blancos que cruzan la
    imagen de lado a lado: son los huecos entre una fila de subplots y la siguiente. Una
    gráfica de un solo eje no tiene ninguna y la lista sale vacía.
    """
    import numpy as np

    alto = im.size[1]
    gris = np.asarray(im.convert("L"))
    blancas = np.where((gris >= 250).all(axis=1))[0]
    if blancas.size == 0:
        return []

    bandas, actual = [], [blancas[0]]
    for f in blancas[1:]:
        if f == actual[-1] + 1:
            actual.append(f)
        else:
            bandas.append(actual)
            actual = [f]
    bandas.append(actual)

    puntos, anterior = [], 0
    for banda in bandas:
        if len(banda) < SPLIT_MIN_BAND_PX:
            continue
        centro = (banda[0] + banda[-1]) // 2
        if centro <= 0 or centro >= alto - 1:
            continue                      # el margen de arriba o el de abajo, no separan nada
        if centro - anterior < SPLIT_MIN_PART * alto:
            continue
        if alto - centro < SPLIT_MIN_PART * alto:
            continue
        puntos.append(centro)
        anterior = centro
    return puntos


def split_figure_rows(html):
    """Sustituye cada PNG de varias filas por sus filas, una imagen por fila."""
    try:
        from PIL import Image
        import numpy  # noqa: F401  (lo usa _puntos_de_corte)
    except ImportError as falta:
        print()
        print("  " + "!" * 72)
        print(f"  SPLIT_FIGURE_ROWS esta activo pero falta {falta.name}, que hace falta")
        print("  para trocear las graficas. El PDF se exporta SIN TROCEAR.")
        print(f"  Interprete en uso: {sys.executable}")
        print(f'  Para arreglarlo:  "{sys.executable}" -m pip install pillow numpy')
        print("  " + "!" * 72)
        print()
        return html

    patron = re.compile(
        r'<img([^>]*?)src="data:image/png;base64,([A-Za-z0-9+/=]+)"([^>]*?)>')
    troceadas = [0, 0]

    def trocear(m):
        antes, datos, despues = m.group(1), m.group(2), m.group(3)
        try:
            im = Image.open(io.BytesIO(base64.b64decode(datos)))
        except Exception:
            return m.group(0)
        puntos = _puntos_de_corte(im)
        if not puntos:
            return m.group(0)

        # nbconvert ya trae un class="" en la etiqueta: si se añade otro, el navegador
        # se queda con el primero y la clase del trozo no llega a aplicarse nunca.
        antes = re.sub(r'\s*class="[^"]*"', ' ', antes)
        despues = re.sub(r'\s*class="[^"]*"', ' ', despues)

        ancho, alto = im.size
        trozos = []
        for arriba, abajo in zip([0] + puntos, puntos + [alto]):
            buf = io.BytesIO()
            im.crop((0, arriba, ancho, abajo)).save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            trozos.append(
                f'<img{antes}class="epdf-strip" '
                f'src="data:image/png;base64,{b64}"{despues}>')
        troceadas[0] += 1
        troceadas[1] += len(trozos)
        return '<div class="epdf-strips">' + "".join(trozos) + "</div>"

    html = patron.sub(trocear, html)
    if troceadas[0]:
        print(f"  {troceadas[0]} gráficas troceadas en {troceadas[1]} filas")
    return html


# Celda que separa un notebook del siguiente al unirlos: fuerza página nueva
PAGE_BREAK_CELL = '<div style="break-before: page; page-break-before: always;"></div>'


def _heading_level(cell):
    """Nivel del título con el que empieza una celda markdown (1 para #, 2 para ##...); 0 si no es título."""
    if cell.cell_type != "markdown":
        return 0
    m = re.match(r"\s*(#{1,6})\s", cell.source)
    return len(m.group(1)) if m else 0


def _is_collapsed(cell):
    meta = cell.get("metadata", {})
    return bool(set(meta.get("tags", [])) & set(COLLAPSE_SECTION_TAGS)
                or meta.get("jp-MarkdownHeadingCollapsed"))


def collapse_sections(nb):
    """Deja solo el título de las secciones marcadas como minimizadas."""
    cells, level, hidden = [], 0, 0
    for cell in nb.cells:
        current = _heading_level(cell)
        if level and (not current or current > level):
            hidden += 1
            continue                      # dentro de una sección minimizada
        level = current if current and _is_collapsed(cell) else 0
        cells.append(cell)
    nb.cells = cells
    return hidden


def read_notebook(path):
    nb = nbformat.read(path, as_version=4)
    hidden = collapse_sections(nb)
    if hidden:
        print(f"  {path.name}: {hidden} celdas omitidas en secciones minimizadas")
    return nb


def merge_notebooks(nb_paths):
    """Une varios notebooks en uno solo, en orden, con un salto de página entre ellos."""
    merged = read_notebook(nb_paths[0])
    for path in nb_paths[1:]:
        nb = read_notebook(path)
        merged.cells.append(nbformat.v4.new_markdown_cell(PAGE_BREAK_CELL))
        merged.cells.extend(nb.cells)
    return merged


def notebook_to_html(nb_paths):
    nb = merge_notebooks(nb_paths)

    # Las etiquetas de celda deciden qué se queda fuera. El preprocesador de nbconvert
    # trabaja sobre el cuaderno en memoria, así que el archivo no se toca.
    cfg = Config()
    cfg.TagRemovePreprocessor.enabled = True
    cfg.TagRemovePreprocessor.remove_input_tags = set(HIDE_INPUT_TAGS)
    cfg.TagRemovePreprocessor.remove_all_outputs_tags = set(HIDE_OUTPUT_TAGS)
    cfg.TagRemovePreprocessor.remove_cell_tags = set(HIDE_CELL_TAGS)

    exporter = HTMLExporter(config=cfg, template_name="lab")
    exporter.register_preprocessor(TagRemovePreprocessor(config=cfg), enabled=True)
    exporter.exclude_input = NO_INPUT
    exporter.exclude_input_prompt = PROMPT_STYLE == "none"
    exporter.exclude_output_prompt = PROMPT_STYLE == "none"

    etiquetadas = sum(
        1 for c in nb.cells
        if set(c.get("metadata", {}).get("tags", []))
        & (set(HIDE_INPUT_TAGS) | set(HIDE_OUTPUT_TAGS) | set(HIDE_CELL_TAGS)))
    if etiquetadas:
        print(f"  {etiquetadas} celdas con etiqueta de ocultar")
    html, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": nb_paths[0].stem}})
    style = f"<style>{build_css()}</style>"
    if SPLIT_FIGURE_ROWS:
        html = split_figure_rows(html)
    html = html.replace("</head>", style + "\n</head>", 1)
    if TRUNCATE_WIDE_TABLES:
        js = TRUNCATE_TABLES_JS.replace("__CONTENT_WIDTH__", content_width_css())
        html = html.replace("</body>", js + "\n</body>", 1)
    if KEEP_CAPTION_WITH_FIGURE:
        html = html.replace("</body>", CAPTION_GROUPING_JS + "\n</body>", 1)
    return html


def html_to_pdf(html_path, pdf_path, browser):
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--virtual-time-budget={WAIT_MS}",  # deja tiempo a MathJax
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def choose_notebook(base):
    """Abre un diálogo para elegir el notebook; si no hay entorno gráfico, menú en consola."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # que la ventana no quede detrás de VS Code
        path = filedialog.askopenfilename(
            parent=root, title="Elige el notebook a exportar", initialdir=base,
            filetypes=[("Jupyter Notebook", "*.ipynb"), ("Todos los archivos", "*.*")],
        )
        root.destroy()
        if not path:
            sys.exit("Cancelado: no se eligió ningún notebook.")
        return Path(path)
    except Exception:  # sin tkinter o sin entorno gráfico
        notebooks = sorted(base.glob("*.ipynb"))
        if not notebooks:
            sys.exit(f"No hay notebooks en {base}")
        for i, nb in enumerate(notebooks, 1):
            print(f"  {i}. {nb.name}")
        choice = input("Número del notebook a exportar: ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(notebooks):
            sys.exit("Opción no válida.")
        return notebooks[int(choice) - 1]


def main():
    # Las rutas relativas se resuelven respecto a la carpeta del script,
    # así funciona sin importar desde dónde se ejecute.
    base = Path(__file__).resolve().parent
    if not NOTEBOOK:
        nb_paths = [choose_notebook(base)]
    elif isinstance(NOTEBOOK, (str, Path)):
        nb_paths = [base / NOTEBOOK]
    else:
        nb_paths = [base / nb for nb in NOTEBOOK]
    for nb_path in nb_paths:
        if not nb_path.is_file():
            sys.exit(f"No existe: {nb_path}")
    pdf_path = (base / OUTPUT) if OUTPUT else nb_paths[0].with_suffix(".pdf")
    browser = find_browser(BROWSER)

    print(f"Python: {sys.executable}")
    print(f"Convirtiendo {', '.join(nb.name for nb in nb_paths)} a HTML...")
    html = notebook_to_html(nb_paths)

    # Si el PDF anterior sigue abierto en un visor, Chrome no puede sobrescribirlo y
    # termina sin error: se borra antes para que el fallo se note.
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError:
            sys.exit(f"No se puede reemplazar {pdf_path.name}: ciérralo y repite.")

    print(f"Generando PDF con {Path(browser).name}...")
    if KEEP_HTML:
        html_path = pdf_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        html_to_pdf(html_path, pdf_path, browser)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "notebook.html"
            html_path.write_text(html, encoding="utf-8")
            html_to_pdf(html_path, pdf_path, browser)

    if not pdf_path.is_file():
        sys.exit(
            "No se escribió el PDF. La causa más común es que el archivo esté abierto en "
            "un visor: Chrome no puede sobrescribirlo. Ciérralo y vuelve a ejecutar."
        )
    print(f"Listo: {pdf_path}")


if __name__ == "__main__":
    main()
