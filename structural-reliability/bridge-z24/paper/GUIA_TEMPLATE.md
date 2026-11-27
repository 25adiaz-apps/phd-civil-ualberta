# Guía del template de Optica

Comandos y técnicas que ilustraba el template (`opticajnl`) antes de borrar su contenido de ejemplo. Sirve de referencia al terminar el paper.

## Preámbulo

- `\documentclass[9pt,twocolumn,twoside]{opticajnl}` y `\journal{opticajournal}`.
- `\setboolean{shortarticle}{true}`: carta (*letter*) o tutorial; `false` para artículo de investigación o revisión.
- `\setboolean{memo}{true}`: imprime "Memorandum" en el encabezado (solo con `shortarticle`).
- `\usepackage{lineno}` y `\linenumbers`: números de línea para la revisión; se quitan en preprints de Optica Open.
- `\setboolean{displaycopyright}{false}`: sin copyright ni licencia en el envío.

## Autores

- `\author[1,*]{Nombre}`, `\affil[1]{Afiliación}`, `\affil[*]{correo}`.
- Un solo autor de correspondencia, marcado con asterisco. Los correos de los demás autores llevan el número de su afiliación.
- Si no hay autor de correspondencia, se puede usar una nota de contribución igual (`\authormark{3}The authors contributed equally to this work.`).

## Figuras

- No ponerlas al final del documento: van donde aparecerían en el artículo final y con el tamaño final.
- `\label{}` y `\ref{}` para numerarlas y citarlas.
- Ejemplo: `\begin{figure}[ht]`, `\centering`, `\fbox{\includegraphics[width=\linewidth]{archivo}}`, `\caption{...}`, `\label{fig:...}`.
- **Accesibilidad:** no distinguir elementos solo por el color (curvas azul y roja). Usar también líneas discontinuas o punteadas, distintos marcadores, etiquetas de texto o números.
- Figuras reimpresas con permiso: el pie lleva la línea de permiso que pide la editorial (p. ej. "Reprinted with permission from ... Copyright 2024 by ...").

## Tablas

- Llevan título, no pie: `\caption{\bf Título}` arriba de la tabla.
- La información adicional va como nota al pie dentro del entorno `table`, con superíndice: `$^\textit{a}$` en el título o la celda y `$^\textit{a}$Nota.` debajo del `tabular`.
- `\hline` arriba, debajo del encabezado y al final.

## Ecuaciones

- Entorno `equation` con `\label{eq:...}`, integrada en la frase que la rodea y seguida de "donde ..." para los símbolos.
- Se cita con `\ref{}` (o `\eqref{}`).

## Algoritmos

- Entornos `algorithm` y `algorithmic[1]` (paquete algpseudocode):
  `\caption{...}\label{alg:...}`, `\Procedure{Nombre}{$a,b$}`, `\State`, `\While{...}` ... `\EndWhile`, `\Comment{...}`, `\EndProcedure`.

## Material suplementario

- Se sube en Prism y se cita en el texto y en la sección final *Supplemental document* ("See Supplement 1 for supporting content.").
- JOCN y PRJ no aceptan PDF suplementarios.

## Citas de datos y código

- Datos: `M. Partridge, "Spectra evolution during coating," figshare (2014), http://dx.doi.org/...`.
- Código: `C. Rivers, "Epipy: Python tools for epidemiology," figshare (2014) [retrieved 13 May 2015], http://dx.doi.org/...`.

## Secciones finales (*back matter*)

Dentro de `\begin{backmatter} ... \end{backmatter}`, con `\bmsection{...}` (sin número), en este orden:

1. **Funding:** se genera con los datos de Prism; en preprints se escriben los financiadores y números de proyecto.
2. **Acknowledgment:** personas que contribuyeron u otra información que no va en Funding.
3. **Disclosures:** códigos de la política de conflictos de interés; si no hay: "The authors declare no conflicts of interest."
4. **Data availability** (obligatoria, salvo en preprints), una de estas:
   - Datos como material suplementario: "Data underlying the results presented in this paper are available in Dataset 1, Ref. [3]."
   - Datos citados pero no incluidos: "Data underlying the results presented in this paper are available in Ref. [3]."
   - Datos no públicos: "Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the authors upon reasonable request."
   - Sin datos: "No data were generated or analyzed in the presented research."
5. **Supplemental document:** solo si hay documento suplementario.

## Referencias

- BibTeX con `\cite{}`, `\bibliography{references}` y `\bibliographyfullrefs{references}` (lista completa extra para Optics Letters; no cuenta en la extensión).
- *Optics Letters* y los artículos cortos de *Optica* usan el estilo abreviado: sin título del artículo ni página final (automático con BibTeX).
- Hasta tres autores; si hay más, los tres primeros y *et al.*
- Tipos de entrada del ejemplo: `@article`, `@phdthesis`, `@incollection`, `@inproceedings`, `@misc` (sitio web, código).

## Biografías

- Solo para *Advances in Optics and Photonics* (aop): foto y biografía de cada autor con `wrapfigure` dentro de `minipage`.
