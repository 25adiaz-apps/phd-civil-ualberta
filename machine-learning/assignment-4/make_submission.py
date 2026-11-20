"""
Arma el ZIP del entregable con los notebooks de las respuestas y los archivos que
necesitan para ejecutarse, cada uno en la misma ruta relativa que usa el código.

Edita la sección CONFIGURACIÓN y ejecuta:
    python make_submission.py

Antes de crear el ZIP revisa el código de los notebooks: cada ruta entre comillas que
exista en esta carpeta debe ir en el ZIP (o estar en NOT_NEEDED); si falta alguna, avisa.
También cuenta las páginas del PDF del informe contra el límite del enunciado.
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Notebooks con las respuestas (rutas relativas a la carpeta de este script)
NOTEBOOKS = [
    "question-1.ipynb",
    "question-2.ipynb",
]

# Archivos y carpetas que los notebooks necesitan, con la ruta que usa el código
DATA = [
    "train-val.csv",                # Question 1
    "test.csv",                     # Question 1
    "question-2-images/set1",       # Question 2, step 9
    "question-2-images/set2",       # Question 2, step 9
]

# Rutas que el código menciona pero que no hace falta entregar
NOT_NEEDED = {
    "mnist": "torchvision la descarga sola (download=True)",
}

# ZIP de salida
OUTPUT = "assignment-4-code.zip"

# PDF del informe (se entrega aparte). Solo se revisa su número de páginas.
REPORT_PDF = "assignment-4.pdf"
MAX_PDF_PAGES = 30

# Archivos que nunca se meten en el ZIP
IGNORE = ("__pycache__", ".ipynb_checkpoints", ".DS_Store", "Thumbs.db", "desktop.ini")

# =============================================================================

import json
import re
import sys
import zipfile
from pathlib import Path


def collect_files(base, entries):
    """Expand each entry (file or folder) into the list of files it contains."""
    files = []
    for entry in entries:
        path = base / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.is_file() and not set(p.relative_to(base).parts) & set(IGNORE))
        else:
            sys.exit(f"No existe: {path}")
    return files


def referenced_paths(nb_path):
    """String literals in the notebook's code cells that look like relative paths."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    paths = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for line in source.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for literal in re.findall(r"""['"]([^'"\n]+)['"]""", line):
                literal = literal.strip()
                if "/" in literal or "\\" in literal or "." in literal:
                    paths.add(literal)
    return paths


def check_references(base, notebooks, included):
    """Warn about paths used by the code that exist here but are not in the ZIP."""
    included_rel = {p.relative_to(base).as_posix() for p in included}
    missing = []
    for nb_path in notebooks:
        for literal in sorted(referenced_paths(nb_path)):
            rel = literal.replace("\\", "/").removeprefix("./").rstrip("/")
            if not re.search(r"\w", rel) or rel.startswith("/") or ":" in rel:
                continue
            if not (base / rel).exists():
                continue
            if rel.split("/")[0] in NOT_NEEDED or rel in NOT_NEEDED:
                continue
            covered = rel in included_rel or any(f.startswith(rel + "/") for f in included_rel)
            if not covered:
                missing.append((nb_path.name, rel))
    return missing


def count_pdf_pages(pdf_path):
    """Page count from the PDF page objects (enough for the PDFs that Chrome writes)."""
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", data))


def main():
    base = Path(__file__).resolve().parent
    notebooks = collect_files(base, NOTEBOOKS)
    files = notebooks + collect_files(base, DATA)

    print("Revisando las rutas que usa el código...")
    missing = check_references(base, notebooks, files)
    for name, rel in missing:
        print(f"  FALTA en el ZIP: {rel}  (lo usa {name})")
    for rel, why in NOT_NEEDED.items():
        print(f"  No se incluye {rel}: {why}")
    if missing:
        sys.exit("Añade esas rutas a DATA (o a NOT_NEEDED si no hacen falta) y repite.")

    zip_path = base / OUTPUT
    if zip_path.exists():
        try:
            zip_path.unlink()
        except OSError:
            sys.exit(f"No se puede reemplazar {zip_path.name}: ciérralo y repite.")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(base).as_posix())

    print(f"\n{len(files)} archivos en {zip_path.name} "
          f"({zip_path.stat().st_size / 1e6:.1f} MB):")
    for entry in NOTEBOOKS + DATA:
        n = sum(1 for p in files if p.relative_to(base).as_posix().startswith(entry))
        print(f"  {entry}" + (f"  ({n} archivos)" if n > 1 else ""))

    pdf_path = base / REPORT_PDF
    if pdf_path.is_file():
        pages = count_pdf_pages(pdf_path)
        status = "OK" if pages <= MAX_PDF_PAGES else f"SUPERA EL LÍMITE DE {MAX_PDF_PAGES}"
        print(f"\nInforme {pdf_path.name}: {pages} páginas -> {status}")
    else:
        print(f"\nNo se encontró el informe {REPORT_PDF}: genéralo con export_pdf.py")

    print(f"\nListo: {zip_path}")


if __name__ == "__main__":
    main()
