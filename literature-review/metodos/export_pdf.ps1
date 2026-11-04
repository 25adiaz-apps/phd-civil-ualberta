# Export the two research map methods notebooks to PDF with only text, equations and figures:
#   research_map_methods.ipynb        -> research_map_methods.pdf
#   research_map_methods_simple.ipynb -> research_map_methods_simple.pdf
# Code cells tagged "hide-input" are removed. The PDFs are tracked in git: re-export them
# after each notebook change.
# Requires playwright + chromium in the .venv:
#   pip install playwright; python -m playwright install chromium
# Usage (from any folder): powershell -ExecutionPolicy Bypass -File literature-review\metodos\export_pdf.ps1
#
# Equivalent to, for each notebook:
#   jupyter nbconvert --to webpdf --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags hide-input <notebook>.ipynb
# On Windows, nbconvert forces WindowsSelectorEventLoopPolicy, which cannot start the
# Chromium subprocess used by webpdf (NotImplementedError). The wrapper below keeps the
# default Proactor event loop and then runs nbconvert with the same arguments.

Set-Location $PSScriptRoot
$python = (Resolve-Path "$PSScriptRoot\..\..\.venv\Scripts\python.exe").Path

$wrapper = "import asyncio, sys; asyncio.set_event_loop_policy = lambda policy: None; from nbconvert.nbconvertapp import main; sys.exit(main())"

$notebooks = @("research_map_methods.ipynb", "research_map_methods_simple.ipynb")
foreach ($nb in $notebooks) {
    & $python -c $wrapper --to webpdf --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags hide-input $nb
}
