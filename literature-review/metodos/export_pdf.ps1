# Export research_map_methods.ipynb to PDF with only text, equations and figures:
# code cells tagged "hide-input" are removed. The PDF is not tracked in git (.gitignore).
# Requires playwright + chromium in the .venv:
#   pip install playwright; python -m playwright install chromium
# Usage (from any folder): powershell -ExecutionPolicy Bypass -File literature-review\metodos\export_pdf.ps1
#
# Equivalent to:
#   jupyter nbconvert --to webpdf --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags hide-input research_map_methods.ipynb
# On Windows, nbconvert forces WindowsSelectorEventLoopPolicy, which cannot start the
# Chromium subprocess used by webpdf (NotImplementedError). The wrapper below keeps the
# default Proactor event loop and then runs nbconvert with the same arguments.

Set-Location $PSScriptRoot
$python = (Resolve-Path "$PSScriptRoot\..\..\.venv\Scripts\python.exe").Path

$wrapper = "import asyncio, sys; asyncio.set_event_loop_policy = lambda policy: None; from nbconvert.nbconvertapp import main; sys.exit(main())"

& $python -c $wrapper --to webpdf --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags hide-input research_map_methods.ipynb
