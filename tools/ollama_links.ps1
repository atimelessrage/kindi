# Thin wrapper over tools/ollama_links.py: the logic lives in one file so Windows and Linux
# cannot drift. It reads $env:OLLAMA_MODELS when -m is not given, links each Ollama model under
# a readable .gguf name, and uses a symlink when allowed, else a same-volume hard link.
#
#   powershell -ExecutionPolicy Bypass -File tools\ollama_links.ps1 --dry-run
#   powershell -ExecutionPolicy Bypass -File tools\ollama_links.ps1 -m <ollama-models-dir> -o <links-dir>
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& python (Join-Path $ScriptDir "ollama_links.py") @args
exit $LASTEXITCODE
