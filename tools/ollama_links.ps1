# Expose Ollama's content-addressed GGUF blobs under readable .gguf names.
#
# Ollama stores model weights as blobs\sha256-<hash> files. llama-server and
# Kindi only need the GGUF header, so this script creates zero-copy hard
# links (or symbolic links across drives) named after the Ollama manifest,
# e.g. gemma4-26b.gguf, without downloading or duplicating the weights.
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File tools\ollama_links.ps1
#   powershell -ExecutionPolicy Bypass -File tools\ollama_links.ps1 -ModelsDir D:\ollamaModels -OutputDir T:\kindi-models\ollamaLinks

param(
    [string]$ModelsDir = $(if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } elseif (Test-Path -LiteralPath "D:\ollamaModels") { "D:\ollamaModels" } else { Join-Path $env:USERPROFILE ".ollama\models" }),
    [string]$OutputDir = "T:\kindi-models\ollamaLinks"
)

$ManifestsRoot = Join-Path $ModelsDir "manifests"
$BlobsRoot = Join-Path $ModelsDir "blobs"

if (-not (Test-Path -LiteralPath $ManifestsRoot -PathType Container)) {
    Write-Error "Ollama manifests directory not found at: $ManifestsRoot"
    exit 1
}

if (-not (Test-Path -LiteralPath $BlobsRoot -PathType Container)) {
    Write-Error "Ollama blobs directory not found at: $BlobsRoot"
    exit 1
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Check that both folders reside on the same NTFS volume (required for hard links).
$SourceDrive = (Get-Item -LiteralPath $BlobsRoot).PSDrive.Name
$TargetDrive = (Get-Item -LiteralPath $OutputDir).PSDrive.Name
if ($SourceDrive -ne $TargetDrive) {
    Write-Warning "Source ($SourceDrive`:) and output ($TargetDrive`:) are on different drives. Hard links require the same drive; falling back to symbolic links (may require admin/Developer Mode)."
}

$ManifestFiles = Get-ChildItem -Path $ManifestsRoot -File -Recurse
$LinkedCount = 0

foreach ($ManifestFile in $ManifestFiles) {
    try {
        $ManifestJson = Get-Content -LiteralPath $ManifestFile.FullName -Raw | ConvertFrom-Json
    } catch {
        Write-Warning "Skipping unreadable manifest: $($ManifestFile.FullName)"
        continue
    }

    # Find the main model weights layer (excludes mmproj, templates, licenses).
    $ModelLayers = @($ManifestJson.layers | Where-Object { $_.mediaType -eq "application/vnd.ollama.image.model" })
    if ($ModelLayers.Count -eq 0) { continue }

    # Pick the largest model layer in case a multimodal model has multiple model-typed layers.
    $PrimaryLayer = $ModelLayers | Sort-Object -Property size -Descending | Select-Object -First 1
    $BlobFileName = $PrimaryLayer.digest -replace ":", "-"
    $BlobPath = Join-Path $BlobsRoot $BlobFileName

    if (-not (Test-Path -LiteralPath $BlobPath)) {
        Write-Warning "Blob missing on disk for $($ManifestFile.Name): $BlobPath"
        continue
    }

    # Build a clean filename from the relative manifest path (e.g. library\gemma4\26b -> gemma4-26b.gguf).
    $RelPath = $ManifestFile.FullName.Substring($ManifestsRoot.Length).TrimStart('\', '/')
    $Parts = $RelPath -split '[\\/]'

    # Strip the standard registry prefix for cleaner names.
    if ($Parts.Count -ge 4 -and $Parts[0] -eq "registry.ollama.ai" -and $Parts[1] -eq "library") {
        $CleanName = ($Parts[2..($Parts.Count - 1)] -join "-")
    } elseif ($Parts.Count -ge 3 -and $Parts[0] -eq "registry.ollama.ai") {
        $CleanName = ($Parts[1..($Parts.Count - 1)] -join "-")
    } else {
        $CleanName = ($Parts -join "-")
    }

    # Sanitize invalid Windows filename characters.
    $CleanName = $CleanName -replace '[\\/:*?"<>|]', '-'
    $TargetLink = Join-Path $OutputDir "$CleanName.gguf"

    # Remove the old link if present.
    if (Test-Path -LiteralPath $TargetLink) {
        Remove-Item -LiteralPath $TargetLink -Force
    }

    # Create a hard link on the same drive or a symbolic link across drives.
    $LinkType = if ($SourceDrive -eq $TargetDrive) { "HardLink" } else { "SymbolicLink" }
    New-Item -ItemType $LinkType -Path $TargetLink -Target $BlobPath | Out-Null

    $SizeGB = [math]::Round($PrimaryLayer.size / 1GB, 2)
    Write-Host "[OK] $CleanName.gguf ($SizeGB GB) -> $BlobFileName" -ForegroundColor Green
    $LinkedCount++
}

Write-Host "`nDone! Created $LinkedCount .gguf links in $OutputDir (0 bytes additional disk space used)." -ForegroundColor Cyan
