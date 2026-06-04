param(
    [string]$PythonExe = "",
    [string]$VenvDir = "venv_npu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$RequirementsFile = Join-Path $Root "requirements\requirements-npu-openvino.txt"
$VenvPath = Join-Path $Root $VenvDir
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Get-BasePython {
    param([string]$PreferredPython)

    if ($PreferredPython) {
        return (Resolve-Path $PreferredPython).Path
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    foreach ($cfgPath in @(
        (Join-Path $Root "venv\pyvenv.cfg"),
        (Join-Path $Root "venv_gpu\pyvenv.cfg")
    )) {
        if (-not (Test-Path $cfgPath)) {
            continue
        }

        $match = Select-String -Path $cfgPath -Pattern "^executable = (.+)$" | Select-Object -First 1
        if (-not $match) {
            continue
        }

        $candidate = $match.Matches[0].Groups[1].Value.Trim()
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Python 3.12 executable was not found. Pass -PythonExe explicitly."
}

$BasePython = Get-BasePython -PreferredPython $PythonExe

Write-Host "Using base Python: $BasePython"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating $VenvDir ..."
    & $BasePython -m venv $VenvPath
}

Write-Host "Installing NPU(OpenVINO) dependencies ..."
& $BasePython -m pip --python $VenvPython install --upgrade pip -r $RequirementsFile

Write-Host "Removing standard onnxruntime if it was installed as a transitive dependency ..."
try {
    & $VenvPython -m pip uninstall -y onnxruntime
} catch {
    Write-Host "onnxruntime uninstall was skipped: $($_.Exception.Message)"
}

Write-Host "Reinstalling OpenVINO runtime with app-compatible dependency versions ..."
& $VenvPython -m pip install --force-reinstall `
    onnxruntime-openvino==1.24.1 `
    openvino==2025.4.1 `
    numpy==1.26.4 `
    packaging==24.2 `
    protobuf==5.29.6

Get-ChildItem -LiteralPath (Join-Path $VenvPath "Lib\site-packages") -Filter "~nnxruntime*.dist-info" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force

$OpenVinoLibs = Join-Path $VenvPath "Lib\site-packages\openvino\libs"
if (Test-Path $OpenVinoLibs) {
    $env:PATH = "$OpenVinoLibs;$env:PATH"
}

Write-Host "Validating available providers ..."
& $VenvPython -c "import onnxruntime as ort; print('providers=', ort.get_available_providers())"

Write-Host "Validating app-side NPU resolution ..."
Push-Location $Root
try {
    & $VenvPython -c "from app.core import resolve_embedder_spec; from app.embedding_model import load_embedding_model_name; model=load_embedding_model_name(); print(resolve_embedder_spec('npu', model))"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "NPU environment is ready: $VenvPath"
