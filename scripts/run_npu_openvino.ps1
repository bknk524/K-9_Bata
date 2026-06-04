param(
    [string]$VenvDir = "venv_npu",
    [switch]$PrepareSettings,
    [switch]$Launch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path (Join-Path $Root $VenvDir) "Scripts\python.exe"
$SettingsPath = Join-Path $Root "data\app_settings.json"
$AppPath = Join-Path $Root "app\ui_app.py"

function ConvertTo-HashtableCompat {
    param([Parameter(ValueFromPipeline = $true)]$InputObject)

    if ($null -eq $InputObject) {
        return $null
    }

    if ($InputObject -is [System.Collections.IDictionary]) {
        $hash = @{}
        foreach ($key in $InputObject.Keys) {
            $hash[$key] = ConvertTo-HashtableCompat $InputObject[$key]
        }

        return $hash
    }

    if ($InputObject -is [System.Collections.IEnumerable] -and $InputObject -isnot [string]) {
        $items = @()
        foreach ($item in $InputObject) {
            $items += ,(ConvertTo-HashtableCompat $item)
        }

        return $items
    }

    if ($InputObject -is [pscustomobject]) {
        $hash = @{}
        foreach ($property in $InputObject.PSObject.Properties) {
            $hash[$property.Name] = ConvertTo-HashtableCompat $property.Value
        }

        return $hash
    }

    return $InputObject
}

function Write-Utf8NoBomFile {
    param(
        [string]$Path,
        [string]$Content
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

if (-not (Test-Path $VenvPython)) {
    throw "$VenvPython was not found. Run scripts\setup_npu_openvino.ps1 first."
}

$OpenVinoLibs = Join-Path (Join-Path $Root $VenvDir) "Lib\site-packages\openvino\libs"
if (Test-Path $OpenVinoLibs) {
    $env:PATH = "$OpenVinoLibs;$env:PATH"
}

if ($PrepareSettings) {
    $settingsDir = Split-Path -Parent $SettingsPath
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Path $settingsDir | Out-Null
    }

    $settings = @{}
    if (Test-Path $SettingsPath) {
        $settings = Get-Content $SettingsPath -Raw | ConvertFrom-Json | ConvertTo-HashtableCompat
    }

    $settings["device"] = "npu"
    $settingsJson = $settings | ConvertTo-Json -Depth 8
    Write-Utf8NoBomFile -Path $SettingsPath -Content $settingsJson
    Write-Host "Set data/app_settings.json device=npu"
}

$streamlitCommand = "`"$VenvPython`" -m streamlit run `"$AppPath`""
Write-Host $streamlitCommand

if ($Launch) {
    Push-Location $Root
    try {
        & $VenvPython -m streamlit run $AppPath
    } finally {
        Pop-Location
    }
}
