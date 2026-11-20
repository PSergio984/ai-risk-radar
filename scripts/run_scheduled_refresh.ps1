<#
.SYNOPSIS
    Scheduled market data refresh and real-time risk inference runner for AI Investment Risk Radar.
.DESCRIPTION
    Runs on Windows Task Scheduler or on-demand to execute the latest market inference pipeline.
    Ensures cold-bootstrap checks, sets required environment variables, redirects logs,
    and handles failures gracefully without state corruption.
.PARAMETER Config
    Path to configuration JSON file (default: config.json).
.PARAMETER DataMode
    Data mode: refresh or cache (default: refresh).
#>
[CmdletBinding()]
param (
    [string]$Config = "config.json",
    [string]$DataMode = "refresh"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path "$ScriptDir\..").Path
Set-Location -Path $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogFile = Join-Path $LogDir "scheduled_refresh.log"

function Write-Log {
    param([string]$Level, [string]$Message)
    $Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $LogLine = "[$Timestamp] [$Level] $Message"
    Write-Output $LogLine
    Add-Content -Path $LogFile -Value $LogLine
}

Write-Log "INFO" "================================================================="
Write-Log "INFO" "AI Risk Radar scheduled refresh started in $RepoRoot"
Write-Log "INFO" "Config: $Config | Data Mode: $DataMode"

# 1. Cold Bootstrap: Find Python
$PythonBin = ""
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
} else {
    $Command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Command) {
        $PythonBin = $Command.Source
    } else {
        Write-Log "ERROR" "Python executable not found in .venv or PATH."
        exit 127
    }
}
Write-Log "INFO" "Using Python binary: $PythonBin"

# Verify prerequisites
if (-not (Test-Path $Config)) {
    Write-Log "ERROR" "Config file '$Config' not found."
    exit 2
}
$ReleaseFile = Join-Path $RepoRoot "artifacts\release.json"
if (-not (Test-Path $ReleaseFile)) {
    Write-Log "ERROR" "Release artifact 'artifacts/release.json' not found."
    exit 3
}

# 2. Environment Variables
$env:PYTHONPATH = "$RepoRoot;" + $env:PYTHONPATH
$env:PYTHONUNBUFFERED = "1"
$env:AI_RISK_RADAR_LOG_DIR = $LogDir

# 3. Execution
$StartTime = Get-Date
Write-Log "INFO" "Executing: & '$PythonBin' -m ai_risk_radar latest --config '$Config' --data-mode '$DataMode'"

try {
    & $PythonBin -m ai_risk_radar latest --config $Config --data-mode $DataMode *>> $LogFile
    $ExitCode = $LASTEXITCODE
} catch {
    Write-Log "ERROR" "Process invocation error: $_"
    $ExitCode = 1
}

$EndTime = Get-Date
$Duration = [math]::Round(($EndTime - $StartTime).TotalSeconds, 2)

if ($ExitCode -eq 0) {
    Write-Log "INFO" "Scheduled refresh completed successfully in ${Duration}s."
    Write-Log "INFO" "Updated artifacts/latest_report.json and data/raw/latest/manifest.json."
    Write-Log "INFO" "================================================================="
    exit 0
} else {
    Write-Log "ERROR" "Scheduled refresh FAILED with exit code $ExitCode after ${Duration}s."
    Write-Log "ERROR" "Prior snapshot and release artifacts remain preserved."
    Write-Log "ERROR" "================================================================="
    exit $ExitCode
}
