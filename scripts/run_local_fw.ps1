# Test ESCALATE mode locally (local-first + Fireworks safety net).
# The API key is passed as a parameter and never written to disk or the image.
#
# Usage:
#   .\scripts\run_local_fw.ps1 -ImageTag uvlb/amd-router:latest `
#       -ApiKey fw_xxx -Models "accounts/fireworks/models/llama-v3p1-8b-instruct"
param(
    [Parameter(Mandatory = $true)][string]$ImageTag,
    [Parameter(Mandatory = $true)][string]$ApiKey,
    [Parameter(Mandatory = $true)][string]$Models,
    [string]$BaseUrl = "https://api.fireworks.ai/inference/v1"
)
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."

New-Item -ItemType Directory -Force -Path .\tests\output | Out-Null

docker run --rm `
    -v "${PWD}\tests\sample:/input:ro" `
    -v "${PWD}\tests\output:/output" `
    -e FIREWORKS_MODE=escalate `
    -e FIREWORKS_API_KEY=$ApiKey `
    -e FIREWORKS_BASE_URL=$BaseUrl `
    -e ALLOWED_MODELS=$Models `
    $ImageTag

Write-Host "`n--- results.json ---"
Get-Content .\tests\output\results.json
Write-Host "`n--- inference_log.json (routes + tokens) ---"
Get-Content .\tests\output\inference_log.json
Pop-Location
