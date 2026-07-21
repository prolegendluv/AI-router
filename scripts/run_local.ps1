# Run the built image exactly like the judging harness would.
# Usage:  .\scripts\run_local.ps1  amd-router:latest
param(
    [Parameter(Mandatory = $true)][string]$ImageTag
)
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."

New-Item -ItemType Directory -Force -Path .\tests\output | Out-Null

# Mounts tests/sample as /input and tests/output as /output.
# Pass Fireworks env vars to exercise the safety net; omit them (or set
# FIREWORKS_MODE=never) to force pure-local, zero-token behaviour.
docker run --rm `
    -v "${PWD}\tests\sample:/input:ro" `
    -v "${PWD}\tests\output:/output" `
    -e FIREWORKS_MODE=never `
    $ImageTag

Write-Host "`nResults:"
Get-Content .\tests\output\results.json
Pop-Location
