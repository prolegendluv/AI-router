# Build a linux/amd64 image and push it to a public registry.
# Usage:  .\scripts\build_and_push.ps1  ghcr.io/<user>/amd-router:latest
param(
    [Parameter(Mandatory = $true)][string]$ImageTag
)

$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."

if (-not (Get-ChildItem -Path .\models -Filter *.gguf -ErrorAction SilentlyContinue)) {
    Write-Error "No *.gguf in .\models - copy your Gemma E4B GGUF there first."
}

# buildx builds for the judging platform (linux/amd64) and pushes in one step.
docker buildx build --platform linux/amd64 --tag $ImageTag --push .

Write-Host "Pushed $ImageTag. Confirm it is PUBLIC in your registry settings."
Pop-Location
