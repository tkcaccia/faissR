param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$FaissHome,
    [Parameter(Mandatory=$true)][string]$DependencyLibrary
)
$ErrorActionPreference = 'Stop'
$old = $env:FAISSR_NUMERICAL_LIBS
try {
    $env:FAISSR_NUMERICAL_LIBS = '-lfaissr_deliberately_missing'
    # Launch a child PowerShell because run.ps1 exits with the check status.
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\run.ps1" `
        -Archive $Archive -Output "$Output\invalid-numerical" -Profile functional `
        -FaissHome $FaissHome -DependencyLibrary $DependencyLibrary -FaissSource system
    if ($LASTEXITCODE -ne 1) { throw 'Invalid numerical libraries were not rejected' }
    $text = Get-Content "$Output\invalid-numerical\install.log" -Raw
    if ($text -notmatch 'requested installed FAISS/numerical libraries are unusable') {
        throw 'Installation failed for an unexpected reason'
    }
    $env:FAISSR_NUMERICAL_LIBS = ''
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\run.ps1" `
        -Archive $Archive -Output "$Output\missing-faiss" -Profile functional `
        -FaissHome "$Output\absent-faiss-prefix" -DependencyLibrary $DependencyLibrary `
        -FaissSource system
    if ($LASTEXITCODE -ne 1) { throw 'Missing FAISS was not rejected' }
    $text = Get-Content "$Output\missing-faiss\install.log" -Raw
    if ($text -notmatch 'No Rtools-compatible static FAISS was found') {
        throw 'Missing-FAISS installation failed for an unexpected reason'
    }
    'Both negative linkage tests passed' | Out-File "$Output\negative-tests.txt"
    Write-Output 'Both negative linkage tests passed'
} finally {
    $env:FAISSR_NUMERICAL_LIBS = $old
}
