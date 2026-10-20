param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$RHome = "$env:USERPROFILE\r-package-test-lab\R-4.6.1",
    [string]$Rtools = 'C:\rtools45',
    [ValidateSet('functional','diagnostic')][string]$Profile = 'functional',
    [string]$FaissHome = '',
    [string]$DependencyLibrary = '',
    [ValidateSet('auto','system','bundled','none')][string]$FaissSource = 'auto'
)
$ErrorActionPreference = 'Stop'
$env:PATH = "$RHome\bin;$Rtools\usr\bin;$Rtools\x86_64-w64-mingw32.static.posix\bin;$env:PATH"
$pandoc = "$env:USERPROFILE\r-package-test-lab\pandoc-3.6.4"
if (Test-Path "$pandoc\pandoc.exe") {
    $env:RSTUDIO_PANDOC = $pandoc
    $env:PATH = "$pandoc;$env:PATH"
}
$texmf = "$env:USERPROFILE\r-package-test-lab\check-texmf"
if (Test-Path $texmf) { $env:TEXMFHOME = $texmf }
if (Test-Path "$texmf\bin\windows") {
    $env:PATH = "$texmf\bin\windows;$env:PATH"
}
if ($DependencyLibrary) { $env:R_LIBS = "$DependencyLibrary;$env:R_LIBS" }
$env:FAISS_HOME = $FaissHome
$env:CONDA_PREFIX = ''
$env:FAISSR_USE_CUDA = '0'
$env:FAISSR_USE_CUVS = '0'
$env:FAISSR_REQUIRE_CUDA = '0'
$env:FAISSR_REQUIRE_CUVS = '0'
$env:FAISSR_REQUIRE_FAISS = $(if ($Profile -eq 'functional') { '1' } else { '0' })
$env:FAISSR_FAISS_SOURCE = $(if ($Profile -eq 'diagnostic') { 'none' } else { $FaissSource })
$env:MAKEFLAGS = '-j2'
$here = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Get-FileHash -Algorithm SHA256 $Archive | Format-List | Out-File "$Output\inputs.sha256.txt"
& "$RHome\bin\Rscript.exe" "$here\run.R" $Archive $Output $Profile "$here\faissR-smoke.R"
exit $LASTEXITCODE
