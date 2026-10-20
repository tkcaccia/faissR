param(
    [string]$Root = "$env:USERPROFILE\r-package-test-lab",
    [string]$RHome = "$Root\R-4.6.1"
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# A private TeX tree: do not update or replace a user's TinyTeX installation.
$texmf = "$Root\check-texmf"
$archive = "$Root\courier-texlive2024.tar.xz"
$url = 'https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2024/tlnet-final/archive/courier.tar.xz'
$expected = '1918ee2c702257d9f0a20975ec7674c727a335be3f5f1ac45c3e882155834afd'
New-Item -ItemType Directory -Force $texmf | Out-Null
if (!(Test-Path $archive)) { Invoke-WebRequest $url -OutFile $archive }
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $expected) {
    throw 'Courier source archive checksum mismatch'
}
& "$RHome\bin\Rscript.exe" --vanilla "$PSScriptRoot\prepare-check-fonts.R" $archive $texmf 'fonts/tfm/adobe/courier/pcrr8t.tfm'
if ($LASTEXITCODE -ne 0) { throw 'Font archive extraction failed' }
$indexArchive = "$Root\makeindex-texlive2024-windows.tar.xz"
$indexUrl = 'https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2024/tlnet-final/archive/makeindex.windows.tar.xz'
$indexHash = 'e3d32dff8233529c2c5da8ba87a2b39e5339b546aaf195539408b4a38612cf9b'
if (!(Test-Path $indexArchive)) { Invoke-WebRequest $indexUrl -OutFile $indexArchive }
if ((Get-FileHash $indexArchive -Algorithm SHA256).Hash -ne $indexHash) {
    throw 'MakeIndex archive checksum mismatch'
}
& "$RHome\bin\Rscript.exe" --vanilla "$PSScriptRoot\prepare-check-fonts.R" $indexArchive $texmf 'bin/windows/makeindex.exe'
if ($LASTEXITCODE -ne 0) { throw 'MakeIndex extraction failed' }
$env:PATH = "$texmf\bin\windows;$env:PATH"
$env:TEXMFHOME = $texmf
& kpsewhich pcrr8t.tfm
if ($LASTEXITCODE -ne 0) { throw 'Private Courier metrics are not visible' }
Write-Output "Private check fonts ready: $texmf"
$fixture = "$Root\check-makeindex.idx"
Set-Content -LiteralPath $fixture -Encoding Ascii -Value '\indexentry{sample}{1}'
Push-Location $Root
try { & makeindex -q 'check-makeindex.idx' }
finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw 'Private MakeIndex is not usable' }
if (!(Test-Path "$Root\check-makeindex.ind")) { throw 'Index output is missing' }
