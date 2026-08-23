# Build SuperAgent into a Windows onedir app.
# Run from the repo root:  .\build\build.ps1
$ErrorActionPreference = "Stop"
$env:PATH = "C:\Program Files\nodejs;" + $env:PATH

Write-Host "[1/3] Building frontend (Vite)..."
npm --prefix frontend install
npm --prefix frontend run build

Write-Host "[2/3] Installing backend deps + PyInstaller..."
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pyinstaller

Write-Host "[3/3] Packaging onedir app..."
$outputDir = Join-Path (Get-Location) "dist\SuperAgent"
$outputExe = Join-Path $outputDir "SuperAgent.exe"
$legacyOnefile = Join-Path (Get-Location) "dist\SuperAgent.exe"
$running = Get-Process -Name SuperAgent -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $outputExe -or $_.Path -eq $legacyOnefile }
if ($running) {
    throw "SuperAgent is still running. Close it before rebuilding."
}
if (Test-Path $outputDir) {
    Remove-Item -LiteralPath $outputDir -Recurse -Force
}
if (Test-Path $legacyOnefile) {
    Remove-Item -LiteralPath $legacyOnefile -Force
}
.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm `
    --workpath build\_pyiwork --distpath dist build\superagent.spec

$skillsDir = Join-Path $outputDir "skills"
$toolsDir = Join-Path $outputDir "tools"
New-Item -ItemType Directory -Force -Path $skillsDir, $toolsDir | Out-Null
Set-Content -LiteralPath (Join-Path $skillsDir "README.md") -Encoding UTF8 -Value @"
# SuperAgent skills

Put skill folders here. Each skill folder should contain a SKILL.md file with YAML frontmatter:

---
name: my-skill
description: When to use this skill.
---
Skill instructions...
"@
Set-Content -LiteralPath (Join-Path $toolsDir "README.md") -Encoding UTF8 -Value @"
# SuperAgent tool plugins

Put trusted Python .py tool plugins here.

A plugin can export register(registry) or TOOLS = [Tool(), ...].
Plugins run as local Python code, so only install plugins you trust.
"@

# Copy builtin skills to user-writable skills directory (so users can add/remove freely)
$builtinSkillsDir = Join-Path (Get-Location) "backend\app\skills_builtin"
if (Test-Path $builtinSkillsDir) {
    $skillDirs = Get-ChildItem -Path $builtinSkillsDir -Directory
    foreach ($dir in $skillDirs) {
        $src = $dir.FullName
        $dst = Join-Path $skillsDir $dir.Name
        if (Test-Path $dst) {
            Remove-Item -LiteralPath $dst -Recurse -Force
        }
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
        Write-Host "  Copied skill: $($dir.Name)"
    }
}

# Copy SonettoHere-migrated tool plugins from repo root tools/ directory
$repoToolsDir = Join-Path (Get-Location) "tools"
if (Test-Path $repoToolsDir) {
    $pluginFiles = Get-ChildItem -Path $repoToolsDir -Filter "*.py" -File
    foreach ($file in $pluginFiles) {
        Copy-Item -LiteralPath $file.FullName -Destination $toolsDir -Force
        Write-Host "  Copied tool plugin: $($file.Name)"
    }
}

Write-Host "Done. Output: dist\SuperAgent\SuperAgent.exe"
Write-Host "Extension dirs: dist\SuperAgent\skills and dist\SuperAgent\tools"
