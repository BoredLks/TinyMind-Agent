# PyInstaller spec for SuperAgent (onedir Windows app).
# Build:  .venv\Scripts\python.exe -m PyInstaller --clean --noconfirm \
#           --workpath build\_pyiwork --distpath dist build\superagent.spec
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
BACKEND = os.path.join(ROOT, "backend")

# Bundle shipped read-only skills + the built frontend. User-extensible skills
# and tools live beside SuperAgent.exe in onedir builds.
datas = [
    # Skills are now copied to dist/SuperAgent/skills (user-writable) instead of bundled inside _internal.
    # Users can freely add/remove skills there.
    (os.path.join(BACKEND, "app", "personas_builtin"), "personas_builtin"),
    (os.path.join(ROOT, "frontend", "dist"), "frontend_dist"),
]
binaries = []
hiddenimports = collect_submodules("uvicorn")
hiddenimports += collect_submodules("webview")

for pkg in (
    "webview",
    "pythonnet",
    "clr_loader",
    "proxy_tools",
    "bottle",
    "openai",
    "keyring",
    "keyring.backends",
    "tenacity",
    "fastapi",
    "starlette",
    # Document processing tools
    "openpyxl",
    "fpdf",
    "docx",
    "pypdf",
    "lxml",
    "PIL",
    "defusedxml",
    "fonttools",
    "et_xmlfile",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

icon = os.path.join(SPECPATH, "icon.ico")
icon = icon if os.path.exists(icon) else None

a = Analysis(
    [os.path.join(BACKEND, "app", "desktop", "launch.py")],
    pathex=[BACKEND],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    name="SuperAgent",
    debug=False,
    exclude_binaries=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SuperAgent",
)
