@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".lattice-env\Scripts\python.exe" (
  echo 尚未安装可视化环境。请参阅 LATTICE_STUDIO.md 的安装说明。
  pause
  exit /b 1
)
start "Lattice Studio" /b ".lattice-env\Scripts\pythonw.exe" lattice_studio.py
