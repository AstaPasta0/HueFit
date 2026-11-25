@echo off
title HueFit - Outfit Color Recommender
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" goto run_app

echo ============================================================
echo   HueFit - First-Time Local Preparation (Inside This Folder)
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 goto no_python

echo [1/4] Creating local Python environment...
python -m venv venv
if errorlevel 1 goto err_exit

echo [2/4] Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip

nvidia-smi >nul 2>&1
if errorlevel 1 goto install_cpu

echo [3/4] NVIDIA GPU detected - installing PyTorch CUDA 12.8...
"venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto err_exit
goto install_reqs

:install_cpu
echo [3/4] Installing CPU PyTorch...
"venv\Scripts\pip.exe" install torch torchvision
if errorlevel 1 goto err_exit

:install_reqs
echo [4/4] Installing HueFit dependencies...
"venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 goto err_exit
"venv\Scripts\python.exe" "scripts\download_artifacts.py" --if-missing

:run_app
"venv\Scripts\python.exe" "launch.py" %*
if errorlevel 1 goto err_exit
goto end

:no_python
echo [ERROR] Python 3.12 was not found on PATH.
echo Install Python 3.12 from https://www.python.org/downloads/ (check "Add Python to PATH").
pause
goto end

:err_exit
echo.
echo [HueFit] Exited with an error. Review the message above.
pause

:end