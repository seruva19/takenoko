@echo off
REM Windows installation script for Takenoko
REM This script installs all dependencies using uv package manager

echo ========================================
echo Takenoko Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
python --version

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo uv package manager not found. Installing uv...
    echo.
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Failed to install uv
        echo Please install uv manually from https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    echo uv installed successfully!
    echo.
    echo Please restart your terminal and run this script again.
    pause
    exit /b 0
)

echo uv found. Version:
uv --version
echo.

REM Check CUDA availability
echo Checking CUDA availability...
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version --format=csv >nul 2>&1
if errorlevel 1 (
    echo WARNING: NVIDIA GPU not detected or CUDA not installed
    echo Will install CPU-only version
    set CUDA_EXTRA=""
) else (
    echo NVIDIA GPU detected. Installing CUDA 12.6 version...
    set CUDA_EXTRA="[cu126]"
    echo.
    echo GPU summary ^(model, memory, driver^):
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version --format=csv
)

echo.
echo Setting up virtual environment...
if exist ".venv" (
    echo Virtual environment already exists at .venv
) else (
    echo Creating virtual environment...
    uv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment!
        echo Please check the error messages above and try again.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment!
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)

echo Virtual environment activated successfully!
echo.
echo Installing Takenoko dependencies...
echo.

REM Install the project in editable mode with appropriate CUDA extras
uv pip install -e .%CUDA_EXTRA%

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)

echo.
echo Checking installed torch version...
python -c "import torch; print('Torch version:', torch.__version__)" || echo Torch is not installed or import failed.

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Virtual environment is now ready at: .venv
echo.
echo Running installation test...
python tools/verify_installation.py
echo.
echo You can now run the trainer using:
echo   run_trainer.bat
echo.
echo Or manually with:
echo   python src/takenoko.py configs/your_config.toml
echo.
echo To activate the virtual environment manually:
echo   .venv\Scripts\activate.bat
echo.
pause 