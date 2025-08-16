@echo off
REM Set number of CPU threads per process for PyTorch/NumPy/Accelerate
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
REM Cursor: Windows script to run Takenoko with unified operations menu
REM This script detects all config files and presents them as a menu

setlocal enabledelayedexpansion

REM Parse optional flags: --gpus N and/or --ask-gpus (prompt only when requested)
set ASK_GPUS=0
set GPUS=
set NEXT_IS_GPUS=0
for %%A in (%*) do (
    if /I "%%~A"=="--ask-gpus" set ASK_GPUS=1
    if "!NEXT_IS_GPUS!"=="1" (
        set GPUS=%%~A
        set NEXT_IS_GPUS=0
    ) else (
        if /I "%%~A"=="--gpus" set NEXT_IS_GPUS=1
    )
)

echo ========================================
echo Takenoko Unified Operations Menu
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at .venv\Scripts\activate.bat
    echo Continuing without virtual environment...
)

REM Check if configs directory exists
if not exist "configs" (
    echo ERROR: configs directory not found!
    echo Please make sure you're running this script from the project root.
    pause
    exit /b 1
)

:scan_configs
REM Find all TOML config files
set config_count=0
set config_list=

echo Scanning for configuration files...
echo.

echo 0. Quit
for %%f in (configs\*.toml) do (
    if exist "%%f" (
        set /a config_count+=1
        set config_list=!config_list! %%f
        echo !config_count!. %%~nxf
    )
)

if %config_count%==0 (
    echo No configuration files found in configs directory!
    echo Please add some .toml files to the configs folder.
    pause
    exit /b 1
)

echo.
echo Found %config_count% configuration file(s)
echo.

REM Ask user to select a config
:select_config
set /p choice="Enter the number of the config to run (0-%config_count%): "

REM Validate input: must be non-empty digits only
echo %choice%| findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Invalid input. Please enter a number.
    goto select_config
)
set /a choice_num=%choice%

if %choice_num%==0 (
    echo Exiting...
    exit /b 0
)

if %choice_num% lss 0 (
    echo Invalid choice. Please enter a number between 0 and %config_count%.
    goto select_config
)

if %choice_num% gtr %config_count% (
    echo Invalid choice. Please enter a number between 0 and %config_count%.
    goto select_config
)

REM Get the selected config file
set current_count=0
for %%f in (%config_list%) do (
    set /a current_count+=1
    if !current_count!==%choice_num% (
        set selected_config=%%f
        goto run_unified_trainer
    )
)

:run_unified_trainer
echo.
echo ========================================
echo Starting Takenoko with: %selected_config%
echo ========================================
echo.

REM Prompt for GPUs only if --ask-gpus was provided and --gpus not supplied
if "%ASK_GPUS%"=="1" if "%GPUS%"=="" (
    set /p GPUS="Enter number of GPUs to use (press Enter for 1): "
)

if "%GPUS%"=="" set GPUS=1

REM Validate GPUS is a positive integer
echo %GPUS%| findstr /R "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo Invalid GPU count. Defaulting to 1.
    set GPUS=1
)

REM Build command based on GPU count
if %GPUS% GTR 1 (
    echo Running distributed with torchrun on %GPUS% GPUs
    echo Command: torchrun --nproc_per_node=%GPUS% src/takenoko.py "%selected_config%" --non-interactive --train
    torchrun --nproc_per_node=%GPUS% src/takenoko.py "%selected_config%" --non-interactive --train
    goto post_run
)

echo Running single process
echo Command: python src/takenoko.py "%selected_config%"
python src/takenoko.py "%selected_config%"

:post_run

REM If the trainer exits with code 100, return to config selection
if %errorlevel%==100 (
    echo.
    echo Returning to config selection...
    echo.
    goto scan_configs
)

if errorlevel 1 (
    echo.
    echo ERROR: Unified trainer failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Unified trainer completed!
echo ========================================
echo.
pause 