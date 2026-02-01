@echo off
setlocal

rem --- 1) Resolve project folder (folder where this .bat lives) ---
set "PROJ=%~dp0"

rem --- 2) Point to embedded Python inside the project ---
set "PYDIR=%PROJ%python"
set "PYEXE=%PYDIR%\python.exe"

echo Project folder: %PROJ%
echo Embedded Python: %PYEXE%

if not exist "%PYEXE%" (
    echo.
    echo ERROR: Could not find embedded Python.
    echo Expected here:
    echo     %PYEXE%
    echo.
    echo Make sure you unzipped python-3.10.11-embed-amd64.zip into:
    echo     %PYDIR%
    pause
    exit /b 1
)

rem --- 3) Point Python at its home and libraries ---
set "PYTHONHOME=%PYDIR%"
rem Adjust PYTHONPATH for embedded Python with python311.zip
set "PYTHONPATH=%PYDIR%\python311.zip;%PYDIR%"

rem --- 4) Use venv Python instead of embedded Python directly ---
set "VENV_PYEXE=%PROJ%.venv_poomsae\Scripts\python.exe"

rem --- 5) Go to project folder and run the app ---
cd /d "%PROJ%"

"%VENV_PYEXE%" app_gradio_poomsae14.py

echo.
echo (If the browser did not open automatically, open this in Chrome/Edge:)
echo     http://127.0.0.1:7860/
echo.
pause
exit /b 0
