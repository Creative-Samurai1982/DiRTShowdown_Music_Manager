@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Showdown Music Manager - Versioned Builder

REM ===============================
REM Configuration
REM ===============================
set "APP_BASE_NAME=ShowdownMusicManager"
set "APP_DIR=app"

cd /d "%~dp0"

echo.
echo ============================================
echo        Showdown Music Manager Builder
echo ============================================
echo.

REM ===============================
REM List available Python files
REM ===============================
echo Available Python files in %APP_DIR%:
echo.

set /a count=0
for %%f in (%APP_DIR%\*.py) do (
    set /a count+=1
    set "file!count!=%%f"
    echo   !count!^) %%~nxf
)

if %count%==0 (
    echo ERROR: No .py files found in %APP_DIR%
    pause
    exit /b 1
)

echo.
set /p choice=Enter number of script to build: 

if not defined file%choice% (
    echo Invalid selection.
    pause
    exit /b 1
)

set "ENTRY_SCRIPT=!file%choice%!"
echo.
echo Selected:
echo   %ENTRY_SCRIPT%
echo.

REM ===============================
REM Extract APP_VERSION from script
REM ===============================
for /f "tokens=2 delims== " %%a in ('findstr /i "APP_VERSION" "%ENTRY_SCRIPT%"') do (
    set "APP_VERSION_RAW=%%a"
    goto :version_found
)

:version_found
set "APP_VERSION=%APP_VERSION_RAW:"=%"

if not defined APP_VERSION (
    set "APP_VERSION=unknown"
)

echo Detected APP_VERSION: %APP_VERSION%
echo.

REM ===============================
REM Check bin folder
REM ===============================
if not exist "bin\" (
  echo ERROR: bin\ folder not found in project root.
  pause
  exit /b 1
)

REM ===============================
REM Install / Upgrade PyInstaller
REM ===============================
echo [1/6] Installing / upgrading PyInstaller...
py -m pip install --upgrade pip
py -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo ERROR: Could not install/upgrade PyInstaller.
  pause
  exit /b 1
)

REM ===============================
REM Clean old builds
REM ===============================
echo.
echo [2/6] Cleaning old builds...
if exist "build\" rmdir /s /q "build"
if exist "dist\" rmdir /s /q "dist"
if exist "%APP_BASE_NAME%.spec" del /q "%APP_BASE_NAME%.spec"

REM ===============================
REM Build EXE
REM ===============================
set "FINAL_EXE_NAME=%APP_BASE_NAME%_v%APP_VERSION%"

echo.
echo [3/6] Building EXE...
py -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name "%FINAL_EXE_NAME%" ^
  --onefile ^
  --console ^
  --collect-all rich ^
  "%ENTRY_SCRIPT%"

if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  pause
  exit /b 1
)

REM ===============================
REM Create versioned release folder
REM ===============================
echo.
echo [4/6] Creating release folder...
set "RELEASE_DIR=dist\%FINAL_EXE_NAME%-RELEASE"
mkdir "%RELEASE_DIR%" >nul

copy /Y "dist\%FINAL_EXE_NAME%.exe" "%RELEASE_DIR%\" >nul
xcopy /E /I /Y "bin\*" "%RELEASE_DIR%\bin\" >nul

REM Copy docs if present
if exist "README.md" copy /Y "README.md" "%RELEASE_DIR%\" >nul
if exist "LICENSE.txt" copy /Y "LICENSE.txt" "%RELEASE_DIR%\" >nul
if exist "THIRD_PARTY_NOTICES.md" copy /Y "THIRD_PARTY_NOTICES.md" "%RELEASE_DIR%\" >nul
if exist "MANUAL.pdf" copy /Y "MANUAL.pdf" "%RELEASE_DIR%\" >nul

REM ===============================
REM Finished
REM ===============================
echo.
echo ============================================
echo Build Complete
echo ============================================
echo Script Built: %ENTRY_SCRIPT%
echo Version:      %APP_VERSION%
echo Output Folder: %RELEASE_DIR%
echo EXE Name:     %FINAL_EXE_NAME%.exe
echo ============================================
echo.
pause
endlocal