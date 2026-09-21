@echo off
setlocal

echo ============================================
echo ARCUS-X Convergence Study
echo Model: Poolside Laguna XS 2.1
echo ============================================

REM =========================
REM Shared settings
REM =========================

set SEEDS=42 1337
set GRAVITIES=0.5,3.0,15.0
set PROBES=1 3

REM =========================
REM Model Configuration
REM =========================

set MODEL=poolside/laguna-xs-2.1
set API_BASE=https://ai.hackclub.com/proxy/v1

for %%P in (%PROBES%) do (

    echo.
    echo ============================================
    echo Running Laguna XS 2.1
    echo Probes: %%P
    echo ============================================

    python quickstart_api.py ^
        --api-key %GEMINI_KEY% ^
        --model %MODEL% ^
        --api-base %API_BASE% ^
        --n-probes %%P ^
        --gravity-levels %GRAVITIES% ^
        --master-seeds %SEEDS%

    echo.
    echo Finished Laguna %%P probes.
)

echo.
echo ============================================
echo Laguna convergence study complete.
echo ============================================

pause