@echo off
setlocal

echo ============================================
echo ARCUS-X Convergence Study
echo Model: GPT-5-mini
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

set MODEL=gpt-5-mini
set API_BASE=https://arcusx-resource.services.ai.azure.com/openai/v1

for %%P in (%PROBES%) do (

    echo.
    echo ============================================
    echo Running GPT-5-mini
    echo Probes: %%P
    echo ============================================

    python quickstart_api.py ^
        --api-key %GPT_KEY% ^
        --model %MODEL% ^
        --api-base %API_BASE% ^
        --n-probes %%P ^
        --gravity-levels %GRAVITIES% ^
        --master-seeds %SEEDS%

    echo.
    echo Finished GPT-5-mini %%P probes.
)

echo.
echo ============================================
echo GPT-5-mini convergence study complete.
echo ============================================

pause