@echo off
setlocal

echo ============================================
echo ARCUS-X Probe Convergence Study
echo ============================================

REM =========================
REM Shared settings
REM =========================

set SEEDS=42 1337
set GRAVITIES=0.5,3.0,15.0
set PROBES=1 3


REM =========================
REM Laguna XS 2.1
REM Hack Club Shared API
REM =========================

set LAGUNA_MODEL=poolside/laguna-xs-2.1
set LAGUNA_BASE=https://ai.hackclub.com/proxy/v1

echo.
echo ============================================
echo Laguna XS 2.1 convergence study
echo ============================================

for %%P in (%PROBES%) do (
    echo.
    echo Running Laguna with %%P probes

    python quickstart_api.py ^
        --api-key %GE_KEY% ^
        --model %LAGUNA_MODEL% ^
        --api-base %LAGUNA_BASE% ^
        --n-probes %%P ^
        --gravity-levels %GRAVITIES% ^
        --master-seeds %SEEDS%

    echo Finished Laguna %%P probes
)


REM =========================
REM GPT-5-mini
REM Azure
REM =========================

set GPT_MODEL=gpt-5-mini
set GPT_BASE=https://arcusx-resource.services.ai.azure.com/openai/v1

echo.
echo ============================================
echo GPT-5-mini convergence study
echo ============================================

for %%P in (%PROBES%) do (
    echo.
    echo Running GPT-5-mini with %%P probes

    python quickstart_api.py ^
        --api-key %GPT_KEY% ^
        --model %GPT_MODEL% ^
        --api-base %GPT_BASE% ^
        --n-probes %%P ^
        --gravity-levels %GRAVITIES% ^
        --master-seeds %SEEDS%

    echo Finished GPT-5-mini %%P probes
)


echo.
echo ============================================
echo Convergence study complete
echo ============================================

pause