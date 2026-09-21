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
set PROBES=1 3 5


REM =========================
REM GPT-5-mini (Azure)
REM =========================

set GPT_MODEL=gpt-5-mini
set GPT_BASE=YOUR_AZURE_BASE
set GPT_KEY=YOUR_AZURE_KEY

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


REM =========================
REM Gemini 2.5 Flash Lite
REM =========================

set GEMINI_MODEL=google/gemini-2.5-flash-lite
set GEMINI_BASE=YOUR_GEMINI_BASE
set GEMINI_KEY=YOUR_GEMINI_KEY

echo.
echo ============================================
echo Gemini 2.5 Flash Lite convergence study
echo ============================================

for %%P in (%PROBES%) do (
    echo.
    echo Running Gemini with %%P probes

    python quickstart_api.py ^
        --api-key %GEMINI_KEY% ^
        --model %GEMINI_MODEL% ^
        --api-base %GEMINI_BASE% ^
        --n-probes %%P ^
        --gravity-levels %GRAVITIES% ^
        --master-seeds %SEEDS%

    echo Finished Gemini %%P probes
)


echo.
echo ============================================
echo Convergence study complete
echo ============================================

pause