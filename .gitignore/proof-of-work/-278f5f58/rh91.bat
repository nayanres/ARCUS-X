@echo off
setlocal

echo ============================================
echo ARCUS-X Probe Convergence Study Launcher
echo ============================================

echo.
echo Launching Laguna XS 2.1...
start "ARCUS-X - Laguna XS 2.1" lang.bat

echo Launching GPT-5-mini...
start "ARCUS-X - GPT-5-mini" gpt.bat

echo.
echo ============================================
echo Both convergence studies are now running.
echo ============================================
echo.
echo Two command windows should now be open:
echo   - Laguna XS 2.1
echo   - GPT-5-mini
echo.
echo You can monitor each independently.
echo.
pause