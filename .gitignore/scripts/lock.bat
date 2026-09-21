@echo off
REM Regenerate both pinned lockfiles from their flexible specs.
REM
REM   requirements.txt        -> requirements-lock.txt        (runtime / API-only)
REM   requirements-local.txt  -> requirements-lock-local.txt  (optional local inference)
REM
REM Requires pip-tools:  pip install -r requirements-dev.txt
REM
REM NOTE: vllm's wheel contains very long nested paths that exceed Windows'
REM MAX_PATH when pip extracts metadata. We redirect TEMP to a short path to
REM avoid "FileNotFoundError" during resolution.

setlocal
set "SHORT_TMP=%SystemDrive%\tmp"
if not exist "%SHORT_TMP%" mkdir "%SHORT_TMP%"
set "TEMP=%SHORT_TMP%"
set "TMP=%SHORT_TMP%"

python -m piptools compile requirements.txt -o requirements-lock.txt --strip-extras
python -m piptools compile requirements-local.txt -o requirements-lock-local.txt --strip-extras

echo Locks regenerated: requirements-lock.txt, requirements-lock-local.txt
endlocal
