@echo off
rem VTModelMaker - installer for Windows. Double-click this file.
rem This file is ASCII on purpose: cmd.exe reads batch files in the console's code page. The Japanese messages are
rem PowerShell commands encoded in Base64 (:say - quoted, since cmd splits arguments at '='), and the rest come
rem from Python.
setlocal
cd /d "%~dp0"
call :say "VwByAGkAdABlAC0ASABvAHMAdAAgACcAPQA9ACAAVgBUAE0AbwBkAGUAbABNAGEAawBlAHIAIACSMKQw8zC5MMgw/DDrMFcwfjBZMCcA"
echo    %CD%
call :say "VwByAGkAdABlAC0ASABvAHMAdAAgACcAIAAgACAACP9QAHkAdABoAG8AbgAgADMALgAxADIAIABoMMVfgYlqMOiQwVQBMAR9MQBHAEIAkjABMFMwbjDVMKkw6zDAMG4wIAAuAHYAZQBuAHYAIABrMMAwpjDzMO0w/DDJMFcwfjBZMAn/JwA="

set "UV="
where uv >nul 2>nul && set "UV=uv"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV (
  call :say "VwByAGkAdABlAC0ASABvAHMAdAAgACcAPQA9ACAAdQB2AAj/UAB5AHQAaABvAG4AIACSMCh1D2FZMIswU5B3UToAIABoAHQAdABwAHMAOgAvAC8AYQBzAHQAcgBhAGwALgBzAGgALwB1AHYACf+SMGVRjDB+MFkwJwA="
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)

if not defined UV goto fallback
"%UV%" venv --python 3.12 --allow-existing .venv
if errorlevel 1 goto fallback
"%UV%" pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto fallback
goto finish

:fallback
call :say "VwByAGkAdABlAC0ASABvAHMAdAAgACcAPQA9ACAAdQB2ACAAZzBGMH4wTzBEMEswajBLMGMwXzBuMGcwATBTMG4w0TC9MLMw8zBuMCAAUAB5AHQAaABvAG4AIABnMGaKVzB+MFkwJwA="
py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail

:finish
.venv\Scripts\python.exe -X utf8 vtmm\install_finish.py %*
echo.
pause
exit /b 0

:fail
call :say "VwByAGkAdABlAC0ASABvAHMAdAAgACcAIQAhACAApDDzMLkwyDD8MOswZzBNMH4wWzCTMGcwVzBfMAIwCk5uMOEwwzC7MPwwuDCSMIuJZjBPMGAwVTBEMAIwJwA="
pause
exit /b 1

:say
powershell -NoProfile -EncodedCommand %~1
exit /b 0
