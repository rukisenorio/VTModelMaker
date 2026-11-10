@echo off
rem VTModelMaker - opens the tool in the browser. Keep this window open while you use it.
rem (ASCII on purpose; the Japanese message is a PowerShell command in Base64, see install.bat)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  powershell -NoProfile -EncodedCommand VwByAGkAdABlAC0ASABvAHMAdAAgACcAfjBgMKQw8zC5MMgw/DDrMFUwjDBmMEQwfjBbMJMwAjBIUWswIABpAG4AcwB0AGEAbABsAC4AYgBhAHQAIACSMMAw1jDrMK8w6jDDMK8wVzBmME8wYDBVMEQwAjAnAA==
  pause
  exit /b 1
)
.venv\Scripts\python.exe -X utf8 vtmm\app\server.py %*
pause
