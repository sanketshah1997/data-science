@echo off

set script_path=D:\data_science\Projects\Scripts

:: Fetch environment name as param1
setlocal
set "par1=%~1"
goto :param1Check

:param1Prompt
set /p "par1=Enter Environment name/path to be activated: "

:param1Check
if "%par1%"=="" goto :param1Prompt

echo Setting the conda path
call %script_path%\set_conda_path.bat 

echo activating the environment %par1%

rem calling conda activate %par1%
ENDLOCAL & call conda activate %par1% 
