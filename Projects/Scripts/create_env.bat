@echo off

set script_path=D:\data_science\Projects\Scripts

:: Fetch environment name as param1


echo Environment will be created in the current directory
set "param1=%~1"
goto :param1Check

:param1Prompt
set /p "param1=Enter Environment name to be created: "

:param1Check
if "%param1%"=="" goto :param1Prompt


call %script_path%\set_conda_path.bat 


set env_folder_name=env_%param1%
call conda create --prefix ./%env_folder_name%
rem use this if you want to create environment in default conda env folder --> call conda create -n %param1% 

echo %errorlevel%
set /p "is_activate=Do you want to activate this environment[y/n]?(It is preferred to activate newly created env and use it) "
goto :activateCheck

:activate_env

set path_to_env=%cd%\%env_folder_name%
echo %path_to_env%
call %script_path%\activate_env.bat %path_to_env%
goto :end

:activateCheck
if "%is_activate%"=="y" goto :activate_env


:end




rem conda remove --name myenv --all
rem endlocal
