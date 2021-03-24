@echo off

rem check if path for conda is already set

echo %path |findstr /R "Scripts\conda">nul && (
    rem no need to do anything conda path is already set
	echo %path%
	echo Conda path is already set!
) || (
	
	rem setlocal used for local variable settings. It will destroy the variables after the execution of script.
	rem setlocal 

	echo %path%
	rem initialize the variable conda_path to nul
	set conda_path=nul

	rem execute the where conda command and get the exe location of conda to set the path variable.
	for /f "delims=" %%i in ('where conda') do (
	
		rem find the substring Scripts inside the locations and if present it is conda exe location
		echo %%i|find "Scripts" >nul
		if not errorlevel 1 (
			set conda_path=%%i
		)
	) 
	echo %conda_path%

	rem set path environment variable
	set path=%conda_path%;%path%
	
)