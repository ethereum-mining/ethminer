@ECHO OFF

SETLOCAL
SET "FILE_DIR=%~dp0"
PUSHD "%FILE_DIR%"

SET "AStyleVerReq=3.0.1"
astyle --ascii --version 2>NUL || (ECHO. & ECHO ERROR: AStyle not found & GOTO End)
CALL :SubCheckVer || GOTO End


:Start
TITLE Running astyle using %FILE_DIR%astyle.ini

IF "%~1" == "" (
  astyle --ascii -r --options=astyle.ini ..\*.cpp
  astyle --ascii -r --options=astyle.ini --keep-one-line-blocks ..\*.h
) ELSE (
  FOR %%G IN (%*) DO astyle --ascii --options=astyle.ini %%G
)

IF %ERRORLEVEL% NEQ 0 ECHO. & ECHO ERROR: Something went wrong!


:End
POPD
ECHO. & ECHO Press any key to close this window...
PAUSE >NUL
ENDLOCAL
EXIT /B


:SubCheckVer
TITLE Checking astyle version
FOR /F "tokens=4 delims= " %%A IN ('astyle --ascii --version 2^>^&1 NUL') DO (
  SET "AStyleVer=%%A"
)

IF %AStyleVer% NEQ %AStyleVerReq% (
  ECHO. & ECHO ERROR: AStyle v%AStyleVerReq% is required.
  EXIT /B 1
)
EXIT /B
