@echo off

:: Check if RELEASE file exists
if not exist "RELEASE" (
    echo Error: RELEASE file not found
    exit /b 1
) else (
    echo Start building
)

:: Get the version from the RELEASE file
for /f "tokens=*" %%i in (RELEASE) do set version=%%i
echo Current version: %version%

:: Log files for each job
set ENGINE_LOG=logs\engine_build.log
set WEB_LOG=logs\web_build.log

:: Start to build web and log output
echo.
echo Start to build web . . .
(
    pushd web || exit
    docker buildx create --use --name bun-builder --node bun-builder0
    docker buildx build --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-web:latest . --load 2>&1 | tee ..\%WEB_LOG%
    popd
) &

:: Start to build engine and log output
echo.
echo Start to build engine . . .
(
    pushd engine || exit
    docker buildx create --use --name py3-builder --node py3-builder0
    docker buildx build --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-engine:latest . --load 2>&1 | tee ..\%ENGINE_LOG%
    popd
) &

:: Wait for both builds to finish
:waitloop
timeout /t 1 >nul
tasklist | find /i "docker" >nul
if errorlevel 1 (
    goto endbuild
) else (
    goto waitloop
)

:endbuild
echo Both builds completed.
