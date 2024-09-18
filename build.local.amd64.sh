#!/bin/bash

if [ ! -f "RELEASE" ]; then
    echo "Error: RELEASE file not found"
    exit 1
else
    echo "Start building"
fi

version=$(cat RELEASE)
echo "Current version: $version"

# Log files for each job
ENGINE_LOG="logs/engine_build.log"
WEB_LOG="logs/web_build.log"

# Start to build web and log output
echo -e "\nStart to build web . . .\n"
(
    cd web || exit
    docker buildx create --use --name bun-builder --node bun-builder0
    docker buildx build --platform linux/amd64 --tag ifelsedotone/asktube-web:latest . --load 2>&1 | tee "../$WEB_LOG"
) & # Run in background

# Start to build engine and log output
echo -e "\nStart to build engine . . .\n"
(
    cd engine || exit
    docker buildx create --use --name py3-builder --node py3-builder0
    docker buildx build --platform linux/amd64 --tag ifelsedotone/asktube-engine:latest . --load 2>&1 | tee "../$ENGINE_LOG"
) & # Run in background

wait

echo "Both builds completed."
