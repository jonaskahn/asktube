#!/bin/bash

if [ ! -f "../RELEASE" ]; then
    echo "Error: RELEASE file not found"
    exit 1
else
    echo "Start building"
fi

version=$(cat ../RELEASE)
echo "Current version: $version"

# Start to build web and log output
echo -e "\nStart to build web . . .\n"
docker buildx create --use --name bun-builder --node bun-builder0
docker buildx build --progress=plain --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-web:latest --tag ifelsedotone/asktube-web:$version . --push

