#!/bin/bash

if [ ! -f "../RELEASE" ]; then
    echo "Error: RELEASE file not found"
    exit 1
else
    echo "Start building"
fi

version=$(cat ../RELEASE)
echo "Current version: $version"

# Start to build engine and log output
echo -e "\nStart to build engine . . .\n"
docker buildx create --use --name py3-builder --node py3-builder0
docker buildx build --progress=plain --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-engine:latest --tag ifelsedotone/asktube-engine:$version . --push

