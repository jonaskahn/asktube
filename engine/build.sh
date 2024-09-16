#!/bin/bash
docker buildx create --use --name py3-builder --node py3-builder0
docker buildx build --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-engine:latest . --push
