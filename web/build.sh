#!/bin/bash
docker buildx create --use --name bun-builder --node bun-builder0
docker buildx build --platform linux/arm64,linux/amd64 --tag ifelsedotone/asktube-web:latest . --push
