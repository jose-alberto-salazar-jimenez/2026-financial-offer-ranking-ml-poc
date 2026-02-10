#!/usr/bin/env bash
# Run app via docker-compose
set -e
cd "$(dirname "$0")/.."
docker-compose up --build
