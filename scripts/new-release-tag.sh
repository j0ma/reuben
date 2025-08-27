#!/usr/bin/env bash

# Error out if argument not provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version-tag>"
    exit 1
fi

version_tag=${1}

git tag -a ${version_tag} -m "Release ${version_tag}"
