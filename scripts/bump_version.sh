#!/usr/bin/env bash

# bump_tag.sh
# Usage: ./bump_tag.sh {patch|minor|major}

set -euo pipefail

kind="${1:-}"

if [[ ! "$kind" =~ ^(patch|minor|major)$ ]]; then
  echo "usage: $0 {patch|minor|major}" >&2
  exit 1
fi

# Get latest tag; default to v0.0.0 if none exists
latest_tag="$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")"

# Expect tags like v1.2.3 or 1.2.3
if [[ "$latest_tag" =~ ^v?([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
else
  echo "error: latest tag '$latest_tag' is not SemVer (expected vX.Y.Z)" >&2
  exit 1
fi

case "$kind" in
  patch)
    new_major="$major"
    new_minor="$minor"
    new_patch="$((patch + 1))"
    ;;
  minor)
    new_major="$major"
    new_minor="$((minor + 1))"
    new_patch=0
    ;;
  major)
    new_major="$((major + 1))"
    new_minor=0
    new_patch=0
    ;;
esac

new_tag="v${new_major}.${new_minor}.${new_patch}"

# Create annotated tag
git tag -a "$new_tag" -m "Release $new_tag"

# Print it (useful for scripting)
echo "$new_tag" > /dev/stderr
