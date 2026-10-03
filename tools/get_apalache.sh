#!/usr/bin/env bash
# Install the pinned Apalache release into tools/apalache/ (git-ignored).
#   tools/get_apalache.sh                  download from GitHub releases
#   APALACHE_TGZ=/path/to.tgz tools/get_apalache.sh   use a local copy
# Then: tools/apalache/bin/apalache-mc version     (needs Java >= 21)
set -euo pipefail
VER=0.62.3
SHA256=14482cc91a3184cf5612d304a245c25126c01cf7ef0a8fe9d56bcf174167850e
here="$(cd "$(dirname "$0")" && pwd)"
dest="$here/apalache"
if [ -x "$dest/bin/apalache-mc" ]; then echo "already installed: $dest"; exit 0; fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
tgz="${APALACHE_TGZ:-}"
if [ -z "$tgz" ]; then
  tgz="$tmp/apalache-$VER.tgz"
  curl -fsSL -o "$tgz" "https://github.com/apalache-mc/apalache/releases/download/v$VER/apalache-$VER.tgz"
fi
echo "$SHA256  $tgz" | sha256sum -c -
tar -xzf "$tgz" -C "$tmp"
rm -rf "$dest" && mv "$tmp/apalache-$VER" "$dest"
"$dest/bin/apalache-mc" version
