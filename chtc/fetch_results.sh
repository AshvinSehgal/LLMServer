#!/bin/bash
# Pulls every results tarball from a CHTC submit node and unpacks each into results/<tag>/.
# usage: chtc/fetch_results.sh <user@submit-node> [remote dir, default: LLMServer] [--dry-run]
#   e.g. chtc/fetch_results.sh asehgal9@ap2002.chtc.wisc.edu
set -e
HOST=$1
if [ -z "$HOST" ]; then echo "usage: $0 <user@submit-node> [remote dir] [--dry-run]" >&2; exit 1; fi
SRC=${2:-LLMServer}
EXTRA=$3
[ "$SRC" = "--dry-run" ] && { EXTRA=--dry-run; SRC=LLMServer; }

cd "$(dirname "$0")/.."
rsync -avh --progress $EXTRA --include='llmserver_results*.tar.gz' --exclude='*' "$HOST:$SRC/" ./
[ -n "$EXTRA" ] && exit 0

for tarball in llmserver_results*.tar.gz; do
    tag=${tarball#llmserver_results}; tag=${tag%.tar.gz}; tag=${tag#_}
    dir=results/${tag:-default}
    if [ -f "$dir/.unpacked_from" ] && [ "$dir/.unpacked_from" -nt "$tarball" ]; then continue; fi  # already unpacked, tarball unchanged
    rm -rf "$dir"; mkdir -p "$dir"
    tar -xzf "$tarball" -C "$dir"
    touch "$dir/.unpacked_from"
    echo "unpacked $tarball -> $dir ($(grep -h 'code version' "$dir"/STATUS.txt "$dir"/*/STATUS.txt 2>/dev/null | head -1))"
done
