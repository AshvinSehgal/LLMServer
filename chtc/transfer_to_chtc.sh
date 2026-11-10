#!/bin/bash
# Copies this repo folder to a CHTC submit node, skipping secrets, caches and large local files.
# usage: chtc/transfer_to_chtc.sh <user@submit-node> [remote dir, default: LLMServer] [--dry-run]
#   e.g. chtc/transfer_to_chtc.sh asehgal9@ap2002.chtc.wisc.edu
set -e
HOST=$1
if [ -z "$HOST" ]; then echo "usage: $0 <user@submit-node> [remote dir] [--dry-run]" >&2; exit 1; fi
DEST=${2:-LLMServer}
EXTRA=$3
[ "$DEST" = "--dry-run" ] && { EXTRA=--dry-run; DEST=LLMServer; }

cd "$(dirname "$0")/.."
rsync -avh --checksum --progress $EXTRA \
    --exclude='.git/' --exclude='.env' --exclude='conversation.md' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.venv/' --exclude='.DS_Store' \
    --exclude='bench_results/' --exclude='results/' --exclude='*.tar.gz' \
    --exclude='chtc/*.log' --exclude='chtc/*.out' --exclude='chtc/*.err' \
    ./ "$HOST:$DEST/"
