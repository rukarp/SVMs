#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/gitall.sh" || exit 1
"$SCRIPT_DIR/runmpi.sh" "$@"
