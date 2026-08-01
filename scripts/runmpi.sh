#!/bin/bash

OVER=""

# 最初の引数が -o または --oversubscribe の場合
if [ "$1" = "-o" ] || [ "$1" = "--oversubscribe" ]; then
    OVER="--oversubscribe"
    shift
fi

if [ $# -lt 2 ]; then
    echo "使い方:"
    echo "  $0 [-o|--oversubscribe] <プロセス数> <Pythonファイル> [Python引数...]"
    exit 1
fi

NP=$1
shift

# MPI実行
mpirun --hostfile hostfile $OVER -np "$NP" python3 "$@"