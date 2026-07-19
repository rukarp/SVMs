#!/bin/bash

OVER=""

# 最初の引数が --oversubscribe の場合
if [ "$1" = "--oversubscribe" ]; then
    OVER="--oversubscribe"
    shift
fi

if [ $# -lt 2 ]; then
    echo "使い方:"
    echo "  $0 <プロセス数> <Pythonファイル> [Python引数...]"
    echo "  $0 --oversubscribe <プロセス数> <Pythonファイル> [Python引数...]"
    exit 1
fi

NP=$1
shift

# MPI実行
mpirun --hostfile hostfile $OVER -np "$NP" python3 "$@"
