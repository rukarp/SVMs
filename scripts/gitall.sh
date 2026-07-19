#!/bin/bash

#親PC以外のhost名
hosts=(
    TABLET-N9QO2HSP
)

# Gitリポジトリか確認
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Gitリポジトリではありません。"
    exit 1
}

# コミットメッセージ
if [ $# -eq 0 ]; then
    msg="$(date '+Update %Y-%m-%d %H:%M:%S')"
else
    msg="$*"
fi

# 変更がないなら終了
if git diff --quiet && git diff --cached --quiet; then
    echo "変更はありません。"
    exit 0
fi

git add .
git commit -m "$msg" || exit 1
git push || exit 1

echo "=== 他PCを更新 ==="

for host in "${hosts[@]}"; do
    echo "[$host]"
    ssh yamada-ryota@$host "cd ~/Documents/DSMO && git pull"
done

echo "完了"
