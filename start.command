#!/bin/bash
# VTModelMaker - ブラウザで画面を開く。使っている間は、このウィンドウを開いたままにしておく。
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "まだインストールされていません。先に install.command をダブルクリックしてください。"
  read -n 1 -s -r -p "（何かキーを押すと閉じます）"; exit 1
fi
exec .venv/bin/python -X utf8 vtmm/app/server.py "$@"
