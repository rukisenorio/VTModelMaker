#!/bin/bash
# VTModelMaker - macOS 用のインストーラー。
# このファイルをダブルクリック（初回は右クリック →「開く」）、またはターミナルで:  bash install.command
cd "$(dirname "$0")" || exit 1
echo "== VTModelMaker をインストールします: $(pwd)"
echo "   （Python 3.12 と必要な部品、約1GBを、このフォルダの .venv にダウンロードします）"

UV=""
if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"; elif [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
  echo "== uv（Python を用意する道具: https://astral.sh/uv）を入れます"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
fi

ok=0
if [ -n "$UV" ]; then
  "$UV" venv --python 3.12 --allow-existing .venv && \
  "$UV" pip install --python .venv/bin/python -r requirements.txt && ok=1
fi
if [ $ok -eq 0 ]; then
  echo "== uv でうまくいかなかったので、この Mac の python3 で試します"
  PY="$(command -v python3.12 || command -v python3.11 || command -v python3)"
  if [ -n "$PY" ]; then
    "$PY" -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt && ok=1
  fi
fi
if [ $ok -eq 0 ]; then
  echo "!! インストールできませんでした。上のメッセージを見てください。"
  read -n 1 -s -r -p "（何かキーを押すと閉じます）"; exit 1
fi

.venv/bin/python -X utf8 vtmm/install_finish.py "$@"
chmod +x start.command 2>/dev/null
echo
read -n 1 -s -r -p "（何かキーを押すと閉じます）"
