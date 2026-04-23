#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo ""
echo "Done. Activate with:  source .venv/bin/activate"
