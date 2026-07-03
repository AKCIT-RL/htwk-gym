#!/usr/bin/env bash
# Abre a janela interativa do MuJoCo com o robô no campo de futebol.
# No macOS o viewer PRECISA do mjpython (nao do python normal).
cd "$(dirname "$0")"
exec .venv/bin/mjpython play_mujoco_soccer_walk.py "$@"
