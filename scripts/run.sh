#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

python -m venv "${HERE}/../.venv"
"${HERE}/../.venv/bin/pip" install numpy matplotlib pandas torch torchvision torchaudio scikit-learn &> /dev/null

cd "${HERE}/.."
"${HERE}/../.venv/bin/python" pacman.py ${@}