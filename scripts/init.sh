#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

if ! [ -d "${HERE}/../.venv" ]; then
    python -m venv "${HERE}/../.venv"
    "${HERE}/../.venv/bin/pip" install numpy matplotlib pandas torch torchvision torchaudio scikit-learn
fi
