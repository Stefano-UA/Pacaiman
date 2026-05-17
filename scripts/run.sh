#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

cd "${HERE}/.."
"${HERE}/../.venv/bin/python" pacman.py ${@}
