#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

n=$1
agent="$2"
shift 2

export PACMAN_RANDOM=True

cd "${HERE}/.."
"${HERE}/../.venv/bin/python" pacman.py -p "${agent}" -n $n --frameTime 0.001 ${@}
