#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

"${HERE}/clear.sh"

data="$1"
shift

cp "${HERE}/../pacman_data/${data}/"*.csv "${HERE}/../pacman_data"

cd "${HERE}/.."
"${HERE}/../.venv/bin/python" net.py -p NeuralAgent ${@}
