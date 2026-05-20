#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

model="$1"
shift

cp "${HERE}/../models/own/${model}/pacman_model.pth" "${HERE}/../models/pacman_model.pth"