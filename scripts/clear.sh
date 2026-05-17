#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

rm -f "${HERE}/../pacman_data/"*.csv
