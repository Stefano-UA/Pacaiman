#!/bin/bash

export HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

n=$1
shift

export PACMAN_RANDOM=True
export ARGS=$*

run_game() {
    local n=$1
    shift

    SECONDS=0

    echo "Ejecutando partida ${n}..."

    local tmpfile="$(mktemp)"

    "${HERE}/run.sh" $ARGS --frameTime 0.001 | tee "$tmpfile"

    local csv="$(grep -o 'pacman_data/[a-zA-Z0-9_]*\.csv' "$tmpfile")"

    if grep -q 'Record: Loss' "$tmpfile"; then
        rm -f "$csv"
        echo "Partida ${n} perdida -> CSV purgado"
    fi

    rm -f "$tmpfile"

    local mins=$((SECONDS / 60))
    local secs=$((SECONDS % 60))
    echo "Partida ${n} completada -> Took ${mins}m ${secs}s"
}

export -f run_game

seq 1 $n | xargs -P $(nproc) -I {} bash -c 'run_game {}'
