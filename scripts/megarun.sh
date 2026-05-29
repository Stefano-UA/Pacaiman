#!/bin/bash

export HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

n=$1
shift

export AGENT_DEPTH='4'
export PACMAN_RANDOM=True
export ARGS=$*

run_game() {
    local n=$1
    shift

    SECONDS=0

    cd "${HERE}/.."

    echo "Ejecutando partida ${n}..."

    # Clave porque sino escriben el mismo csv a la vez
    export PACMANDATA='pacman_data/wk'$n
    mkdir -p "$PACMANDATA"

    local tmpfile="$(mktemp)"

    "${HERE}/run.sh" $ARGS --frameTime 0.001 | tee "$tmpfile"

    local csv="$(grep -o 'pacman_data/[a-zA-Z0-9_]*\.csv' "$tmpfile")"

    if grep -q 'Loss' "$tmpfile"; then
        rm -fr "$PACMANDATA"
        echo "Partida ${n} perdida -> CSV purgado"
    else
        cp "$PACMANDATA"/game_0.csv "pacman_data/game_${n}.csv" 2>/dev/null
        echo "Partida ${n} ganada -> CSV copied to pacman_data"
    fi

    rm -f "$tmpfile"

    local mins=$((SECONDS / 60))
    local secs=$((SECONDS % 60))
    echo "Partida ${n} completada -> Took ${mins}m ${secs}s"
}

export -f run_game

seq 1 $n | xargs -P $(nproc) -I {} bash -c 'run_game {}'
