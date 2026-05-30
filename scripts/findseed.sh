#!/bin/bash

export HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

"${HERE}/init.sh"

if [ $# -lt 2 ]; then
    echo 'Uso: ./script.sh <semilla_inicio> <semilla_fin> [argumentos_extra_pacman...]'
    exit 1
fi

start_seed=$1
end_seed=$2
shift 2

export AGENT_DEPTH='4'
export ARGS=$*

# Archivo de bandera para avisar a los demas procesos que paren
export WIN_FLAG="$(mktemp)"

kill_descendants() {
    if [ -n "$XARGS_PID" ]; then
        # Mata a xargs y a sus hijos directos (las instancias de bash)
        pkill -P $XARGS_PID 2>/dev/null
        kill -9 $XARGS_PID 2>/dev/null
    fi
    # Failsafe: mata cualquier proceso residual de pacman o el script run
    pkill -f 'run.sh' 2>/dev/null
    pkill -f 'pacman.py' 2>/dev/null
}

# Trap para interceptar Ctrl+C (SIGINT) y SIGTERM
cleanup() {
    echo ''
    echo 'Interrupcion detectada. Limpiando procesos en background...'
    kill_descendants
    rm -f "$WIN_FLAG"
    exit 130
}
trap cleanup SIGINT SIGTERM

run_game() {
    local seed=$1

    # Si algun proceso ya gano, los procesos en cola mueren inmediatamente
    if [ -s "$WIN_FLAG" ]; then
        exit 0
    fi

    SECONDS=0

    cd "${HERE}/.." || exit 1

    echo "Ejecutando partida con semilla ${seed}..."

    # Asignacion de la semilla para que la lea el entorno (Python)
    export SEED=$seed

    export PACMANDATA='pacman_data/wk'$seed
    mkdir -p "$PACMANDATA"

    local tmpfile="$(mktemp)"

    "${HERE}/run.sh" $ARGS --frameTime 0.001 > "$tmpfile" 2>&1

    if grep -q 'Loss' "$tmpfile"; then
        rm -fr "$PACMANDATA"
        echo "Partida ${seed} perdida -> CSV purgado"
    else
        # Evitar condicion de carrera si dos ganan en el mismo milisegundo
        if [ ! -s "$WIN_FLAG" ]; then
            echo "$seed" > "$WIN_FLAG"
            cp "$PACMANDATA"/game_0.csv "pacman_data/game_${seed}.csv" 2>/dev/null
            echo "Partida ${seed} ganada -> CSV copiado a pacman_data"
        fi
    fi

    rm -f "$tmpfile"

    local mins=$((SECONDS / 60))
    local secs=$((SECONDS % 60))
    echo "Partida ${seed} completada -> Took ${mins}m ${secs}s"
}

export -f run_game

# Lanzamos el xargs en background para no bloquear el script principal
seq $start_seed $end_seed | xargs -P $(nproc) -I {} bash -c 'run_game {}' &
XARGS_PID=$!

# Bucle de monitorizacion
while kill -0 $XARGS_PID 2>/dev/null; do
    if [ -s "$WIN_FLAG" ]; then
        WINNER="$(cat "$WIN_FLAG")"
        echo ''
        echo '=================================================='
        echo " VICTORIA ENCONTRADA CON LA SEMILLA: ${WINNER}"
        echo '=================================================='

        kill_descendants
        break
    fi
    sleep 0.5
done

rm -f "$WIN_FLAG"