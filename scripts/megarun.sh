#!/bin/bash

HERE="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

n=$1
shift

for ((i=1; i<=n; i++))
do
   "${HERE}/run.sh" ${@} --frameTime 0.001
done
