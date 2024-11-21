#!/bin/bash

join_with_comma() {
    arr=("$@")
    joined=$(printf "%s," "${arr[@]}")
    echo "${joined%,}"
}

repeat_elements() {
    elements=$1
    count=$2
    repeated_array=()
    for ((i = 1; i <= count; i++)); do
        repeated_array+=("$elements")
    done
    join_with_comma "${repeated_array[@]}"
}

repeat_templates_for_tasks() {
    declare -n _templates=$1
    declare -n _tasks=$2
    repeated_array=()
    for template in "${_templates[@]}"; do
        repeated=$(repeat_elements "$template" "${#_tasks[@]}")
        repeated_array+=("$repeated")
    done
    join_with_comma "${repeated_array[@]}"
}
repeat_tasks_for_templates() {
    declare -n _tasks=$1
    declare -n _templates=$2
    joined_tasks=$(join_with_comma "${_tasks[@]}")
    repeat_elements "$joined_tasks" "${#_templates[@]}"
}
