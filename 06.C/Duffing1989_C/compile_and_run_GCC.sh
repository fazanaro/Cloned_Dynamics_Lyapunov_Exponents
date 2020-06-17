#!/bin/bash
# $0 is the script name, $1 id the first ARG, $2 is second...
NAME="$1"

# Compile
gcc ${NAME}.c -o ${NAME} -lm -std=c99


# Execute
clear

./${NAME}
