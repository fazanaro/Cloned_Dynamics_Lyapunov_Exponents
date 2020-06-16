#!/bin/bash
# $0 is the script name, $1 id the first ARG, $2 is second...
NAME="$1"

# Compila
gcc ${NAME}.c -o ${NAME} -lm -std=c99


# Executa Trial 01
clear

./${NAME}

# Copia os dados para o diretorio correspondente
cp -rf *.dat Trial01/



# Executa Trial 02
#clear

./${NAME}

# Copia os dados para o diretorio correspondente
cp -rf *.dat Trial02/



# Executa Trial 03
#clear

./${NAME}

# Copia os dados para o diretorio correspondente
cp -rf *.dat Trial03/


# Executa Trial 04
#clear

./${NAME}

# Copia os dados para o diretorio correspondente
cp -rf *.dat Trial04/



# Executa Trial 05
#clear

./${NAME}

# Copia os dados para o diretorio correspondente
cp -rf *.dat Trial05/
