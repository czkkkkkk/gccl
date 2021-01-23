#!/bin/bash

RANK=$((OMPI_COMM_WORLD_RANK))
ARGS=$@
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]
then
  $ARGS
else
  $ARGS >./logs/m${RANK} 2>&1
fi
