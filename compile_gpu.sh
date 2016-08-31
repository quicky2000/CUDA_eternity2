#!/bin/sh
nvcc -O3 -lineinfo $1 -std=c++11 --ptxas-options=-v -DNDEBUG
#EOF
