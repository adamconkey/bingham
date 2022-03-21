#!/usr/bin/env bash

# TODO add building c code

CFLAGS="-I../c/local/include" LDFLAGS="-L../c/local/lib" python setup.py build_ext --inplace
CFLAGS="-I../c/local/include" LDFLAGS="-L../c/local/lib" python setup.py install
