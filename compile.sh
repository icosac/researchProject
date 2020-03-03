#!/bin/bash

if [[ -d build ]] 
then
  rm -rf build
fi

mkdir build 
cd build 
cmake .. -DTEST:STRING=$1
make 
ctest $2 
