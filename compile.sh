#!/bin/bash

if [[ -d build ]] 
then
  rm -rf build
fi

mkdir build 
cp test/build_clothoid.txt build/
cd build
cmake .. -DTEST:STRING=$1 
make 
ctest $2 
