#!/bin/bash
#First option is CUDA_ON=ON or CUDA_ON=OFF to activate or not CUDA
#Second option is DEBUG=ON, DEBUG=TS or DEBUG=OFF to activate for all code, only for testing or deactivate respectively
#Third option is the choice of the testing framework, either GTEST or BOOST

if [[ -d build ]] 
then
  echo "Removing dir"
  rm -rf build
fi

mkdir build 
cd build
cmake .. -D$1 -D$2 -DTEST:STRING=$3 
make 
ctest $3 --rerun-failed --output-on-failure
