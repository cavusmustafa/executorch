#!/bin/bash

cp main_inpfile.cpp main.cpp

rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO_BACKEND=ON ..
make -j$(nproc)
