#!/bin/bash

python export_and_validate.py --model_name yolo12s --input_dims=[640,640] --backend openvino --device GPU
