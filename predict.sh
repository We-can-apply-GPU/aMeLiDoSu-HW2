#!/usr/bin/env sh
echo "$3"
./svm-python-v204/svm_python_classify -c $3 $1 $2 
