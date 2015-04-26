#!/usr/bin/env sh
./svm-python-v204/svm_python_learn -e $2 -c $3 $1 tmmppp
mv model/tmmppp model/$(./check.py tmmppp)
