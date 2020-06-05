#!/bin/bash
# Test if Cuda can be used for compiling

current_dir=`pwd`
script_dir=`dirname $0`
cd "$script_dir"
if [[ ! -f "test_cuda" ]]; then
	$1 -I$2 -L$3 -lcuda -lcudart -o test_cuda test_cuda.cpp &> /dev/null
	test -e test_cuda && echo yes || echo no
else
	test -e test_cuda && echo yes || echo no
fi
cd "$current_dir"

