#!/bin/bash
# Copy license preamble to all AutoDock-GPU source and header files
# if license preamble is not present

# license preamble
LICENSE_PREAMBLE="./preamble_license"
LICENSE_PREAMBLE_LGPL="./preamble_license_lgpl"

# kernel-header files
KRNL_HEADER_DIR="./common"
KRNL_HEADERS="$KRNL_HEADER_DIR/*.h"

# kernel-source files
KRNL_SOURCE_DIR="./device"
KRNL_SOURCES="$KRNL_SOURCE_DIR/*.cl"

# kernel-source files
CUDA_SOURCE_DIR="./cuda"
CUDA_SOURCES="$CUDA_SOURCE_DIR/*.cu"
CUDA_HEADERS="$CUDA_SOURCE_DIR/*.h"

# host-header files
HOST_HEADER_DIR="./host/inc"
HOST_HEADERS="$HOST_HEADER_DIR/calcenergy.h $HOST_HEADER_DIR/correct_grad_axisangle.h $HOST_HEADER_DIR/getparameters.h $HOST_HEADER_DIR/miscellaneous.h $HOST_HEADER_DIR/processgrid.h
$HOST_HEADER_DIR/processligand.h $HOST_HEADER_DIR/processresult.h"
HOST_HEADERS_LGPL="$HOST_HEADER_DIR/performdocking.h $HOST_HEADER_DIR/*.hpp" 

# host-source files
HOST_SOURCE_DIR="./host/src"
HOST_SOURCES="$HOST_SOURCE_DIR/calcenergy.cpp $HOST_SOURCE_DIR/getparameters.cpp $HOST_SOURCE_DIR/miscellaneous.cpp $HOST_SOURCE_DIR/processgrid.cpp $HOST_SOURCE_DIR/processligand.cpp $HOST_SOURCE_DIR/processresult.cpp"
HOST_SOURCES_LGPL="$HOST_SOURCE_DIR/performdocking.cpp $HOST_SOURCE_DIR/main.cpp"

# wrapcl-header files
WRAPCL_HEADER_DIR="./wrapcl/inc"
WRAPCL_HEADERS="$WRAPCL_HEADER_DIR/*.h"

# wrapcl-source files
WRAPCL_SOURCE_DIR="./wrapcl/src"
WRAPCL_SOURCES="$WRAPCL_SOURCE_DIR/*.cpp"

# full list of source files
AUTODOCKGPU_SOURCE="$HOST_HEADERS $HOST_SOURCES $WRAPCL_HEADERS $WRAPCL_SOURCES"
AUTODOCKGPU_SOURCE_LGPL="$HOST_HEADERS_LGPL $HOST_SOURCES_LGPL $KRNL_HEADERS $KRNL_SOURCES $CUDA_SOURCES $CUDA_HEADERS"

# Add license-preamble
# Excluding sources that already have it, and
# excluding the automatically-generated ./host/inc/stringify.h
for f in $AUTODOCKGPU_SOURCE; do
	if [ "$f" != "$HOST_HEADER_DIR/stringify.h" ]; then

		if (grep -q "Copyright (C)" $f); then
			echo "License-preamble was found in $f"
			echo "No license-preamble is added."
                        if (grep -q "GNU Lesser" $f); then
                          echo "Wrong license found in $f"
                        fi
		else
			echo "Adding license-preamble to $f ..."
			cat $LICENSE_PREAMBLE "$f" > "$f.new"
			mv "$f.new" "$f"
			echo "Done!"
		fi
		echo " "
	fi
done

for f in $AUTODOCKGPU_SOURCE_LGPL; do
	if [ "$f" != "$HOST_HEADER_DIR/stringify.h" ]; then

		if (grep -q "Copyright (C)" $f); then
			echo "License-preamble was found in $f"
			echo "No license-preamble is added."
                        if (grep -q "GNU General Public License" $f); then
                          echo "Wrong license found in $f"
                        fi
		else
			echo "Adding LGPL license-preamble to $f ..."
			cat $LICENSE_PREAMBLE_LGPL "$f" > "$f.new"
			mv "$f.new" "$f"
			echo "Done!"
		fi
		echo " "
	fi
done
