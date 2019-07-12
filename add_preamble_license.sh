#!/bin/bash
# Copy license preamble to all AutoDock-GPU source and header files
# if license preamble is not present

# license preamble
LICENSE_PREAMBLE="./preamble_license"

# kernel-header files
KRNL_HEADER_DIR="./common"
KRNL_HEADERS="$KRNL_HEADER_DIR/*.h"

# kernel-source files
KRNL_SOURCE_DIR="./device"
KRNL_SOURCES="$KRNL_SOURCE_DIR/*.cl"

# host-header files
HOST_HEADER_DIR="./host/inc"
HOST_HEADERS="$HOST_HEADER_DIR/*.h"

# host-source files
HOST_SOURCE_DIR="./host/src"
HOST_SOURCES="$HOST_SOURCE_DIR/*.cpp"

# wrapcl-header files
WRAPCL_HEADER_DIR="./wrapcl/inc"
WRAPCL_HEADERS="$WRAPCL_HEADER_DIR/*.h"

# wrapcl-source files
WRAPCL_SOURCE_DIR="./wrapcl/src"
WRAPCL_SOURCES="$WRAPCL_SOURCE_DIR/*.cpp"

# full list of source files
#AUTODOCKGPU_SOURCE="$KRNL_HEADER_DIR/*.h $KRNL_SOURCE_DIR/*.cl $HOST_HEADER_DIR/*.h $HOST_SOURCE_DIR/*.cpp"

AUTODOCKGPU_SOURCE="$KRNL_HEADERS $KRNL_SOURCES $HOST_HEADERS $HOST_SOURCES $WRAPCL_HEADERS $WRAPCL_SOURCES"

# Print variables
#echo $KRNL_HEADER_DIR/*.h
#echo $KRNL_SOURCE_DIR/*.cl
#echo $HOST_HEADER_DIR/*.h
#echo $HOST_SOURCE_DIR/*.cpp
#echo $AUTODOCKGPU_SOURCE

# Add license-preamble
# Excluding sources that already have it, and
# excluding the automatically-generated ./host/inc/stringify.h
for f in $AUTODOCKGPU_SOURCE; do
	if [ "$f" != "$HOST_HEADER_DIR/stringify.h" ]; then

		if (grep -q "Copyright (C)" $f); then
			echo "License-preamble was found in $f"
			echo "No license-preamble is added."
		else
			echo "Adding license-preamble to $f ..."
			cat $LICENSE_PREAMBLE "$f" > "$f.new"
			mv "$f.new" "$f"
			echo "Done!"
		fi
		echo " "
	fi
done
