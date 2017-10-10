#!/bin/bash
# Remove license preamble from all OCLADock source and header files
# if license preamble is present

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

# full list of source files
#OCLADOCK_SOURCE="$KRNL_HEADER_DIR/*.h $KRNL_SOURCE_DIR/*.cl $HOST_HEADER_DIR/*.h $HOST_SOURCE_DIR/*.cpp"

OCLADOCK_SOURCE="$KRNL_HEADERS $KRNL_SOURCES $HOST_HEADERS $HOST_SOURCES"

# Print variables
#echo $KRNL_HEADER_DIR/*.h
#echo $KRNL_SOURCE_DIR/*.cl
#echo $HOST_HEADER_DIR/*.h
#echo $HOST_SOURCE_DIR/*.cpp
#echo $OCLADOCK_SOURCE

# Add license-preamble
# Excluding sources that already have it, and
# excluding the automatically-generated ./host/inc/stringify.h
for f in $OCLADOCK_SOURCE; do
	if [ "$f" != "$HOST_HEADER_DIR/stringify.h" ]; then

		if (grep -q "Copyright (C)" $f); then
			echo "Removing existing license-preamble from $f"
			sed '/\/\*/,/\*\//{0,/\*\//d}' "$f" > "$f.old"
			mv "$f.old" "$f"
			echo "Done!"
		else
			echo "License-preamble was not found in $f"
			echo "No license-preamble is removed."
		fi
		echo " "
	fi
done

# Reference
# sed '/PATTERNSTART/,/PATTERNEND/{0,/PATTERNEND/d}' file 













































































































































































































































































HOST_SOURCE_DIR="./host/src"
HOST_SOURCES="$HOST_SOURCE_DIR/*.cpp"

# full list of source files
#OCLADOCK_SOURCE="$KRNL_HEADER_DIR/*.h $KRNL_SOURCE_DIR/*.cl $HOST_HEADER_DIR/*.h $HOST_SOURCE_DIR/*.cpp"

OCLADOCK_SOURCE="$KRNL_HEADERS $KRNL_SOURCES $HOST_HEADERS $HOST_SOURCES"

# Print variables
#echo $KRNL_HEADER_DIR/*.h
#echo $KRNL_SOURCE_DIR/*.cl
#echo $HOST_HEADER_DIR/*.h
#echo $HOST_SOURCE_DIR/*.cpp
#echo $OCLADOCK_SOURCE

# Add license-preamble
# Excluding sources that already have it, and
# excluding the automatically-generated ./host/inc/stringify.h
for f in $OCLADOCK_SOURCE; do
	if [ "$f" != "$HOST_HEADER_DIR/stringify.h" ]; then

		if (grep -q "Copyright (C)" $f); then
            		echo "Removing existing license-preamble from $f"
		        awk '/^\/\*/{c++} c!=1; /^ \*\//{c++}' ${f} > tmp.txt
        		cp tmp.txt $f
        		echo "Done!"	
		else
			echo "License-preamble was not found in $f"
			echo "No license-preamble is removed."
		fi
		echo " "
	fi
done
