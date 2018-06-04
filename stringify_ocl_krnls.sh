#!/bin/bash

# kernel-header files
HEADER_DIR="./common"
IN_HEADER1=$HEADER_DIR/"defines.h"
IN_HEADER2=$HEADER_DIR/"calcenergy_basic.h"

echo " "
echo "Stringified input header-files: "
echo $IN_HEADER1
echo $IN_HEADER2

# kernel files
KERNEL_DIR="./device"
IN_KERNELm=$KERNEL_DIR/"calcenergy.cl"
IN_KERNEL1=$KERNEL_DIR/"kernel1.cl"
IN_KERNEL2=$KERNEL_DIR/"kernel2.cl"
IN_KERNELa=$KERNEL_DIR/"auxiliary_genetic.cl"
IN_KERNEL3=$KERNEL_DIR/"kernel3.cl"
IN_KERNEL4=$KERNEL_DIR/"kernel4.cl"
IN_KERNELb=$KERNEL_DIR/"calcgradient.cl"
IN_KERNEL5=$KERNEL_DIR/"kernel_gradient.cl"
IN_KERNEL6=$KERNEL_DIR/"kernel_fire.cl"

echo " "
echo "Stringified input kernel-files: "
echo $IN_KERNELm
echo $IN_KERNEL1
echo $IN_KERNEL2
echo $IN_KERNELa
echo $IN_KERNEL3
echo $IN_KERNEL4
echo $IN_KERNELb
echo $IN_KERNEL5
echo $IN_KERNEL6

# output file
OUT=host/inc/stringify.h

echo " "
echo "Stringified output file: "
echo $OUT
echo " "

# temporal file
TMP=$KERNEL_DIR/stringify_tmp

echo "// OCLADOCK: AUTOMATICALLY GENERATED FILE, DO NOT EDIT" >$TMP
echo "#ifndef STRINGIFY_H" >>$TMP
echo "#define STRINGIFY_H" >>$TMP

echo "const char *calcenergy_ocl =" >>$TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_HEADER1 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_HEADER2 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNELm >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL1 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL2 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNELa >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL3 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL4 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNELb >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL5 >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL6 >> $TMP
echo ";" >>$TMP

echo "#endif // End of STRINGIFY_H" >>$TMP

# remove "#include" lines
grep -v '#include' $TMP > $OUT
