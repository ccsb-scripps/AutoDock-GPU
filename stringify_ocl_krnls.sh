#!/bin/bash

# kernel-header files
HEADER_DIR="./common"
IN_HEADER1=$HEADER_DIR/"defines.h"
IN_HEADER2=$HEADER_DIR/"calcenergy_basic.h"

echo " "
echo "Stringified input header files: "
echo $IN_HEADER1
echo $IN_HEADER2

# device source-code folder
KERNEL_DIR="./device"

# non-kernel function files
## AUXI:   auxiliary
## ENER:   energy
## GRAD:   gradient
## ENEGRA: merged energy & gradient
IN_FUNCTION_AUXI=$KERNEL_DIR/"auxiliary_genetic.cl"
IN_FUNCTION_ENER=$KERNEL_DIR/"calcenergy.cl"
IN_FUNCTION_GRAD=$KERNEL_DIR/"calcgradient.cl"
IN_FUNCTION_MERGED_EG=$KERNEL_DIR/"calcMergedEneGra.cl"

# kernel files
## INIT: energy calculation of initial population
## EVAL: evaluation count 
## GS_ORIG_GA: Global Search Original Genetic Algorithm
## LS_ORIG_SW: Local Search Original Solis-Wets
## LS_GRAD_{SD | FI | AD}: Local Search Gradient-based {Steepest-Descent | Fire | Ada-Delta}
IN_KERNEL_INIT=$KERNEL_DIR/"kernel1.cl"
IN_KERNEL_EVAL=$KERNEL_DIR/"kernel2.cl"
IN_KERNEL_GS_ORIG_GA=$KERNEL_DIR/"kernel4.cl"
IN_KERNEL_LS_ORIG_SW=$KERNEL_DIR/"kernel3.cl"
IN_KERNEL_LS_GRAD_SD=$KERNEL_DIR/"kernel_sd.cl"
IN_KERNEL_LS_GRAD_FI=$KERNEL_DIR/"kernel_fire.cl"
IN_KERNEL_LS_GRAD_AD=$KERNEL_DIR/"kernel_ad.cl"

echo " "
echo "Stringified input non-kernel files: "
echo $IN_FUNCTION_AUXI
echo $IN_FUNCTION_ENER
echo $IN_FUNCTION_GRAD
echo $IN_FUNCTION_MERGED_EG

echo " "
echo "Stringified input kernel-files: "
echo $IN_KERNEL_INIT
echo $IN_KERNEL_EVAL
echo $IN_KERNEL_GS_ORIG_GA
echo $IN_KERNEL_LS_ORIG_SW
echo $IN_KERNEL_LS_GRAD_SD
echo $IN_KERNEL_LS_GRAD_FI
echo $IN_KERNEL_LS_GRAD_AD

# output file
OUT=host/inc/stringify.h

echo " "
echo "Stringified output file: "
echo $OUT
echo " "

# temporal file
TMP=$KERNEL_DIR/stringify_tmp

echo "// OCLADock: AUTOMATICALLY GENERATED FILE, DO NOT EDIT." >$TMP
echo "#ifndef STRINGIFY_H" >>$TMP
echo "#define STRINGIFY_H" >>$TMP

echo "const char *calcenergy_ocl =" >>$TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_HEADER1            >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_HEADER2            >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_FUNCTION_AUXI      >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_FUNCTION_ENER      >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_FUNCTION_GRAD      >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_FUNCTION_MERGED_EG >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_INIT        >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_EVAL        >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_GS_ORIG_GA  >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_LS_ORIG_SW  >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_LS_GRAD_SD  >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_LS_GRAD_FI  >> $TMP
sed 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN_KERNEL_LS_GRAD_AD  >> $TMP
echo ";" >>$TMP

echo "#endif // End of STRINGIFY_H" >>$TMP

# remove "#include" lines
grep -v '#include' $TMP > $OUT
