#ifndef IMPORT_BINARY_H
#define IMPORT_BINARY_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <stdlib.h>
  #include "commonMacros.h"
  #include "Programs.h"
  #include "Kernels.h"

/*

*/
  int load_file_to_memory(const char* filename,
                               char** result);

/*

*/
  int ImportBinary(const char*    kernel_xclbin,
                   const char*    kernel_name,
                   cl_device_id*  device_id,
                   cl_context     context,
                   /*cl_program*  program,*/
                   const char*    options,
                   cl_kernel*     kernel);

#endif /* IMPORT_BINARY_H */
