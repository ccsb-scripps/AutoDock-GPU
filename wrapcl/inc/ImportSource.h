#ifndef IMPORT_SOURCE_H
#define IMPORT_SOURCE_H

// L30nardoSV
//#include <stdio.h>
#include <iostream>
#include <string.h>
#include <fstream>

//#include <stdlib.h>
#include "commonMacros.h"
#include "Programs.h"
#include "Kernels.h"

using namespace std;

/*
*/
int convertToString2(const char *filename, std::string& s);


/*

*/
int ImportSource(const char*    filename,
				 const char*    kernel_name,
				 cl_device_id*  device_id,
				 cl_context     context,
				 /*cl_program*  program,*/
				 const char*    options,
				 cl_kernel*     kernel);

#endif /* IMPORT_SOURCE_H */
