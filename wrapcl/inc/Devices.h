#ifndef DEVICES_H
#define DEVICES_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"

/*
Get devices of the first platform
Inputs:
	platform_id -
	plaformCount -
Outputs:
	device_id -
	deviceCount -
*/

  int getDevices(cl_platform_id  platform_id,
                 cl_uint         platformCount,
                 cl_device_id**  device_id,
                 cl_uint*        deviceCount);

/*
Get devices' attributes
Inputs:
        device_id -
	deviceCount -
Outputs:
        none
*/
  int getDeviceAttributes(cl_device_id* device_id,
			  cl_uint       deviceCount);


#endif /* DEVICES_H */
