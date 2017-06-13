#ifndef PLATFORMS_H
#define PLATFORMS_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"

/* Get all available platforms
   Inputs:
       	none
   Outputs:
       	platform_id -
       	platformCount -
*/
  int getPlatforms(cl_platform_id** platforms_id,
                   cl_uint*         platformCount);

/* Get all platforms' attributes
   Inputs:
        none
   Outputs:
        platform_id -
        platformCount -
*/
  int getPlatformAttributes(cl_platform_id* platform_id,
			    cl_uint         platformCount);

#endif /* PLATFORMS_H */
