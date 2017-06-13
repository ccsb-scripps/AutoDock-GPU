#ifndef COMMAND_QUEUES_H
#define COMMAND_QUEUES_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"

/*
Create command queue
Inputs:
       	context -
        device_id -

Outputs:
       	command_queue -
*/
  int createCommandQueue(cl_context        context,
                         cl_device_id	   device_id,
                         cl_command_queue* command_queue);

/*
Get devices of the first platform
Inputs:
        cmd_queue -
Outputs:
        none
*/
  int getCommandQueueInfo(cl_command_queue cmd_queue);

// Include code for setCommandQueueProperties()?

#endif /* COMMAND_QUEUES_H */

