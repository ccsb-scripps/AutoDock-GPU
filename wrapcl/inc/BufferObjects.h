#ifndef BUFFER_OBJECTS_H
#define BUFFER_OBJECTS_H

  // L30nardoSV
  //#include <stdio.h>
  //#include <CL/opencl.h>
  #include "commonMacros.h"
  /*
   Memory objects are represented by cl_mem objects.
   They come in two types:
   - buffer objects
   - image  objects
  */

/*

*/
  int mallocBufferObject(cl_context   context,
			 cl_mem_flags flags,
			 size_t       size,
			 cl_mem*      mem);

/*

*/
  int getBufferObjectInfo(cl_mem object);

/*

*/
  int memcopyBufferObjectToDevice(cl_command_queue cmd_queue,
                                  cl_mem           dest,
                                  void*            src,
                                  size_t           size);

/*

*/
  int memcopyBufferObjectFromDevice(cl_command_queue cmd_queue,
                                    void*            dest,
                                    cl_mem           src,
                                    size_t           size);

/*

*/
  int memcopyBufferObjectToBufferObject(cl_command_queue cmd_queue,
				        cl_mem     dest,
				        cl_mem     src,
				        size_t     size);

/*

*/
  void* memMap(cl_command_queue cmd_queue,
               cl_mem           dev_mem,
               cl_map_flags     flags,
               size_t           size);
/*

*/
  int unmemMap(cl_command_queue cmd_queue,
               cl_mem           dev_mem,
               void*            host_ptr);

#endif /* BUFFER_OBJECTS_H */
