#include "BufferObjects.h"

int mallocBufferObject(cl_context   context,
                       cl_mem_flags flags,
                       size_t       size,
		       cl_mem*      mem){
  cl_mem local_mem;

  local_mem = clCreateBuffer(context, flags, size, NULL, NULL);
  if (!local_mem){
	printf("Error: clCreateBuffer()\n");
	fflush(stdout);
	return EXIT_FAILURE;
  }

#ifdef BUFFER_OBJECT_INFO_DISPLAY
  cl_int err;
  err = getBufferObjectInfo(local_mem);
  if (err != CL_SUCCESS){
	printf("Error: getBufferObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }
#endif

  *mem = local_mem;
  return CL_SUCCESS;
}

#ifdef BUFFER_OBJECT_INFO_DISPLAY
int getBufferObjectInfo(cl_mem object){

  cl_int                 err;

  cl_mem_object_type*	 mem_object_type;
  cl_mem_flags*		 mem_flags;
  void**                 mem_host_ptr;	      // host ptr that references the memory object's data
  size_t*		 mem_size;
  cl_context*	         mem_context;
  /*
  cl_mem*   		 mem_assoc_memobject; // only valid for subbuffer objects
  size_t*		 mem_offset;          // only valid for subbuffer objects
  */
  cl_uint*		 mem_ref_count;

  size_t sizeParam;

  // ----------------------------------------------------------------------------
  // Query Mem object types
  printf("\n-----------------------------------------------------------------------\n");
  err = clGetMemObjectInfo(object,CL_MEM_TYPE,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_object_type = (cl_mem_object_type*) malloc(sizeof(cl_mem_object_type) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_TYPE,sizeParam,mem_object_type,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %s \n", "CL_MEM_TYPE", (*mem_object_type == 0x10F0)? "CL_MEM_OBJECT_BUFFER":(*mem_object_type == 0x10F1)? "CL_MEM_OBJECT_IMAGE2D":(*mem_object_type == 0x10F2)? "CL_MEM_OBJECT_IMAGE3D":"UNKNOWN");  
  free(mem_object_type);

  // ----------------------------------------------------------------------------
  // Query Flags used to configure the memory object's accessibility and allocation
  err = clGetMemObjectInfo(object,CL_MEM_FLAGS,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_flags = (cl_mem_flags*) malloc(sizeof(cl_mem_flags) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_FLAGS,sizeParam,mem_flags,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %lx \n", "CL_MEM_FLAGS (bit-field) ", *mem_flags);
  free(mem_flags);

  // ----------------------------------------------------------------------------
  // Query host pointer that references the memory object's data
  err = clGetMemObjectInfo(object,CL_MEM_HOST_PTR,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_host_ptr = (void**) malloc(sizeof(void*) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_HOST_PTR,sizeParam,*mem_host_ptr,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %x \n", "CL_MEM_HOST_PTR", *mem_host_ptr);
  free(mem_host_ptr);

  // ----------------------------------------------------------------------------
  // Query Mem object size
  err = clGetMemObjectInfo(object,CL_MEM_SIZE,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_size = (size_t*) malloc(sizeof(size_t) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_SIZE,sizeParam,mem_size,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %lu \n", "CL_MEM_SIZE", *mem_size);
  free(mem_size);

  // ----------------------------------------------------------------------------
  // Query Context associated with memory object
  err = clGetMemObjectInfo(object,CL_MEM_CONTEXT,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_context = (cl_context*) malloc(sizeof(cl_context) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_CONTEXT,sizeParam,mem_context,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %x \n", "CL_MEM_CONTEXT", *mem_context);
  free(mem_context);

  // ----------------------------------------------------------------------------
  // Query Mem associated to mem object (only valid for subbuffers)


  // ----------------------------------------------------------------------------
  // Query Mem object offset (only valid for subbuffers)
  /*
  err = clGetMemObjectInfo(object,CL_MEM_OFFSET,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){printf("Error: clGetMemObjectInfo() %d\n",err); return EXIT_FAILURE;}

  mem_offset = (size_t*) malloc(sizeof(size_t) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_OFFSET,sizeParam,mem_offset,NULL);
  if (err != CL_SUCCESS){printf("Error: clGetMemObjectInfo() %d\n",err); return EXIT_FAILURE;}

  printf("%-30s: %u \n", "Offset of memory object: ", *mem_offset);
  free(mem_offset);
  */

  // ----------------------------------------------------------------------------
  // Query Memory object reference count
  err = clGetMemObjectInfo(object,CL_MEM_REFERENCE_COUNT,0,NULL,&sizeParam);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  mem_ref_count = (cl_uint*) malloc(sizeof(cl_uint) * sizeParam);
  err = clGetMemObjectInfo(object,CL_MEM_REFERENCE_COUNT,sizeParam,mem_ref_count,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clGetMemObjectInfo() %d\n",err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  printf("  %-45s: %u \n", "CL_MEM_REFERENCE_COUNT", *mem_ref_count);
  free(mem_ref_count);

  return CL_SUCCESS;
}
#endif


int memcopyBufferObjectToDevice(cl_command_queue cmd_queue,
			        cl_mem           dest,
			        void*            src,
			        size_t           size){
  cl_int err;
  err = clEnqueueWriteBuffer(cmd_queue,dest,CL_TRUE,0,size,src,0,NULL,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clEnqueueWriteBuffer() %d\n", err);
	fflush(stdout);
	return EXIT_FAILURE;
  }

  // Mem copy to Device is blocking!!!

  return CL_SUCCESS;
}

int memcopyBufferObjectFromDevice(cl_command_queue cmd_queue,
                                  void*            dest,
			          cl_mem           src,
			          size_t           size){
  cl_int err;
  err = clEnqueueReadBuffer(cmd_queue,src,CL_TRUE,0,size,dest,0,NULL,NULL);
  if (err != CL_SUCCESS){
	printf("Error: clEnqueueReadBuffer() %d\n", err);
	fflush(stdout);
	return EXIT_FAILURE;
  }
 
  // Mem copy from Device is blocking!!!

  return CL_SUCCESS;
}

int memcopyBufferObjectToBufferObject(cl_command_queue cmd_queue,
				      cl_mem     dest,
				      cl_mem     src,
				      size_t     size){
   cl_int err;
   err = clEnqueueCopyBuffer(cmd_queue, 
			     src,
			     dest,
			     0,		/* size_t src_offset */
			     0,		/* size_t dst_offset */
			     size,
			     0,		/* cl_uint num_events_in_wait_list */
			     NULL,	/* const cl_event *event_wait_list */
			     NULL);	/* cl_event *event */
   if (err != CL_SUCCESS){
 	printf("Error: clEnqueueCopyBuffer() %d\n", err);
	fflush(stdout);
	return EXIT_FAILURE;
   }
  
   // Mem copy from Buffer to Buffer is blocking!!!

   // clFinish commented out, as this causes slow down
   // Our command queue is in-order and this is not needed
   //clFinish(cmd_queue);

   return CL_SUCCESS;
}

void* memMap(cl_command_queue cmd_queue,
	   cl_mem           dev_mem,
	   cl_map_flags     flags,
	   size_t           size){

   cl_int err;
   void*  local_host_mem;

   local_host_mem = clEnqueueMapBuffer(cmd_queue,
			               dev_mem,
			               CL_TRUE,
			               flags,
			               0,
			               size,
			               0,
			               NULL,
			               NULL,
			               &err);

   if (err != CL_SUCCESS || !local_host_mem){
 	printf("Error: clEnqueueMapBuffer() %d\n", err);
	fflush(stdout);
	return NULL;
   }

   // Mem copy from Buffer to Buffer is blocking!!!

   // clFinish commented out, as this causes slow down
   // Our command queue is in-order and this is not needed
   //clFinish(cmd_queue);

   return local_host_mem;
}

int unmemMap(cl_command_queue cmd_queue,
	     cl_mem           dev_mem,
	     void*            host_ptr){

   cl_int err;
   err = clEnqueueUnmapMemObject(cmd_queue,
				 dev_mem,
				 host_ptr,
				 0,
				 NULL,
				 NULL);

   if (err != CL_SUCCESS){
        printf("Error: clEnqueueUnmapMemObjetc() %d\n", err);
        fflush(stdout);
        return EXIT_FAILURE;
   }

   return CL_SUCCESS;
}
