#include "ImportSource.h"

/* convert the kernel file into a string */
int convertToString2(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return EXIT_FAILURE;
}

int ImportSource(const char*    filename,
			     const char*    kernel_name,
				 cl_device_id*  device_id,
				 cl_context	context,
				 /*cl_program*	program,*/
				 const char*    options,
				 cl_kernel*	kernel)
{
	cl_int err;

	// Create the compute program ONLINE
	string sourceStr;
	err = convertToString2(filename, sourceStr);
	const char *source = sourceStr.c_str();

	// L30nardoSV
	size_t sourceSize[] = { strlen(source) };

	cl_program local_program;
	local_program = clCreateProgramWithSource(context, 1, &source, sourceSize, &err);
		
	if ((!local_program) || (err != CL_SUCCESS)){
		printf("Error: clCreateProgramWithBinary() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef PROGRAM_INFO_DISPLAY
	err = getProgramInfo(local_program);
	if (err != CL_SUCCESS){
		printf("Error: getProgramInfo() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	/*Step 6: Build program. */
	// Build the program executable
	err = clBuildProgram(local_program, 1, device_id, options, NULL, NULL);

	/*
	if (err != CL_SUCCESS){
		size_t len;
		char buffer[2048];
		printf("Error: clBuildProgram() %d\n", err);
		clGetProgramBuildInfo(local_program, device_id[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		fflush(stdout);
		return EXIT_FAILURE;
	}
	*/
	
	if (err != CL_SUCCESS) {
		int    err_build;
		size_t sizeParam;
		char*  program_build_log;

		err_build = clGetProgramBuildInfo(local_program,device_id[0],CL_PROGRAM_BUILD_LOG,0,NULL,&sizeParam);
		if (err_build != CL_SUCCESS){
			printf("Error: clGetProgramBuildInfo() %d\n",err_build);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		program_build_log = (char*)malloc(sizeof(char) * sizeParam);
		err_build = clGetProgramBuildInfo(local_program, device_id[0], CL_PROGRAM_BUILD_LOG, sizeParam, program_build_log, NULL);
		if (err_build != CL_SUCCESS){
			printf("Error: clGetProgramBuildInfo() %d\n", err_build);
			fflush(stdout);
			return EXIT_FAILURE;
		}
		printf("  %-45s: %s \n", "CL_PROGRAM_BUILD_LOG", program_build_log);
		fflush(stdout);
		
		FILE* plogfile;
		plogfile = fopen("README_LOG_ProgramBuildInfo.txt","w");
		fprintf(plogfile,"%s",program_build_log);

		fclose(plogfile);
		free(program_build_log);
		return EXIT_FAILURE;
	}

	








#ifdef PROGRAM_BUILD_INFO_DISPLAY
	err = getprogramBuildInfo(local_program, device_id[0]);
	if (err != CL_SUCCESS){
		printf("Error: getprogramBuildInfo() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	// Create the compute kernel in the program we wish to run
	cl_kernel local_kernel;
	local_kernel = clCreateKernel(local_program, kernel_name, &err);

	if ((!local_kernel) || (err != CL_SUCCESS)){
		printf("Error: clCreateKernel() %s %d\n", kernel_name, err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef KERNEL_INFO_DISPLAY
	err = getKernelInfo(local_kernel);
	if (err != CL_SUCCESS){
		printf("Error: getKernelInfo() %d\n", kernel_name, err);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

#ifdef KERNEL_WORK_GROUP_INFO_DISPLAY
	err = getKernelWorkGroupInfo(local_kernel, device_id[0]);
	if (err != CL_SUCCESS){
		printf("Error: getKernelWorkGroupInfo() %d\n", kernel_name, err);
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	/* *program = local_program;*/
	*kernel = local_kernel;
	return CL_SUCCESS;
}
