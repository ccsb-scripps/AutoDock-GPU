/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/


#include "ImportSource.h"

/* convert the kernel file into a string */
int convertToString2(
                     const char*        filename,
                           std::string& s
                    )
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

int ImportSourceToProgram(
                          const char*         filename,
                                cl_device_id* device_id,
                                cl_context    context,
                                cl_program*   program,
                          const char*         options)
{
	cl_int err;

	// Added as kernel is stringified already
	const char *source = filename;

	// OCLADock
	size_t sourceSize[] = { strlen(source) };


	*program = clCreateProgramWithSource(context, 1, &source, sourceSize, &err);
		
	if ((!*program) || (err != CL_SUCCESS)){
		printf("Error: clCreateProgramWithBinary() %d\n", err);
		fflush(stdout);
		return EXIT_FAILURE;
	}

#ifdef PROGRAM_INFO_DISPLAY
	err = getProgramInfo(*program);
	if (err != CL_SUCCESS){
		printf("Error: getProgramInfo() \n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	// Build the program executable
	err = clBuildProgram(*program, 1, device_id, options, NULL, NULL);
	
	if (err != CL_SUCCESS) {
		int    err_build;
		size_t sizeParam;
		char*  program_build_log;

		err_build = clGetProgramBuildInfo(*program,device_id[0],CL_PROGRAM_BUILD_LOG,0,NULL,&sizeParam);
		if (err_build != CL_SUCCESS){
			printf("Error: clGetProgramBuildInfo() %d\n",err_build);
			fflush(stdout);
			return EXIT_FAILURE;
		}

		program_build_log = (char*)malloc(sizeof(char) * sizeParam);
		err_build = clGetProgramBuildInfo(*program, device_id[0], CL_PROGRAM_BUILD_LOG, sizeParam, program_build_log, NULL);
		if (err_build != CL_SUCCESS){
			printf("Error: clGetProgramBuildInfo() %d\n", err_build);
			fflush(stdout);
			return EXIT_FAILURE;
		}
		printf("  %-45s: %s \n", "CL_PROGRAM_BUILD_LOG", program_build_log);
		fflush(stdout);
		
		FILE* plogfile;
		plogfile = fopen("KernelProgramBuildInfo.txt","w");
		fprintf(plogfile,"%s",program_build_log);

		fclose(plogfile);
		free(program_build_log);
		return EXIT_FAILURE;
	}

#ifdef PROGRAM_BUILD_INFO_DISPLAY
	err = getprogramBuildInfo(*program, device_id[0]);
	if (err != CL_SUCCESS){
		printf("Error: getprogramBuildInfo() \n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
#endif

	return CL_SUCCESS;
}
