#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "../cl_helper.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("jit vectorAdd.cl\n");
		return 0;
	}

	FILE* source_file = fopen(argv[1], "rb");
	fseek(source_file, 0, SEEK_END);
	const size_t source_length = ftell(source_file);
	fseek(source_file, 0, SEEK_SET);
	char* const source = (char*)malloc(sizeof(char) * source_length);
	fread(source, sizeof(char), source_length, source_file);
	fclose(source_file);

	cl_int error;
	char buffer[256];
	cl_uint num_platforms;
	checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	checkOclErrors(clGetPlatformIDs(num_platforms, platforms, NULL));
	for (cl_uint k = 0; k < num_platforms; ++k)
	{
		cl_platform_id platform = platforms[k];
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
		printf("CL_PLATFORM_NAME: %s\n\n", buffer);

		cl_uint num_devices;
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

		cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &error);
		checkOclErrors(error);
		cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_length, &error);
		checkOclErrors(error);

		checkOclErrors(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math -cl-denorms-are-zero", NULL, NULL));
//		checkOclErrors(clCompileProgram(program, 0, NULL, "-cl-fast-relaxed-math -cl-denorms-are-zero", 0, NULL, NULL, NULL, NULL));
//		program = clLinkProgram(context, 0, NULL, NULL, 1, &program, NULL, NULL, &error);
//		checkOclErrors(error);

		for (int j = 0; j < num_devices; ++j)
		{
			cl_device_id device = devices[j];
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			printf("CL_DEVICE_NAME: %s\n", buffer);

			cl_build_status build_status;
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL));
			if (build_status == CL_BUILD_SUCCESS) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_SUCCESS");
			if (build_status == CL_BUILD_NONE) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_NONE");
			if (build_status == CL_BUILD_ERROR) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_ERROR");
			if (build_status == CL_BUILD_IN_PROGRESS) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_IN_PROGRESS");
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, NULL));
			printf("CL_PROGRAM_BUILD_OPTIONS: %s\n", buffer);
			cl_program_binary_type binary_type;
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BINARY_TYPE, sizeof(binary_type), &binary_type, NULL));
			if (binary_type == CL_PROGRAM_BINARY_TYPE_NONE) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_NONE");
			if (binary_type == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT");
			if (binary_type == CL_PROGRAM_BINARY_TYPE_LIBRARY) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_LIBRARY");
			if (binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_EXECUTABLE");
			size_t build_log_size;
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size));
			char* build_log = (char*)malloc(build_log_size);
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL));
			printf("%s\n", build_log);
			free(build_log);
			if (build_status != CL_BUILD_SUCCESS) continue;

			cl_uint num_kernels;
			checkOclErrors(clCreateKernelsInProgram(program, 0, NULL, &num_kernels));
			cl_kernel* kernels = (cl_kernel*)malloc(sizeof(cl_kernel) * num_kernels);
			checkOclErrors(clCreateKernelsInProgram(program, num_kernels, kernels, NULL));
			for (int i = 0; i < num_kernels; ++i)
			{
				cl_kernel kernel = kernels[i];
				checkOclErrors(clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(buffer), buffer, NULL));
				printf("CL_KERNEL_FUNCTION_NAME: %s\n", buffer);
				cl_uint num_args;
				checkOclErrors(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL));
				printf("CL_KERNEL_NUM_ARGS: %d\n", num_args);
				for (int a = 0; a < num_args; ++a)
				{
					printf("%2d", a);
					static const char* const address_qualifiers[] = { "global", "local", "constant", "private" };
					cl_kernel_arg_address_qualifier address_qualifier;
					checkOclErrors(clGetKernelArgInfo(kernel, a, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(address_qualifier), &address_qualifier, NULL));
					printf(" %s", address_qualifiers[address_qualifier - CL_KERNEL_ARG_ADDRESS_GLOBAL]);
//					static const char* const access_qualifiers[] = { "READ_ONLY", "WRITE_ONLY", "READ_WRITE", "NONE" };
//					cl_kernel_arg_access_qualifier access_qualifier;
//					checkOclErrors(clGetKernelArgInfo(kernel, a, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(access_qualifier), &access_qualifier, NULL));
//					printf("CL_KERNEL_ARG_ACCESS_QUALIFIER: %s\n", access_qualifiers[access_qualifier - CL_KERNEL_ARG_ACCESS_READ_ONLY]);
					checkOclErrors(clGetKernelArgInfo(kernel, a, CL_KERNEL_ARG_TYPE_NAME, sizeof(buffer), buffer, NULL));
					printf(" %s", buffer);
					cl_kernel_arg_type_qualifier type_qualifier;
					checkOclErrors(clGetKernelArgInfo(kernel, a, CL_KERNEL_ARG_TYPE_QUALIFIER, sizeof(type_qualifier), &type_qualifier, NULL));
					if (type_qualifier & CL_KERNEL_ARG_TYPE_CONST) printf(" const");
					if (type_qualifier & CL_KERNEL_ARG_TYPE_RESTRICT) printf(" restrict");
					if (type_qualifier & CL_KERNEL_ARG_TYPE_VOLATILE) printf(" volatile");
					checkOclErrors(clGetKernelArgInfo(kernel, a, CL_KERNEL_ARG_NAME, sizeof(buffer), buffer, NULL));
					printf(" %s\n", buffer);
				}

				cl_ulong local_mem_size;
				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL));
				printf("CL_KERNEL_LOCAL_MEM_SIZE: %lu\n", local_mem_size);
				cl_ulong private_mem_size;
				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &private_mem_size, NULL));
				printf("CL_KERNEL_PRIVATE_MEM_SIZE: %lu\n", private_mem_size);
				size_t work_group_size;
				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL));
				printf("CL_KERNEL_WORK_GROUP_SIZE: %lu\n", work_group_size);
				size_t preferred_work_group_size_multiple;
				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_work_group_size_multiple, NULL));
				printf("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu\n", preferred_work_group_size_multiple);
				size_t compile_work_group_size[3];
				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t) * 3, compile_work_group_size, NULL));
				printf("CL_KERNEL_COMPILE_WORK_GROUP_SIZE: %lu, %lu, %lu\n", compile_work_group_size[0], compile_work_group_size[1], compile_work_group_size[2]);
//				size_t global_work_size[3];
//				checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(size_t) * 3, global_work_size, NULL));
//				printf("CL_KERNEL_GLOBAL_WORK_SIZE: %lu, %lu, %lu\n", global_work_size[0], global_work_size[1], global_work_size[2]);
			}
			free(kernels);
			printf("\n");
		}
		free(devices);
	}
	free(platforms);
	free(source);
}
