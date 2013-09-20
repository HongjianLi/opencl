#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
//#include <CL/cl_ext.h>
#include "../cl_helper.h"
#include "vectorAdd.h"
/*
void CL_CALLBACK input_map_event_callback(cl_event event, cl_int command_exec_status, void* data)
{
//	assert(command_exec_status == CL_COMPLETE);
	printf("input_map_event_callback\n");
	memset(p_s, 0, sizeof(cl_float) * gws);
	checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_s, p_s, 0, NULL, &input_events[0]));
	checkOclErrors(error);
}
*/
void CL_CALLBACK output_map_event_callback(cl_event event, cl_int command_exec_status, void* data)
{
//	assert(command_exec_status == CL_COMPLETE);
	printf("output_map_event_callback, p_e[0] = %f\n", *(float*)data);
}

int main(int argc, char* argv[])
{
	const size_t lws = 256;
	const size_t gws = 32 * lws;
	cl_float* p_l = (cl_float*)malloc(sizeof(cl_float) * lws);
	for (int i = 0; i < lws; ++i)
	{
		p_l[i] = rand() / (float)RAND_MAX;
	}

	cl_int error;
	char buffer[256];
	cl_uint num_platforms;
	checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	checkOclErrors(clGetPlatformIDs(num_platforms, platforms, NULL)); // AMD APP and Intel OpenCL fail valgrind --leak-check
	for (cl_uint i = 0; i < num_platforms; ++i)
	{
		cl_platform_id platform = platforms[i];
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
		printf("CL_PLATFORM_NAME: %s\n\n", buffer);

		cl_uint num_devices;
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

		cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &error);
		checkOclErrors(error);
		const size_t source_length = strlen(sources[0]);
		cl_program program = clCreateProgramWithSource(context, 1, sources, &source_length, &error);
		checkOclErrors(error);

		checkOclErrors(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math -cl-denorms-are-zero"/*-cl-std=CL1.2*/, NULL, NULL)); // Append -cl-nv-maxrregcount <N> to build options for NVIDIA CUDA. Append -g -s vectorAdd.cl to build options to enable debugging for Intel OpenCL. Phi does not support -cl-denorms-are-zero but supports -auto-prefetch-level=[0-3]. AMD APP, NVIDIA CUDA and Intel OpenCL fail valgrind --leak-check.
//		checkOclErrors(clCompileProgram(program, 0, NULL, "-cl-fast-relaxed-math -cl-denorms-are-zero", 0, NULL, NULL, NULL, NULL));
//		program = clLinkProgram(context, 0, NULL, NULL, 1, &program, NULL, NULL, &error);
//		checkOclErrors(error);
		cl_kernel kernel = clCreateKernel(program, "vectorAdd", &error);
		checkOclErrors(error);

		for (int i = 0; i < num_devices; ++i)
		{
			cl_device_id device = devices[i];
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			printf("CL_DEVICE_NAME: %s\n", buffer);
//			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL));
//			cl_bool cl12 = !strcmp(buffer, "OpenCL C 1.2");
			cl_bool host_unified_memory;
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(host_unified_memory), &host_unified_memory, NULL));

			cl_build_status build_status;
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL));
			if (build_status == CL_BUILD_SUCCESS) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_SUCCESS");
			if (build_status == CL_BUILD_NONE) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_NONE");
			if (build_status == CL_BUILD_ERROR) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_ERROR");
			if (build_status == CL_BUILD_IN_PROGRESS) printf("CL_PROGRAM_BUILD_STATUS: %s\n", "CL_BUILD_IN_PROGRESS");
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, NULL));
			printf("CL_PROGRAM_BUILD_OPTIONS: %s\n", buffer);
//			cl_program_binary_type binary_type;
//			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BINARY_TYPE, sizeof(binary_type), &binary_type, NULL));
//			if (binary_type == CL_PROGRAM_BINARY_TYPE_NONE) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_NONE");
//			if (binary_type == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT");
//			if (binary_type == CL_PROGRAM_BINARY_TYPE_LIBRARY) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_LIBRARY");
//			if (binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) printf("CL_PROGRAM_BINARY_TYPE: %s\n", "CL_PROGRAM_BINARY_TYPE_EXECUTABLE");
			size_t build_log_size;
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size));
			char* build_log = (char*)malloc(build_log_size);
			checkOclErrors(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL));
			printf("%s\n", build_log);
			free(build_log);

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
//			size_t global_work_size[3];
//			checkOclErrors(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(size_t) * 3, global_work_size, NULL));
//			printf("CL_KERNEL_GLOBAL_WORK_SIZE: %lu, %lu, %lu\n", global_work_size[0], global_work_size[1], global_work_size[2]);

			cl_command_queue_properties queue_properties;
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL));
			cl_command_queue command_queue = clCreateCommandQueue(context, device, queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0/* | CL_QUEUE_PROFILING_ENABLE*/, &error);
			checkOclErrors(error);
			cl_event input_events[2];
//			if (cl12)
//			{
//				const cl_float pattern = 0.0f;
//				checkOclErrors(clEnqueueFillBuffer(command_queue, d_s, &pattern, sizeof(pattern), 0, sizeof(cl_float) * gws, 0, NULL, &input_events[0]));
//			}
//			else
//			{
			cl_mem d_s = clCreateBuffer(context, CL_MEM_READ_WRITE/* | CL_MEM_USE_PERSISTENT_MEM_AMD*/, sizeof(cl_float) * gws, NULL, &error);
			checkOclErrors(error);
			cl_float* q_s = (cl_float*)clEnqueueMapBuffer(command_queue, d_s, CL_TRUE, CL_MAP_WRITE/*_INVALIDATE_REGION*/, 0, sizeof(cl_float) * gws, 0, NULL, NULL, &error);
			memset(q_s, 0, sizeof(cl_float) * gws);
			checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_s, q_s, 0, NULL, &input_events[0]));
			checkOclErrors(error);
//			}
			cl_mem d_l = clCreateBuffer(context, CL_MEM_READ_ONLY/* | CL_MEM_USE_PERSISTENT_MEM_AMD*/,  sizeof(cl_float) * lws, NULL, &error);
			checkOclErrors(error);
			cl_float* q_l = (cl_float*)clEnqueueMapBuffer(command_queue, d_l, CL_TRUE, CL_MAP_WRITE/*_INVALIDATE_REGION*/, 0, sizeof(cl_float) * lws, 0, NULL, NULL, &error);
			memcpy(q_l, p_l, sizeof(cl_float) * lws);
			checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_l, q_l, 0, NULL, &input_events[1]));
			checkOclErrors(error);
//			cl_mem h_l = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lws, p_l, &error);
//			checkOclErrors(error);
//			checkOclErrors(clEnqueueCopyBuffer(command_queue, h_l, d_l, 0, 0, sizeof(cl_float) * lws, 0, NULL, &input_events[1]));
			checkOclErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_s));
			checkOclErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_l));
			checkOclErrors(clSetKernelArg(kernel, 2, sizeof(cl_float) * lws, NULL));
			cl_event kernel_event;
			checkOclErrors(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &gws, &lws/*NULL*/, 2, input_events, &kernel_event));
//			checkOclErrors(clEnqueueReadBuffer(command_queue, d_c, CL_FALSE, 0, sizeof(cl_float) * gws, h_c, 0, NULL, NULL));
//			checkOclErrors(clFinish(command_queue));
			cl_mem h_e = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_float) * lws, NULL, &error);
			checkOclErrors(error);
			cl_event output_event;
			checkOclErrors(clEnqueueCopyBuffer(command_queue, d_s, h_e, 0, 0, sizeof(cl_float) * lws, 1, &kernel_event, &output_event));
			cl_event output_map_event;
			cl_float* p_e = (cl_float*)clEnqueueMapBuffer(command_queue, h_e, CL_FALSE, CL_MAP_READ, 0, sizeof(cl_float) * lws, 1, &output_event, &output_map_event, &error);
			checkOclErrors(error);
//			checkOclErrors(clEnqueueReadBuffer(command_queue, d_s, CL_FALSE, 0, sizeof(cl_float) * lws, h_c, 1, &kernel_event, &output_event));
			checkOclErrors(clSetEventCallback(output_map_event, CL_COMPLETE, output_map_event_callback, p_e));
			// Unmap p_e
			checkOclErrors(clFlush(command_queue));
//			cl_int command_exec_status;
//			checkOclErrors(clGetEventInfo(output_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &command_exec_status, NULL));
//			if (command_exec_status == CL_COMPLETE);
			checkOclErrors(clWaitForEvents(1, &output_map_event));
			bool passed = true;
			for (int i = 0; i < lws; ++i)
			{
				const float ref = p_l[i] * 2.0f + 1.0f;
				if (fabs(p_e[i] - ref) > 1e-7)
				{
					printf("i = %d, ref = %f, p_e[i] = %f\n", i, ref, p_e[i]);
					passed = false;
					break;
				}
			}
			printf("vectorAdd %s\n\n", passed ? "passed" : "failed");
			checkOclErrors(clEnqueueUnmapMemObject(command_queue, h_e, p_e, 0, NULL, NULL));
			checkOclErrors(clReleaseMemObject(d_l));
//			checkOclErrors(clReleaseMemObject(h_l));
			checkOclErrors(clReleaseMemObject(d_s));
			checkOclErrors(clReleaseCommandQueue(command_queue));
		}
		checkOclErrors(clReleaseKernel(kernel));
		checkOclErrors(clReleaseProgram(program));
		checkOclErrors(clReleaseContext(context));
		free(devices);
	}
	free(platforms);
	free(p_l);
}
