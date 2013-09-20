#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#include "../cl_helper.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

double shrDeltaT()
{
#if defined(_WIN32)
	static LARGE_INTEGER old_time;
	LARGE_INTEGER new_time, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&new_time);
	const double DeltaT = ((double)new_time.QuadPart - (double)old_time.QuadPart) / (double)freq.QuadPart;
	old_time = new_time;
	return DeltaT;
#elif defined(__unix__)
	static struct timeval old_time;
	struct timeval new_time;
	gettimeofday(&new_time, NULL);
	const double DeltaT = ((double)new_time.tv_sec + 1.0e-6 * (double)new_time.tv_usec) - ((double)old_time.tv_sec + 1.0e-6 * (double)old_time.tv_usec);
	old_time.tv_sec  = new_time.tv_sec;
	old_time.tv_usec = new_time.tv_usec;
	return DeltaT;
#elif defined (__APPLE__) || defined (MACOSX)
	static time_t old_time;
	time_t new_time;
	new_time  = clock();
	const double DeltaT = double(new_time - old_time) / CLOCKS_PER_SEC;
	old_time.tv_sec  = new_time.tv_sec;
	old_time.tv_usec = new_time.tv_usec;
	return DeltaT;
#else
	return 0;
#endif
}

int main(int argc, char* argv[])
{
	const int n = 4;
	const size_t sizes[n] = { 3 << 10, 15 << 10, 15 << 20, 100 << 20 };
	const int iterations[n] = { 60000, 60000, 300, 30 };
	printf("device,platform name,device name,size (B),memory,access,direction,time (s),bandwidth (MB/s)\n");
	int g = 0;
	cl_int error;
	cl_uint num_platforms;
	checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	checkOclErrors(clGetPlatformIDs(num_platforms, platforms, NULL));
	for (cl_uint p = 0; p < num_platforms; ++p)
	{
		cl_platform_id platform = platforms[p];
		char platform_name[256];
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL));
//		if (strcmp(platform_name, "NVIDIA CUDA")) continue;
		cl_uint num_devices;
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
		cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));
		for (cl_uint d = 0; d < num_devices; ++d, ++g)
		{
			cl_device_id device = devices[d];
			char device_name[256];
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL));
			cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
			checkOclErrors(error);
			cl_command_queue command_queue = clCreateCommandQueue(context, device, 0/*CL_QUEUE_PROFILING_ENABLE*/, &error);
			checkOclErrors(error);
			for (int s = 0; s < n; ++s)
			{
				const size_t size = sizes[s];
				const int iteration = iterations[s];
				const double bandwidth_unit = (double)size * iteration / (1 << 20);
				void* h_p = NULL;
				void* d_p = NULL;
				cl_mem h_b;
				cl_mem d_b;
				double time;
				double bandwidth;
				d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error);
				checkOclErrors(error);

				// allocate pageable h_p
				h_p = malloc(size);
				// --memory=pageable --access=direct --htod
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueWriteBuffer(command_queue, d_b, CL_FALSE, 0, size, h_p, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "direct", "HtoD", time, bandwidth);
				// --memory=pageable --access=direct --dtoh
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueReadBuffer(command_queue, d_b, CL_FALSE, 0, size, h_p, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "direct", "DtoH", time, bandwidth);
				// --memory=pageable --access=mapped --htod
				shrDeltaT();
				d_p = clEnqueueMapBuffer(command_queue, d_b, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; i++)
				{
					memcpy(d_p, h_p, size);
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_b, d_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "mapped", "HtoD", time, bandwidth);
				// --memory=pageable --access=mapped --dtoh
				shrDeltaT();
				d_p = clEnqueueMapBuffer(command_queue, d_b, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; i++)
				{
					memcpy(h_p, d_p, size);
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_b, d_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "mapped", "DtoH", time, bandwidth);
				// --memory=pageable --access=copy --htod
				h_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, h_p, &error);
				checkOclErrors(error);
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueCopyBuffer(command_queue, h_b, d_b, 0, 0, size, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "copy", "HtoD", time, bandwidth);
				// --memory=pageable --access=copy --dtoh
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueCopyBuffer(command_queue, d_b, h_b, 0, 0, size, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pageable", "copy", "DtoH", time, bandwidth);
				checkOclErrors(clReleaseMemObject(h_b));
				// deallocate pageable h_p
				free(h_p);

				// allocate pinned h_b
				h_b = clCreateBuffer(context, /*CL_MEM_READ_WRITE | */CL_MEM_ALLOC_HOST_PTR, size, NULL, &error);
				checkOclErrors(error);
				// --memory=pinned --access=direct --htod
				shrDeltaT();
				h_p = clEnqueueMapBuffer(command_queue, h_b, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueWriteBuffer(command_queue, d_b, CL_FALSE, 0, size, h_p, 0, NULL, NULL));
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, h_b, h_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "direct", "HtoD", time, bandwidth);
				// --memory=pinned --access=direct --dtoh
				shrDeltaT();
				h_p = clEnqueueMapBuffer(command_queue, h_b, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueReadBuffer(command_queue, d_b, CL_FALSE, 0, size, h_p, 0, NULL, NULL));
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, h_b, h_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "direct", "DtoH", time, bandwidth);
				// --memory=pinned --access=mapped --htod
				shrDeltaT();
				h_p = clEnqueueMapBuffer(command_queue, h_b, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				d_p = clEnqueueMapBuffer(command_queue, d_b, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; i++)
				{
					memcpy(d_p, h_p, size);
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_b, d_p, 0, NULL, NULL));
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, h_b, h_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "mapped", "HtoD", time, bandwidth);
				// --memory=pinned --access=mapped --dtoh
				shrDeltaT();
				h_p = clEnqueueMapBuffer(command_queue, h_b, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				d_p = clEnqueueMapBuffer(command_queue, d_b, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < iteration; i++)
				{
					memcpy(h_p, d_p, size);
				}
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, d_b, d_p, 0, NULL, NULL));
				checkOclErrors(clEnqueueUnmapMemObject(command_queue, h_b, h_p, 0, NULL, NULL));
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "mapped", "DtoH", time, bandwidth);
				// --memory=pinned --access=copy --htod
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueCopyBuffer(command_queue, h_b, d_b, 0, 0, size, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "copy", "HtoD", time, bandwidth);
				// --memory=pinned --access=copy --dtoh
				shrDeltaT();
				for (int i = 0; i < iteration; ++i)
				{
					checkOclErrors(clEnqueueCopyBuffer(command_queue, d_b, h_b, 0, 0, size, 0, NULL, NULL));
				}
				checkOclErrors(clFinish(command_queue));
				time = shrDeltaT();
				bandwidth = bandwidth_unit / time;
				printf("%d,%s,%s,%lu,%s,%s,%s,%.3f,%.0f\n", g, platform_name, device_name, size, "pinned", "copy", "DtoH", time, bandwidth);
				// deallocate pinned h_b
				checkOclErrors(clReleaseMemObject(h_b));

				checkOclErrors(clReleaseMemObject(d_b));
			}
			checkOclErrors(clReleaseCommandQueue(command_queue));
			checkOclErrors(clReleaseContext(context));
		}
		free(devices);
	}
	free(platforms);
}
