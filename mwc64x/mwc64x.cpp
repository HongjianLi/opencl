#include <vector>
#include <memory>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <CL/cl.h>
#include "../cl_helper.h"
using namespace std;

class callback_data
{
public:
	cl_ulong* acch;
	ulong n;
	size_t gws;
	cl_event e;
	callback_data(cl_ulong* acch, ulong n, size_t gws, cl_event e) : acch(acch), n(n), gws(gws), e(e) {}
};

int main(int argc, char* argv[])
{
	const size_t lws = 256;
	const size_t gws = 32 * lws;
	const ulong n = 1 << 30;
	const ulong baseOffset = 0;

	ifstream ifs("mwc64x.cl");
	string image((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
	const char* sources[] = { image.data() };
	const size_t lengths = image.size();

	cl_int error;
	char buffer[256];
	cl_uint num_platforms;
	checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
	vector<cl_platform_id> platforms(num_platforms);
	checkOclErrors(clGetPlatformIDs(num_platforms, platforms.data(), NULL));
	for (const auto platform : platforms)
	{
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
		printf("CL_PLATFORM_NAME: %s\n", buffer);

		cl_uint num_devices;
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
		vector<cl_device_id> devices(num_devices);
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL));
		for (const auto device : devices)
		{
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			printf("CL_DEVICE_NAME: %s\n", buffer);

			cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
			checkOclErrors(error);
			cl_program program = clCreateProgramWithSource(context, 1, sources, &lengths, &error);
			checkOclErrors(error);

			checkOclErrors(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL));
			cl_kernel kernel = clCreateKernel(program, "EstimatePi", &error);
			checkOclErrors(error);

			cl_command_queue_properties queue_properties;
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL));
			cl_command_queue command_queue = clCreateCommandQueue(context, device, queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0, &error);
			checkOclErrors(error);

			cl_mem accd = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * gws, NULL, &error);
			checkOclErrors(error);
			checkOclErrors(clSetKernelArg(kernel, 0, sizeof(ulong), &n));
			checkOclErrors(clSetKernelArg(kernel, 1, sizeof(ulong), &baseOffset));
			checkOclErrors(clSetKernelArg(kernel, 2, sizeof(cl_mem), &accd));
			cl_event kernel_event;
			checkOclErrors(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &gws, &lws, 0, NULL, &kernel_event));
			cl_event output_map_event;
			cl_ulong* acch = (cl_ulong*)clEnqueueMapBuffer(command_queue, accd, CL_FALSE, CL_MAP_READ, 0, sizeof(cl_ulong) * gws, 1, &kernel_event, &output_map_event, &error);
			checkOclErrors(error);
			cl_event e = clCreateUserEvent(context, &error);
			checkOclErrors(error);
			checkOclErrors(clSetEventCallback(output_map_event, CL_COMPLETE, []CL_CALLBACK(cl_event event, cl_int command_exec_status, void* data)
			{
				assert(command_exec_status == CL_COMPLETE);
				unique_ptr<callback_data> cbd(reinterpret_cast<callback_data*>(data));
				uint64_t total = 0;
				for (unsigned i = 0; i < cbd->gws; ++i)
				{
					total += cbd->acch[i];
				}
				printf("%lf\n", 4.0 * total / cbd->n);
				checkOclErrors(clSetUserEventStatus(cbd->e, CL_COMPLETE));
			}, new callback_data(acch, n, gws, e)));
			checkOclErrors(clFlush(command_queue));
			checkOclErrors(clWaitForEvents(1, &e));
			checkOclErrors(clEnqueueUnmapMemObject(command_queue, accd, acch, 0, NULL, NULL));
			checkOclErrors(clReleaseMemObject(accd));
			checkOclErrors(clReleaseCommandQueue(command_queue));
			checkOclErrors(clReleaseKernel(kernel));
			checkOclErrors(clReleaseProgram(program));
			checkOclErrors(clReleaseContext(context));
		}
	}
}
