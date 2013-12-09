#include <iostream>
#include <iomanip>
#include <ctime>
#include <array>
#include <numeric>
#include <fstream>
#include <boost/filesystem/operations.hpp>
#include <CL/cl.h>
#include "../cl_helper.h"
#include "io_service_pool.hpp"
using namespace boost::filesystem;

void spin(const size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		rand();
	}
}

class scoring_function
{
public:
	static const size_t n = 15;
};

class atom
{
public:
	atom() : xs(rand() % scoring_function::n) {}
	size_t xs;
};

class receptor
{
public:
	array<int, 3> num_probes;
	size_t num_probes_product;
	vector<vector<float>> maps;
	explicit receptor(const path& p) : num_probes({100, 80, 70}), num_probes_product(1), maps(scoring_function::n)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			num_probes_product *= num_probes[i];
		}
	}
	void populate(const scoring_function& sf, const vector<size_t>& xs, const size_t z)
	{
		spin(1e+5);
	}
};

class ligand
{
public:
	explicit ligand(const path& p) : filename(p.filename()), atoms(rand() % 10)
	{
		spin(1e+4);
	}
	void encode(float* const ligh, const unsigned int lws) const
	{
		for (int i = 0; i < lws; ++i)
		{
			ligh[i] = rand() / static_cast<float>(RAND_MAX);
		}
		spin(1e+3);
	}
	void write(const float* const cnfh) const
	{
		spin(1e+5);
	}
	path filename;
	vector<atom> atoms;
};

class safe_function
{
public:
	void operator()(function<void(void)>&& f)
	{
		lock_guard<mutex> guard(m);
		f();
	}
private:
	mutex m;
};

template <typename T>
class safe_counter
{
public:
	void init(const T z)
	{
		n = z;
		i = 0;
	}
	void increment()
	{
		lock_guard<mutex> guard(m);
		if (++i == n) cv.notify_one();
	}
	void wait()
	{
		unique_lock<mutex> lock(m);
		if (i < n) cv.wait(lock);
	}
private:
	mutex m;
	condition_variable cv;
	T n;
	T i;
};

template <typename T>
class safe_vector : public vector<T>
{
public:
	using vector<T>::vector;
	void safe_push_back(const T x)
	{
		lock_guard<mutex> guard(m);
		this->push_back(x);
		cv.notify_one();
	}
	T safe_pop_back()
	{
		unique_lock<mutex> lock(m);
		if (this->empty()) cv.wait(lock);
		const T x = this->back();
		this->pop_back();
		return x;
	}
private:
	mutex m;
	condition_variable cv;
};

template <typename T>
class callback_data
{
public:
	callback_data(io_service_pool& io, const size_t lws, const T dev, cl_command_queue queue, cl_mem ligd, cl_mem slnd, float* const cnfh, const float* const prmh, ligand&& lig_, safe_function& safe_print, size_t& num_ligands, safe_vector<T>& idle) : io(io), lws(lws), dev(dev), queue(queue), ligd(ligd), slnd(slnd), cnfh(cnfh), prmh(prmh), lig(move(lig_)), safe_print(safe_print), num_ligands(num_ligands), idle(idle) {}
	io_service_pool& io;
	const size_t lws;
	const T dev;
	cl_command_queue queue;
	cl_mem ligd;
	cl_mem slnd;
	cl_float* const cnfh;
	const cl_float* const prmh;
	ligand lig;
	safe_function& safe_print;
	size_t& num_ligands;
	safe_vector<T>& idle;
};

int main(int argc, char* argv[])
{
	// Initialize constants.
	const size_t lws = 256;
	const size_t gws = 32 * lws;
	const unsigned int num_threads = thread::hardware_concurrency();

	// Initialize variables.
	srand(time(0));
	vector<float> prmh(16);
	for (auto& prm : prmh)
	{
		prm = rand() / static_cast<float>(RAND_MAX);
	}

	cout << "Creating an io service pool of " << num_threads << " worker threads" << endl;
	io_service_pool io(num_threads);
	safe_counter<size_t> cnt;
	safe_function safe_print;

	cout << "Precalculating a scoring function of " << scoring_function::n << " atom types in parallel" << endl;
	scoring_function sf;

	path receptor_path;
	cout << "Parsing receptor " << receptor_path << endl;
	receptor rec(receptor_path);

	cout << "Detecting OpenCL platforms" << endl;
	char buffer[256];
	cl_uint num_platforms;
	checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));
	vector<cl_platform_id> platforms(num_platforms);
	checkOclErrors(clGetPlatformIDs(num_platforms, platforms.data(), NULL));
	const auto platform = platforms[num_platforms == 3 ? 1 : 0];
	checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
	cout << "CL_PLATFORM_NAME: " << buffer << endl;

	cout << "Detecting OpenCL devices" << endl;
	cl_uint num_devices;
	checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
	if (!num_devices)
	{
		cerr << "No OpenCL devices detected" << endl;
		return 2;
	}

	vector<cl_device_id> devices(num_devices);
	checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL));
	vector<bool> cl12(num_devices);
	vector<cl_bool> host_unified_memory(num_devices);
	for (int dev = 0; dev < num_devices; ++dev)
	{
		const auto device = devices[dev];
		checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		printf("CL_DEVICE_NAME: %s\n", buffer);
		checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL));
		cl12[dev] = !strcmp(buffer, "OpenCL C 1.2");
		checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(host_unified_memory[dev]), &host_unified_memory[dev], NULL));
	}

	cout << "Compiling modules for " << num_devices << " devices" << endl;
	std::ifstream ifs("multiDevice.cl");
	string source((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
	const char* sources[] = { source.data() };
	const size_t source_length = source.length();
	vector<cl_context> contexts(num_devices);
	vector<cl_command_queue> queues(num_devices);
	vector<cl_program> programs(num_devices);
	vector<cl_kernel> kernels(num_devices);
	vector<vector<size_t>> xst(num_devices);
	vector<cl_mem> prmd(num_devices);
//	vector<cl_mem> ligh(num_devices);
	vector<cl_mem> ligd(num_devices);
	vector<cl_mem> slnd(num_devices);
//	vector<cl_mem> cnfh(num_devices);
	cl_int error;
	for (int dev = 0; dev < num_devices; ++dev)
	{
		// Get device.
		cl_device_id device = devices[dev];

		// Create context.
		cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
		checkOclErrors(error);
		contexts[dev] = context;

		// Create command queue.
		cl_command_queue_properties queue_properties;
		checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL));
		cl_command_queue queue = clCreateCommandQueue(context, device, queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0/* | CL_QUEUE_PROFILING_ENABLE*/, &error);
		checkOclErrors(error);
		queues[dev] = queue;

		// Create program.
		cl_program program = clCreateProgramWithSource(context, 1, sources, &source_length, &error);
		checkOclErrors(error);
		programs[dev] = program;

		// Build program.
		checkOclErrors(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math"/*-cl-std=CL1.2 -cl-nv-maxrregcount 32*/, NULL, NULL));

		// Create kernel from program.
		cl_kernel kernel = clCreateKernel(program, "monte_carlo", &error);
		checkOclErrors(error);
		kernels[dev] = kernel;

		// Reserve space for xst.
		xst[dev].reserve(sf.n);

		// Allocate prmd, ligh, ligd, slnd and cnfh.
		prmd[dev] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * lws, prmh.data(), &error);
		checkOclErrors(error);
//		ligh[dev] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_float) * lws, NULL, &error);
//		checkOclErrors(error);
		ligd[dev] = clCreateBuffer(context, CL_MEM_READ_ONLY,      sizeof(cl_float) * lws, NULL, &error);
		checkOclErrors(error);
		slnd[dev] = clCreateBuffer(context, CL_MEM_READ_WRITE,     sizeof(cl_float) * gws, NULL, &error);
		checkOclErrors(error);
//		cnfh[dev] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_float) * lws, NULL, &error);
//		checkOclErrors(error);
	}
	source.clear();

	// Initialize a vector of idle devices.
	safe_vector<int> idle(num_devices);
	iota(idle.begin(), idle.end(), 0);

	// Perform docking for each ligand in the input folder.
	size_t num_ligands = 0;
	cout.setf(ios::fixed, ios::floatfield);
	cout << "ID              Ligand D  pKd 1     2     3     4     5     6     7     8     9" << endl << setprecision(2);
	for (directory_iterator dir_iter("."), const_dir_iter; dir_iter != const_dir_iter; ++dir_iter)
	{
		// Parse the ligand.
		ligand lig(dir_iter->path());

		// Find atom types that are presented in the current ligand but not presented in the grid maps.
		vector<size_t> xs;
		for (const atom& a : lig.atoms)
		{
			const size_t t = a.xs;
			if (rec.maps[t].empty())
			{
				rec.maps[t].resize(rec.num_probes_product);
				xs.push_back(t);
			}
		}

		// Create grid maps on the fly if necessary.
		if (xs.size())
		{
			// Create grid maps in parallel.
			cnt.init(rec.num_probes[2]);
			for (size_t z = 0; z < rec.num_probes[2]; ++z)
			{
				io.post([&,z]()
				{
					rec.populate(sf, xs, z);
					cnt.increment();
				});
			}
			cnt.wait();
		}

		// Wait until a device is ready for execution.
		const int dev = idle.safe_pop_back();

		// Find atom types that are presented in the current ligand but are not yet copied to device memory.
		xs.clear();
		for (const atom& a : lig.atoms)
		{
			const size_t t = a.xs;
			if (find(xst[dev].cbegin(), xst[dev].cend(), t) == xst[dev].cend())
			{
				xst[dev].push_back(t);
				xs.push_back(t);
			}
		}

		// Copy grid maps from host memory to device memory if necessary.
		if (xs.size())
		{
			const size_t map_bytes = sizeof(float) * rec.num_probes_product;
			for (const auto t : xs)
			{
//				CUdeviceptr mapd;
//				checkCudaErrors(cuMemAlloc(&mapd, map_bytes));
//				checkCudaErrors(cuMemcpyHtoD(mapd, rec.maps[t].data(), map_bytes));
			}
		}

		// Encode the current ligand.
		cl_event input_events[2];
		cl_float* ligh = (cl_float*)clEnqueueMapBuffer(queues[dev], ligd[dev], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(cl_float) * lws, 0, NULL, NULL, &error);
		checkOclErrors(error);
		lig.encode(ligh, lws);
		checkOclErrors(clEnqueueUnmapMemObject(queues[dev], ligd[dev], ligh, 0, NULL, &input_events[0]));

		// Clear the solution buffer.
		if (cl12[dev])
		{
			const cl_float pattern = 0.0f;
			checkOclErrors(clEnqueueFillBuffer(queues[dev], slnd[dev], &pattern, sizeof(pattern), 0, sizeof(cl_float) * gws, 0, NULL, &input_events[0]));
		}
		else
		{
			cl_float* slnh = (cl_float*)clEnqueueMapBuffer(queues[dev], slnd[dev], CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(cl_float) * gws, 0, NULL, NULL, &error);
			checkOclErrors(error);
			memset(slnh, 0, sizeof(cl_float) * gws);
			checkOclErrors(clEnqueueUnmapMemObject(queues[dev], slnd[dev], slnh, 0, NULL, &input_events[1]));
		}

		// Launch kernel.
		checkOclErrors(clSetKernelArg(kernels[dev], 0, sizeof(cl_mem), &slnd[dev]));
		checkOclErrors(clSetKernelArg(kernels[dev], 1, sizeof(cl_mem), &ligd[dev]));
		checkOclErrors(clSetKernelArg(kernels[dev], 2, sizeof(cl_float) * lws, NULL));
		checkOclErrors(clSetKernelArg(kernels[dev], 3, sizeof(cl_mem), &prmd[dev]));
		cl_event kernel_event;
		checkOclErrors(clEnqueueNDRangeKernel(queues[dev], kernels[dev], 1, NULL, &gws, &lws, 2, input_events, &kernel_event));

		// Copy conformations from device memory to host memory.
		cl_event output_event;
		cl_float* cnfh = (cl_float*)clEnqueueMapBuffer(queues[dev], slnd[dev], CL_FALSE, CL_MAP_READ, 0, sizeof(cl_float) * lws, 1, &kernel_event, &output_event, &error);
		checkOclErrors(error);

		// Add a callback to the output event.
		checkOclErrors(clSetEventCallback(output_event, CL_COMPLETE, [](cl_event event, cl_int command_exec_status, void* data)
		{
			assert(command_exec_status == CL_COMPLETE);
			const shared_ptr<callback_data<int>> cbd(reinterpret_cast<callback_data<int>*>(data));
			cbd->io.post([=]()
			{
				const auto   lws = cbd->lws;
				const auto   dev = cbd->dev;
				const auto queue = cbd->queue;
				const auto  ligd = cbd->ligd;
				const auto  slnd = cbd->slnd;
				const auto  cnfh = cbd->cnfh;
				const auto  prmh = cbd->prmh;
				auto& lig = cbd->lig;
				auto& safe_print = cbd->safe_print;
				auto& num_ligands = cbd->num_ligands;
				auto& idle = cbd->idle;

				// Validate results.
				cl_int error;
				cl_float* ligh = (cl_float*)clEnqueueMapBuffer(queue, ligd, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(cl_float) * lws, 0, NULL, NULL, &error);
				checkOclErrors(error);
				for (int i = 0; i < lws; ++i)
				{
					const float actual = cnfh[i];
					const float expected = ligh[i] * 2.0f + 1.0f + prmh[i % 16];
					if (fabs(actual - expected) > 1e-7)
					{
						printf("cnfh[%d] = %f, expected = %f\n", i, actual, expected);
						break;
					}
				}
				checkOclErrors(clEnqueueUnmapMemObject(queue, ligd, ligh, 0, NULL, NULL));

				// Write conformations.
				lig.write(cnfh);

				// Output and save ligand stem and predicted affinities.
				safe_print([&]()
				{
					cout << setw(2) << ++num_ligands << setw(20) << lig.filename.string() << setw(2) << dev << ' ';
					for (int i = 0; i < 9; ++i)
					{
						cout << setw(6) << cnfh[i];
					}
					cout << endl;
				});

				// Unmap cnfh.
				checkOclErrors(clEnqueueUnmapMemObject(queue, slnd, cnfh, 0, NULL, NULL));

				// Signal the main thread to post another task.
				idle.safe_push_back(dev);
			});
		}, new callback_data<int>(io, lws, dev, queues[dev], ligd[dev], slnd[dev], cnfh, prmh.data(), move(lig), safe_print, num_ligands, idle)));
	}

	// Synchronize queues.
	for (auto& queue : queues)
	{
		checkOclErrors(clFlush(queue));
//		checkOclErrors(clWaitForEvents(1, &callback_event));
	}

	// Wait until the io service pool has finished all its tasks.
	io.wait();
	assert(idle.size() == num_devices);

	// Release resources.
	for (int dev = 0; dev < num_devices; ++dev)
	{
		checkOclErrors(clReleaseMemObject(prmd[dev]));
		checkOclErrors(clReleaseMemObject(slnd[dev]));
		checkOclErrors(clReleaseMemObject(ligd[dev]));
		checkOclErrors(clReleaseCommandQueue(queues[dev]));
		checkOclErrors(clReleaseKernel(kernels[dev]));
		checkOclErrors(clReleaseProgram(programs[dev]));
		checkOclErrors(clReleaseContext(contexts[dev]));
	}

	cout << "Writing log records of " << num_ligands << " ligands to the log file" << endl;
}
