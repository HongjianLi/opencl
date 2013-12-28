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
	size_t map_bytes;
	vector<vector<float>> maps;
	explicit receptor(const path& p) : num_probes({ 100, 80, 70 }), num_probes_product(num_probes[0] * num_probes[1] * num_probes[2]), map_bytes(sizeof(float) * num_probes_product), maps(scoring_function::n)
	{
	}
	void populate(const scoring_function& sf, const vector<size_t>& xs, const size_t z)
	{
		spin(1e+5);
	}
};

class ligand
{
public:
	explicit ligand(const path& p, const unsigned int lws) : filename(p.filename()), atoms(rand() % 10), ligh(lws)
	{
		for (const auto& a : atoms) xs[a.xs] = true;
		for (auto& c : ligh) c = rand() / static_cast<float>(RAND_MAX);
		spin(1e+4);
	}
	void encode(float* const ligh) const
	{
		memcpy(ligh, this->ligh.data(), sizeof(float) * this->ligh.size());
		spin(1e+3);
	}
	void write(const float* const cnfh) const
	{
		spin(1e+5);
	}
	path filename;
	vector<atom> atoms;
	array<bool, scoring_function::n> xs;
	vector<float> ligh;
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
	callback_data(io_service_pool& io, cl_event cbex, const size_t lws, const T dev, cl_command_queue queue, cl_mem slnd, float* const cnfh, const float* const prmh, ligand&& lig_, safe_function& safe_print, size_t& num_ligands, safe_vector<T>& idle) : io(io), cbex(cbex), lws(lws), dev(dev), queue(queue), slnd(slnd), cnfh(cnfh), prmh(prmh), lig(move(lig_)), safe_print(safe_print), num_ligands(num_ligands), idle(idle) {}
	io_service_pool& io;
	cl_event cbex;
	const size_t lws;
	const T dev;
	cl_command_queue queue;
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
	vector<cl_uint> num_platform_devices(num_platforms);
	cl_uint num_devices = 0;
	for (cl_uint i = 0; i < num_platforms; ++i)
	{
		const auto platform = platforms[i];
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
		cout << i << ' ' << buffer;
		checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL));
		cout << ", " << buffer << endl;
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices[i]));
		num_devices += num_platform_devices[i];
	}

	cout << "Detecting OpenCL devices" << endl;
	if (!num_devices)
	{
		cerr << "No OpenCL devices detected" << endl;
		return 2;
	}
	vector<cl_device_id> devices(num_devices);
	vector<bool> cl12(num_devices);
	vector<cl_bool> host_unified_memory(num_devices);
	for (cl_uint i = 0, dev = 0; i < num_platforms; ++i)
	{
		const auto platform = platforms[i];
		const auto npd = num_platform_devices[i];
		checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, npd, &devices[dev], NULL));
		for (int d = 0; d < npd; ++d, ++dev)
		{
			const auto device = devices[dev];
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
			cout << dev << ' ' << buffer;
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL));
			cout << ", " << buffer << endl;
			cl12[dev] = buffer[9] > '1' || buffer[11] >= '2';
			checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(host_unified_memory[dev]), &host_unified_memory[dev], NULL));
		}
	}

	cout << "Creating contexts and compiling modules for " << num_devices << " devices" << endl;
	std::ifstream ifs("multiDevice.cl");
	string source((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
	const char* sources[] = { source.data() };
	const size_t source_length = source.length();
	vector<cl_context> contexts(num_devices);
	vector<cl_command_queue> queues(num_devices);
	vector<cl_program> programs(num_devices);
	vector<cl_kernel> kernels(num_devices);
	vector<cl_mem> prmd(num_devices);
//	vector<cl_mem> ligh(num_devices);
	vector<cl_mem> ligd(num_devices);
	vector<cl_mem> slnd(num_devices);
//	vector<cl_mem> cnfh(num_devices);
	vector<array<cl_mem, sf.n>> mpsd(num_devices);
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

		// Create buffers for prmd, ligh, ligd, slnd and cnfh.
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

		// Set kernel arguments.
		checkOclErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &slnd[dev]));
		checkOclErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &ligd[dev]));
		checkOclErrors(clSetKernelArg(kernel, 2, sizeof(cl_float) * lws, NULL));
		checkOclErrors(clSetKernelArg(kernel, 3, sizeof(cl_mem), &prmd[dev]));
	}
	source.clear();

	// Initialize a vector of idle devices.
	safe_vector<int> idle(num_devices);
	iota(idle.begin(), idle.end(), 0);

	// Perform docking for each ligand in the input folder.
	size_t num_ligands = 0;
	vector<cl_event> cbex(num_devices);
	cout.setf(ios::fixed, ios::floatfield);
	cout << "ID              Ligand D  pKd 1     2     3     4     5     6     7     8     9" << endl << setprecision(2);
	for (directory_iterator dir_iter("."), const_dir_iter; dir_iter != const_dir_iter; ++dir_iter)
	{
		// Parse the ligand.
		ligand lig(dir_iter->path(), lws);

		// Find atom types that are presented in the current ligand but not presented in the grid maps.
		vector<size_t> xs;
		for (size_t t = 0; t < sf.n; ++t)
		{
			if (lig.xs[t] && rec.maps[t].empty())
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

		// Copy grid maps from host memory to device memory if necessary.
		for (size_t t = 0; t < sf.n; ++t)
		{
			if (lig.xs[t] && !mpsd[dev][t])
			{
				mpsd[dev][t] = clCreateBuffer(contexts[dev], CL_MEM_READ_ONLY, rec.map_bytes, NULL, &error);
				checkOclErrors(error);
				checkOclErrors(clEnqueueWriteBuffer(queues[dev], mpsd[dev][t], CL_TRUE, 0, rec.map_bytes, rec.maps[t].data(), 0, NULL, NULL));
			}
		}

		// Encode the current ligand.
		cl_event input_events[2];
		cl_float* ligh = (cl_float*)clEnqueueMapBuffer(queues[dev], ligd[dev], CL_TRUE, cl12[dev] ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE, 0, sizeof(cl_float) * lws, 0, NULL, NULL, &error);
		checkOclErrors(error);
		lig.encode(ligh);
		checkOclErrors(clEnqueueUnmapMemObject(queues[dev], ligd[dev], ligh, 0, NULL, &input_events[0]));

		// Clear the solution buffer.
		if (cl12[dev])
		{
			const cl_float pattern = 0.0f;
			checkOclErrors(clEnqueueFillBuffer(queues[dev], slnd[dev], &pattern, sizeof(pattern), 0, sizeof(cl_float) * gws, 0, NULL, &input_events[1]));
		}
		else
		{
			cl_float* slnh = (cl_float*)clEnqueueMapBuffer(queues[dev], slnd[dev], CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_float) * gws, 0, NULL, NULL, &error);
			checkOclErrors(error);
			memset(slnh, 0, sizeof(cl_float) * gws);
			checkOclErrors(clEnqueueUnmapMemObject(queues[dev], slnd[dev], slnh, 0, NULL, &input_events[1]));
		}

		// Launch kernel.
		cl_event kernel_event;
		checkOclErrors(clEnqueueNDRangeKernel(queues[dev], kernels[dev], 1, NULL, &gws, &lws, 2, input_events, &kernel_event));

		// Copy conformations from device memory to host memory.
		cl_event output_event;
		cl_float* cnfh = (cl_float*)clEnqueueMapBuffer(queues[dev], slnd[dev], CL_FALSE, CL_MAP_READ, 0, sizeof(cl_float) * lws, 1, &kernel_event, &output_event, &error);
		checkOclErrors(error);

		// Create callback events.
		if (cbex[dev]) checkOclErrors(clReleaseEvent(cbex[dev]));
		cbex[dev] = clCreateUserEvent(contexts[dev], &error);
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
				const auto  slnd = cbd->slnd;
				const auto  cnfh = cbd->cnfh;
				const auto  prmh = cbd->prmh;
				auto& lig = cbd->lig;
				auto& safe_print = cbd->safe_print;
				auto& num_ligands = cbd->num_ligands;
				auto& idle = cbd->idle;

				// Validate results.
				cl_int error;
				for (int i = 0; i < lws; ++i)
				{
					const float actual = cnfh[i];
					const float expected = lig.ligh[i] * 2.0f + 1.0f + prmh[i % 16];
					if (fabs(actual - expected) > 1e-7)
					{
						cout << "cnfh[" << i << "] = " << actual << ", expected = " << expected << endl;
						break;
					}
				}

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
			checkOclErrors(clSetUserEventStatus(cbd->cbex, CL_COMPLETE));
		}, new callback_data<int>(io, cbex[dev], lws, dev, queues[dev], slnd[dev], cnfh, prmh.data(), move(lig), safe_print, num_ligands, idle)));
	}

	// Synchronize queues and callback events.
	for (int dev = 0; dev < num_devices; ++dev)
	{
		checkOclErrors(clFinish(queues[dev]));
		if (cbex[dev]) checkOclErrors(clWaitForEvents(1, &cbex[dev]));
	}

	// Wait until the io service pool has finished all its tasks.
	io.wait();
	assert(idle.size() == num_devices);

	// Release resources.
	for (int dev = 0; dev < num_devices; ++dev)
	{
		if (cbex[dev]) checkOclErrors(clReleaseEvent(cbex[dev]));
		for (auto mapd : mpsd[dev])
		{
			if (mapd) checkOclErrors(clReleaseMemObject(mapd));
		}
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
