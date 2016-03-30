#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <list>
#include <numeric>
#include <random>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 4 - memory allocation
		//host - input
		//std::vector<mytype> A(10);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		//std::iota(std::begin(A), std::end(A), 0);

		std::vector<mytype> A = {};
		ifstream f;
		string s;
		f.open("temp_lincolnshire_short.txt");
		std::vector<mytype> Location = {};
		std::vector<mytype> Year = {};
		std::vector<mytype> Month = {};
		std::vector<mytype> Date = {};
		std::vector<mytype> Time = {};
		std::vector<mytype> Temp = {};

		while (std::getline(f, s))
		{
			std::string StringBuffer;
			std:stringstream StringStream(s);
			std::vector<std::string> StringSplit;
			while (StringStream >> StringBuffer)
			{
				StringSplit.push_back(StringBuffer);
			}
			if (StringSplit.size() < 5)
			{
				continue;
			}
			A.push_back(std::stoi(StringSplit[5])*10);

		}
		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 32;// A.size();
		size_t origin_A = A.size();
		//cout << A.size();
		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			//std::vector<int> A_ext(local_size-padding_size, 0);
			std::vector<int> A_ext(local_size - padding_size, UINT16_MAX);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		std::vector<mytype> B(10);
		std::vector<mytype> C(1);
		std::vector<mytype> D(1);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);
		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "max_finder");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_C);
		cl::Kernel kernel_2 = cl::Kernel(program, "min_finder");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_D);
		cl::Kernel kernel_3 = cl::Kernel(program, "hist_simple");		
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_B);
		kernel_3.setArg(2, 10); //Bin Width
		kernel_3.setArg(3, 1000); //Max_Value
		kernel_3.setArg(4, -500); //Min_Value




	
       // kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
	// 	queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		cl::Event kernel_event;
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &kernel_event);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &kernel_event);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &kernel_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, 2, &C[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, 2, &D[0]);

		//std::cout << "A = " << A << std::endl;
		//std::cout << "B = " << B << std::endl;
		std::cout << "Max = " << C << std::endl;
		std::cout << "Min  = " << D << std::endl; //needs change because minus issue (underflow)
		std::cout << "Kernel execution time [ns]:" << kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	system("pause");	
	return 0;


}
