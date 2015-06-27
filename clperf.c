#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include <CL/cl.h>
#include "cl_common.h"
#include "benchmark.h"
#include "cpu_bench.h"

int main()
{
	struct cl_state cl;

	srand(time(NULL));

	populate_platforms(&cl);
	populate_devices(&cl);
	create_context(&cl);
	create_queues(&cl);
	build_program(&cl, "clperf_fmadd.cl");
	create_kernels(&cl, "matrix_fmadd");

	struct bench_buf input = {
		.a = rand_matrix(BUFFER_SIZE),
		.b = rand_matrix(BUFFER_SIZE),
		.c = rand_matrix(BUFFER_SIZE),
	};

	cl_mem *a_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *b_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *c_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *res_d = calloc(cl.dev_cnt, sizeof(cl_mem));

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		a_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), input.a, &cl.error);
		b_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), input.b, &cl.error);
		c_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), input.c, &cl.error);
		res_d[i] = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,		       BUFFER_SIZE * sizeof(float), NULL, &cl.error);
	}

	if (cl.error != CL_SUCCESS)
		printf("Failed to create device buffers with %s\n", cl_errno_str(cl.error));

	const unsigned nelements = BUFFER_SIZE;

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		cl.error =  clSetKernelArg(cl.kernels[i], 0, sizeof(cl_mem), &a_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 1, sizeof(cl_mem), &b_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 2, sizeof(cl_mem), &c_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 3, sizeof(cl_mem), &res_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 4, sizeof(unsigned), &nelements);

		if (cl.error != CL_SUCCESS)
			printf("Error while settings kernel args: %s\n", cl_errno_str(cl.error));
	}

	// Generate a CPU matrix for verification
	float* cpu_result = calloc(BUFFER_SIZE, sizeof(float));
	double cpu_bench_time = cpu_bench(&input, cpu_result);

	printf("CPU bench: native code, %i thread(s)\n", nthreads());
	print_perf_stats(cpu_bench_time);

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		const size_t local_ws = cl.dev_props[i].max_work_group_size;
		const size_t global_ws = BUFFER_SIZE + (BUFFER_SIZE % local_ws);

		cl.error = clEnqueueNDRangeKernel(cl.queues[i], cl.kernels[i], 1, NULL, &global_ws, &local_ws, 0, NULL, &cl.events[i]);
		if (cl.error != CL_SUCCESS) printf("ERROR: Kernel failed to run on GPU. Retval: %s\n", cl_errno_str(cl.error));
	}

	float* device_result = calloc(BUFFER_SIZE, sizeof(float));

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		clWaitForEvents(1, &cl.events[i]);

		cl_ulong time_start, time_end;

		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

		cl.error = clEnqueueReadBuffer(cl.queues[i], res_d[i], CL_TRUE, 0, BUFFER_SIZE * sizeof(float),
				       device_result, 0, NULL, NULL);

		if (cl.error != CL_SUCCESS)
			printf("Copy device buffer to host failed with %s", cl_errno_str(cl.error));

		// Print runtime information
		printf("GPU %i: %s\n", i, cl.dev_props[i].name);

		double sec_elapsed_gpu = (time_end - time_start) / 1000000000.0f;

		print_perf_stats(sec_elapsed_gpu);
		verify_result(cpu_result, device_result);
	}

	free(input.a);
	free(a_d);
	free(input.b);
	free(b_d);
	free(input.c);
	free(c_d);
	free(res_d);
	free(device_result);
	free(cpu_result);

	destroy_cl_state(&cl);

	return 0;
}
