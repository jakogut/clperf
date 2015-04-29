#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>

#include "file.h"
#include "error.h"

#define BUFFER_SIZE_SQRT 4096
#define SQUARE(n) (n * n)

#define NUM_ROUNDS 48
#define FLOPS_PER_ROUND 16

#define FLOPS_PER_CYCLE (NUM_ROUNDS * FLOPS_PER_ROUND)

static float* rand_matrix(const size_t size_sqrt)
{
	float* mat = calloc(SQUARE(size_sqrt), sizeof(float));

	for(unsigned i = 0; i < SQUARE(size_sqrt); i++)
		mat[i] = rand() / (float)RAND_MAX;

	return mat;
}

static float* cpu_result_matrix(const float* a, const float* b, const float* c)
{
	float* res = calloc(SQUARE(BUFFER_SIZE_SQRT), sizeof(float));

	for(int i = 0; i < SQUARE(BUFFER_SIZE_SQRT); i++)
	{
		for(int j = 0; j < NUM_ROUNDS; j++)
		{
			res[i] += a[i] + (b[i] * c[i]) + b[i];
			res[i] += b[i] + (c[i] * a[i]) + c[i];
			res[i] += c[i] + (a[i] * b[i]) + a[i];
			res[i] += a[i] + (b[i] * c[i]) + b[i];
		}
	}

	return res;
}

static unsigned timespec_to_nsec(const struct timespec* start, const struct timespec* end)
{
	return (end->tv_nsec - start->tv_nsec) +
	      ((end->tv_sec - start->tv_sec) * 1000000000);
}

static void print_perf_stats(const double sec_elapsed)
{
	printf("%i Cycles, %i FLOP/cycle, %f sec elapsed\n%f GFLOPS\n",
                (int)pow(BUFFER_SIZE_SQRT, 2),
                FLOPS_PER_CYCLE,
                sec_elapsed,
                ((pow(BUFFER_SIZE_SQRT, 2) *
		FLOPS_PER_CYCLE) / sec_elapsed) / 1000000000.0f);
}

struct cl_device_properties
{
	char name[256];

	cl_ulong global_mem_size;
	cl_ulong max_mem_alloc;

	size_t max_work_group_size;
};

struct cl_state
{
	cl_int 		error;

	cl_uint 	plat_cnt;
	cl_platform_id *platforms;
	cl_uint enabled_platforms;

	cl_context 	context;
	cl_program 	program;

	cl_uint 	 dev_cnt;
	cl_device_id 	 *devices;
	struct cl_device_properties *dev_props;

	cl_kernel 	 *kernels;
	cl_command_queue *queues;
	cl_event	 *events;
};

void destroy_cl_state(struct cl_state *cl)
{
	if(cl->events)	  free(cl->events);
	if(cl->queues)	  free(cl->queues);
	if(cl->kernels)	  free(cl->kernels);
	if(cl->dev_props) free(cl->dev_props);
	if(cl->devices)	  free(cl->devices);
	if(cl->platforms) free(cl->platforms);
}

cl_int populate_platforms(struct cl_state* cl)
{
	clGetPlatformIDs(0, NULL, &cl->plat_cnt);

	if(!cl->plat_cnt) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return -1;
	}

	cl->platforms = calloc(cl->plat_cnt, sizeof(cl_platform_id));
	cl->error = clGetPlatformIDs(cl->plat_cnt, cl->platforms, NULL);

	if(cl->error != CL_SUCCESS) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	return cl->error;
}

cl_int populate_devices(struct cl_state* cl)
{
	clGetDeviceIDs(cl->platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &cl->dev_cnt);
	if(!cl->dev_cnt) {
		return cl->error;
	}

	cl->devices = calloc(cl->dev_cnt, sizeof(cl_device_id));
	cl->error = clGetDeviceIDs(cl->platforms[0], CL_DEVICE_TYPE_GPU, cl->dev_cnt, cl->devices, NULL);

	if(cl->error != CL_SUCCESS) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	cl->dev_props = calloc(cl->dev_cnt, sizeof(struct cl_device_properties));
	for(int i = 0; i < cl->dev_cnt; i++) {
		clGetDeviceInfo(cl->devices[i], CL_DEVICE_NAME, 256, cl->dev_props[i].name, NULL);
		clGetDeviceInfo(cl->devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
				&cl->dev_props[i].max_work_group_size, NULL);
	}

	return cl->error;
}

cl_int create_context(struct cl_state *cl)
{
	cl->context = clCreateContext(NULL, cl->dev_cnt, cl->devices, NULL, NULL, &cl->error);
	if(cl->error != CL_SUCCESS) {
		printf("ERROR: Failed to create context with %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	return cl->error;
}

cl_int create_queues(struct cl_state *cl)
{
	cl->queues = calloc(cl->dev_cnt, sizeof(cl_command_queue));

	for(int i = 0; i < cl->dev_cnt; i++) {
		cl->queues[i] = clCreateCommandQueue(cl->context, cl->devices[i],
						     CL_QUEUE_PROFILING_ENABLE, &cl->error);
		if(cl->error != CL_SUCCESS) {
			printf("Failed to create command queue with error %s\n",
			       cl_errno_str(cl->error));

			return cl->error;
		}
	}

	cl->events = calloc(cl->dev_cnt, sizeof(cl_event));
	return cl->error;
}

cl_int build_program(struct cl_state *cl, char* fname)
{
	FILE* f = fopen(fname, "rb");
	size_t src_size = clp_flength(f);
	cl_uint num_src_files = 1;

	char* source = calloc(src_size, sizeof(char));
	clp_fcopy(source, src_size, f);
	fclose(f);

	cl->program = clCreateProgramWithSource(cl->context, num_src_files, (const char**)&source, &src_size, &cl->error);

	if(cl->error != CL_SUCCESS) {
		printf("ERROR: Failed to create program with %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	free(source);

	cl->error = clBuildProgram(cl->program, cl->dev_cnt, cl->devices, NULL, NULL, NULL);
	if(cl->error != CL_SUCCESS) {
		printf("ERROR: Failed building program with %s\n", cl_errno_str(cl->error));

/*		char* build_log;
		size_t log_size;

		clGetProgramBuildInfo(cl->program, cl->devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = calloc(log_size, sizeof(char));
		clGetProgramBuildInfo(cl->program, cl->devices[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		printf(build_log);
		free(build_log);
*/	}

	return cl->error;
}

cl_int create_kernels(struct cl_state *cl, char* kname)
{
	cl->kernels = calloc(cl->dev_cnt, sizeof(cl_kernel));

	for(int i = 0; i < cl->dev_cnt; i++) {
		cl->kernels[i] = clCreateKernel(cl->program, kname, &cl->error);
		if(cl->error != CL_SUCCESS) printf("ERROR: Failed to create kernel.\n");
	}

	return cl->error;
}

void verify_result(float* a, float* b)
{
	for(int i = 0; i < SQUARE(BUFFER_SIZE_SQRT); i++) {
		float ferror_pct = 100.0 / a[i] * fabs(b[i] - a[i]);
		if(ferror_pct > 5) {
			printf("Results failed verification at index %i with %f pct deviation\n\n", i, ferror_pct);
			return;
		}
	}

	printf("Result passed verification.\n\n");
}

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

	float* a_h = rand_matrix(BUFFER_SIZE_SQRT);
	float* b_h = rand_matrix(BUFFER_SIZE_SQRT);
	float* c_h = rand_matrix(BUFFER_SIZE_SQRT);

	const unsigned nelements = SQUARE(BUFFER_SIZE_SQRT);

	cl_mem *a_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *b_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *c_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *res_d = calloc(cl.dev_cnt, sizeof(cl_mem));

	for(int i = 0; i < cl.dev_cnt; i++) {
		a_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nelements * sizeof(float), a_h, &cl.error);
		b_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nelements * sizeof(float), b_h, &cl.error);
		c_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nelements * sizeof(float), c_h, &cl.error);
		res_d[i] = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, 		       nelements * sizeof(float), NULL, &cl.error);
	}

	if(cl.error != CL_SUCCESS)
		printf("Failed to create device buffers with %s\n", cl_errno_str(cl.error));

	for(int i = 0; i < cl.dev_cnt; i++) {
		cl.error =  clSetKernelArg(cl.kernels[i], 0, sizeof(cl_mem), &a_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 1, sizeof(cl_mem), &b_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 2, sizeof(cl_mem), &c_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 3, sizeof(cl_mem), &res_d[i]);
		cl.error |= clSetKernelArg(cl.kernels[i], 4, sizeof(unsigned), &nelements);

		if(cl.error != CL_SUCCESS)
			printf("Error while settings kernel args: %s\n", cl_errno_str(cl.error));
	}

	// Generate a CPU matrix for verification
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	float* cpu_result = cpu_result_matrix(a_h, b_h, c_h);
	clock_gettime(CLOCK_MONOTONIC, &end);

	for(int i = 0; i < cl.dev_cnt; i++) {
		const size_t local_ws = cl.dev_props[i].max_work_group_size;
		const size_t global_ws = nelements + (nelements % local_ws);

		cl.error = clEnqueueNDRangeKernel(cl.queues[i], cl.kernels[i], 1, NULL, &global_ws, &local_ws, 0, NULL, &cl.events[i]);
		if(cl.error != CL_SUCCESS) printf("ERROR: Kernel failed to run on GPU. Retval: %s\n", cl_errno_str(cl.error));
	}

	float* device_result = calloc(nelements, sizeof(float));
	for(int i = 0; i < cl.dev_cnt; i++) {
		clWaitForEvents(1, &cl.events[i]);

		cl_ulong time_start, time_end;
		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

		cl.error = clEnqueueReadBuffer(cl.queues[i], res_d[i], CL_TRUE, 0, nelements * sizeof(float),
				       device_result, 0, NULL, NULL);

		if(cl.error != CL_SUCCESS)
			printf("Copy device buffer to host failed with %s", cl_errno_str(cl.error));

		// Print runtime information
		printf("GPU %i: %s\n", i, cl.dev_props[i].name);
		double sec_elapsed_gpu = (time_end - time_start) / 1000000000.0f;
		print_perf_stats(sec_elapsed_gpu);

		verify_result(cpu_result, device_result);
	}

	printf("CPU: (single thread, native code)\n");
	double sec_elapsed_cpu = timespec_to_nsec(&start, &end) / 1000000000.0f;
	print_perf_stats(sec_elapsed_cpu);

	free(a_h);
	free(b_h);
	free(c_h);
	free(device_result);
	free(cpu_result);

	destroy_cl_state(&cl);

	return 0;
}
