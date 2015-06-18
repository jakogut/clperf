#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include <CL/cl.h>
#include "cl_common.h"

#define BUFFER_SIZE (2 << 20)

#define ROUNDS_PER_ITERATION 48
#define FLOPS_PER_ROUND 12

#define FLOPS_PER_ITERATION (ROUNDS_PER_ITERATION * FLOPS_PER_ROUND)

static float* rand_matrix(const size_t size)
{
	float* mat = calloc(size, sizeof(float));

	for(unsigned i = 0; i < size; i++)
		mat[i] = rand() / (float)RAND_MAX;

	return mat;
}

struct cpu_res_arg { int tid; int tn; const float* a; const float* b; const float* c; float* ret; };

void* cpu_result_matrix_mt(void* v_arg)
{
	struct cpu_res_arg* arg = (struct cpu_res_arg*)v_arg;

	const unsigned buff_size = BUFFER_SIZE;
	const unsigned round_cnt = ROUNDS_PER_ITERATION / 4;

	const unsigned work_size = buff_size / arg->tn;
	const unsigned work_start = arg->tid * work_size;

	const unsigned work_end = work_start + work_size;

	float lres;
	for(int i = work_start; i < work_end; i++) {

		lres = 0;
		float a = arg->a[i], b = arg->b[i], c = arg->c[i];

		for(int j = 0; j < round_cnt; j++) {
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
		}

		arg->ret[i] = lres;
	}

	return NULL;
}

static float* cpu_result_matrix(const float* a, const float* b, const float* c)
{
	const int tn = 8;
	struct cpu_res_arg targ[tn];

	float* res = aligned_alloc(16, BUFFER_SIZE * sizeof(float));

	for(int i = 0; i < tn; i++) {
		targ[i].tid = i;
		targ[i].tn = tn;
		targ[i].a = a;
		targ[i].b = b;
		targ[i].c = c;
		targ[i].ret = res;
	}

	pthread_t cpu_res_t[tn];
	for(int i = 0; i < tn; i++) pthread_create(&cpu_res_t[i], NULL, cpu_result_matrix_mt, (void*)&targ[i]);
	for(int i = 0; i < tn; i++) pthread_join(cpu_res_t[i], NULL);

	return (float*)res;
}

static unsigned timespec_to_nsec(const struct timespec* start, const struct timespec* end)
{
	return (end->tv_nsec - start->tv_nsec) +
	      ((end->tv_sec - start->tv_sec) * 1000000000);
}

static void print_perf_stats(const double sec_elapsed)
{
	printf("%i Cycles, %i FLOP/iteration, %f sec elapsed\n%f GFLOPS\n",
                BUFFER_SIZE,
                FLOPS_PER_ITERATION,
                sec_elapsed,
                ((BUFFER_SIZE * FLOPS_PER_ITERATION) / sec_elapsed) / 1000000000.0f);
}

void verify_result(float* a, float* b)
{
	for(int i = 0; i < BUFFER_SIZE; i++) {
		float ferror_pct = 100.0 / a[i] * fabs(b[i] - a[i]);
		if(ferror_pct > 5) {
			printf("Results failed verification at index %i with %f pct deviation\n", i, ferror_pct);
			printf("Expected %f, calculated %f\n\n", a[i], b[i]);
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

	float* a_h = rand_matrix(BUFFER_SIZE);
	float* b_h = rand_matrix(BUFFER_SIZE);
	float* c_h = rand_matrix(BUFFER_SIZE);

	cl_mem *a_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *b_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *c_d = calloc(cl.dev_cnt, sizeof(cl_mem));
	cl_mem *res_d = calloc(cl.dev_cnt, sizeof(cl_mem));

	for(int i = 0; i < cl.dev_cnt; i++) {
		a_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), a_h, &cl.error);
		b_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), b_h, &cl.error);
		c_d[i]   = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_SIZE * sizeof(float), c_h, &cl.error);
		res_d[i] = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, 		      BUFFER_SIZE * sizeof(float), NULL, &cl.error);
	}

	if(cl.error != CL_SUCCESS)
		printf("Failed to create device buffers with %s\n", cl_errno_str(cl.error));

	const unsigned nelements = BUFFER_SIZE;
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
		const size_t global_ws = BUFFER_SIZE + (BUFFER_SIZE % local_ws);

		cl.error = clEnqueueNDRangeKernel(cl.queues[i], cl.kernels[i], 1, NULL, &global_ws, &local_ws, 0, NULL, &cl.events[i]);
		if(cl.error != CL_SUCCESS) printf("ERROR: Kernel failed to run on GPU. Retval: %s\n", cl_errno_str(cl.error));
	}

	float* device_result = calloc(BUFFER_SIZE, sizeof(float));
	for(int i = 0; i < cl.dev_cnt; i++) {
		clWaitForEvents(1, &cl.events[i]);

		cl_ulong time_start, time_end;
		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
		clGetEventProfilingInfo(cl.events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

		cl.error = clEnqueueReadBuffer(cl.queues[i], res_d[i], CL_TRUE, 0, BUFFER_SIZE * sizeof(float),
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
