#include "cpu_bench.h"

#include <stdlib.h>
#include <pthread.h>
#include <string.h>

struct cpu_res_arg { unsigned tid; unsigned tc; struct bench_buf *in; float *ret; };

static void *cpu_result_matrix_mt(void *v_arg)
{
	struct cpu_res_arg *arg = (struct cpu_res_arg *)v_arg;

	const unsigned buff_size = BUFFER_SIZE;
	const unsigned round_cnt = ROUNDS_PER_ITERATION / 4;

	const unsigned work_size = buff_size / arg->tc;
	const unsigned work_start = arg->tid * work_size;

	const unsigned work_end = work_start + work_size;

	float lres;
	for(unsigned i = work_start; i < work_end; i++) {

		lres = 0;
		float a = arg->in->a[i], b = arg->in->b[i], c = arg->in->c[i];

		for(unsigned j = 0; j < round_cnt; j++) {
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
			lres += a * ((b * c) + b); lres += b * ((c * a) + c); lres += c * ((a * b) + a);
		}

		arg->ret[i] = lres;
	}

	return NULL;
}

static float* cpu_result_matrix(struct bench_buf *in)
{
	const unsigned tc = nthreads();
	struct cpu_res_arg targ[tc];

	float* res = aligned_alloc(16, BUFFER_SIZE * sizeof(float));

	for(unsigned i = 0; i < tc; i++) {
		targ[i].tid = i;
		targ[i].tc = tc;
		targ[i].in = in;
		targ[i].ret = res;
	}

	pthread_t cpu_res_t[tc];
	for(unsigned i = 0; i < tc; i++) pthread_create(&cpu_res_t[i], NULL, cpu_result_matrix_mt, (void*)&targ[i]);
	for(unsigned i = 0; i < tc; i++) pthread_join(cpu_res_t[i], NULL);

	return (float*)res;
}

double cpu_bench(struct bench_buf *in, float *result)
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	float* mat = cpu_result_matrix(in);
	clock_gettime(CLOCK_MONOTONIC, &end);

	if(result != NULL) memcpy(result, mat, sizeof(float) * BUFFER_SIZE);
	free(mat);

	return timespec_to_nsec(&start, &end) / 1000000000.0f;
}
