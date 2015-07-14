#include "cpu_bench.h"

#include <stdlib.h>
#include <pthread.h>
#include <string.h>

static float *cpu_result_matrix(struct bench_buf *in)
{
	float *res = aligned_alloc(16, BUFFER_SIZE * sizeof(float));

	#pragma omp parallel for
	for (unsigned i = 0; i < BUFFER_SIZE; i++) {
		float a = in->a[i], b = in->b[i], c = in->c[i];

		for (unsigned j = 0; j < ROUNDS_PER_ITERATION; j++) {
			res[i] += a * ((b * c) + b);
			res[i] += b * ((c * a) + c);
			res[i] += c * ((a * b) + a);

			res[i] += a * ((b * c) + b);
			res[i] += b * ((c * a) + c);
			res[i] += c * ((a * b) + a);

			res[i] += a * ((b * c) + b);
			res[i] += b * ((c * a) + c);
			res[i] += c * ((a * b) + a);

			res[i] += a * ((b * c) + b);
			res[i] += b * ((c * a) + c);
			res[i] += c * ((a * b) + a);
		}
	}

	return res;
}

double cpu_bench(struct bench_buf *in, float *result)
{
	struct timespec start, end;
	float *mat;

	clock_gettime(CLOCK_MONOTONIC, &start);
	mat = cpu_result_matrix(in);
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (result != NULL)
		memcpy(result, mat, sizeof(float) * BUFFER_SIZE);

	free(mat);

	return timespec_to_nsec(&start, &end) / 1000000000.0f;
}
