#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

#include "benchmark.h"

#define FLOPS_PER_ROUND 12
#define FLOPS_PER_ITERATION (ROUNDS_PER_ITERATION * FLOPS_PER_ROUND)

float *rand_matrix(const size_t size)
{
	float *mat = calloc(size, sizeof(float));

	for(unsigned i = 0; i < size; i++)
		mat[i] = rand() / (float)RAND_MAX;

	return mat;
}

int nthreads()
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

unsigned timespec_to_nsec(const struct timespec *start, const struct timespec *end)
{
	return (end->tv_nsec - start->tv_nsec) +
	      ((end->tv_sec - start->tv_sec) * 1000000000);
}

void print_perf_stats(const double sec_elapsed)
{
	printf("%i Cycles, %i FLOP/iteration, %f sec elapsed\n%f GFLOPS\n\n",
		BUFFER_SIZE,
		FLOPS_PER_ITERATION,
		sec_elapsed,
		((BUFFER_SIZE * FLOPS_PER_ITERATION) / sec_elapsed) / 1000000000.0f);
}

void verify_result(float *a, float *b)
{
	float max_ferror = 0;
	for(unsigned i = 0; i < BUFFER_SIZE; i++) {
		float ferror_pct = 100.0 / a[i] * fabs(b[i] - a[i]);
		if(ferror_pct > 5) {
			printf("Results failed verification at index %i with %f pct deviation\n", i, ferror_pct);
			printf("Expected %f, calculated %f\n\n", a[i], b[i]);
			return;
		} else if (ferror_pct > max_ferror)
			max_ferror = ferror_pct;
	}

	printf("Result passed verification. Max ferror %f%%\n\n", max_ferror);
}
