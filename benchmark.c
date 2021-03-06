#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>

#include "benchmark.h"

float *rand_matrix(const size_t size)
{
	float *mat = calloc(size, sizeof(float));

	for (unsigned i = 0; i < size; i++)
		mat[i] = rand() / (float)RAND_MAX;

	return mat;
}

int nthreads(void)
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

unsigned timespec_to_nsec(const struct timespec *start,
			  const struct timespec *end)
{
	return (end->tv_nsec - start->tv_nsec) +
	      ((end->tv_sec - start->tv_sec) * 1000000000);
}

float rt_to_gops(const double rt)
{
	uint64_t total_flops = (uint64_t)BUFFER_SIZE * FLOPS_PER_ITERATION;

	return total_flops / (rt * 1000000000.0f);
}

void print_perf_stats(const double sec_elapsed)
{
	printf("%i Cycles, %i FLOP/iteration, %f sec elapsed\n%f GFLOPS\n\n",
		BUFFER_SIZE,
		FLOPS_PER_ITERATION,
		sec_elapsed,
		rt_to_gops(sec_elapsed));
}

void result_fail(int i, float ferror_pct)
{
	printf("Verification failed at %i with %f pct deviation\n",
		i, ferror_pct);
}

void result_pass(float max_ferror)
{
	printf("Result passed verification. Max ferror %f%%\n\n", max_ferror);
}

void verify_result(float *a, float *b)
{
	const float allowable_error = 1;
	float max_ferror = 0;

	for (unsigned i = 0; i < BUFFER_SIZE; i++) {

		float ferror_pct = 100.0 / a[i] * fabs(b[i] - a[i]);

		if (ferror_pct > allowable_error) {
			result_fail(i, ferror_pct);
			return;

		} else if (ferror_pct > max_ferror)
			max_ferror = ferror_pct;
	}

	result_pass(max_ferror);
}
