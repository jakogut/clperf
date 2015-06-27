#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#define BUFFER_SIZE (2 << 20)
#define ROUNDS_PER_ITERATION 64

#include <time.h>

struct bench_buf
{
	float *a;
	float *b;
	float *c;
};

int nthreads(void);

float *rand_matrix(const size_t size);

unsigned timespec_to_nsec(const struct timespec *start,
			  const struct timespec *end);

void print_perf_stats(const double sec_elapsed);

void verify_result(float *a, float *b);

#endif
