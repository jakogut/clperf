#include "cpu_bench.h"

#include <stdlib.h>
#include <pthread.h>
#include <string.h>

static float *cpu_result_matrix(struct bench_buf *in)
{
	float *res = aligned_alloc(16, BUFFER_SIZE * sizeof(float));

	#pragma omp parallel for
	for (unsigned i = 0; i < BUFFER_SIZE; i++) {
		float aa = in->a[i], ba = in->b[i], ca = in->c[i],
		      ab = in->a[i], bb = in->b[i], cb = in->c[i],
		      ac = in->a[i], bc = in->b[i], cc = in->c[i],
		      ad = in->a[i], bd = in->b[i], cd = in->c[i];

		for (unsigned j = 0; j < ROUNDS_PER_ITERATION; j++) {
			ba += aa * ((ba * ca) + ba);
			bb += ab * ((bb * cb) + bb);
			bc += ac * ((bc * cc) + bc);
			bd += ad * ((bd * cd) + bd);

			ca += ba * ((ca * aa) + ca);
			cb += bb * ((cb * ab) + cb);
			cc += bc * ((cc * ac) + cc);
			cd += bd * ((cd * ad) + cd);

			aa += ca * ((aa * ba) + aa);
			ab += cb * ((ab * bb) + ab);
			ac += cc * ((ac * bc) + ac);
			ad += cd * ((ad * bd) + ad);

			res[i] += aa * ab + ac * ad;
			res[i] += aa * ab + ac * ad;
			res[i] += aa * ab + ac * ad;
			res[i] += aa * ab + ac * ad;
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
