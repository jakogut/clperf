__kernel void matrix_fmadd (__global const float* a,
			    __global const float* b,
			    __global const float* c,
			    __global float* res,
			    const unsigned buff_size)
{
	const int i = get_global_id(0);
	if(i < buff_size) {
		float r = 0, al = a[i], bl = b[i], cl = c[i];

		#pragma unroll
		for(int i = 0; i < 512; i++) {
			bl += al * ((bl * cl) + bl);
			cl += bl * ((cl * al) + cl);
			al += cl * ((al * bl) + al);

			bl += al * ((bl * cl) + bl);
			cl += bl * ((cl * al) + cl);
			al += cl * ((al * bl) + al);

			bl += al * ((bl * cl) + bl);
			cl += bl * ((cl * al) + cl);
			al += cl * ((al * bl) + al);

			bl += al * ((bl * cl) + bl);
			cl += bl * ((cl * al) + cl);
			al += cl * ((al * bl) + al);

			r += a + b + c;
		}

		res[i] = r;
	}
}
