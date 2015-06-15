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
		for(int i = 0; i < 12; i++) {
			r += al + (bl * cl) + bl; r += bl + (cl * al) + cl; r += cl + (al * bl) + al; r += al + (bl * cl) + bl;
			r += al + (bl * cl) + bl; r += bl + (cl * al) + cl; r += cl + (al * bl) + al; r += al + (bl * cl) + bl;
			r += al + (bl * cl) + bl; r += bl + (cl * al) + cl; r += cl + (al * bl) + al; r += al + (bl * cl) + bl;
			r += al + (bl * cl) + bl; r += bl + (cl * al) + cl; r += cl + (al * bl) + al; r += al + (bl * cl) + bl;
		}

		res[i] = r;
	}
}
