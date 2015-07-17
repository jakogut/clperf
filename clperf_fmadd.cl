__kernel void matrix_fmadd (__global const float* a,
			    __global const float* b,
			    __global const float* c,
			    __global float* res,
			    const unsigned buff_size)
{
	const int i = get_global_id(0);
	if(i < buff_size) {
		float r = 0,
			aa = a[i], ba = b[i], ca = c[i],
			ab = a[i], bb = b[i], cb = c[i],
			ac = a[i], bc = b[i], cc = c[i],
			ad = a[i], bd = b[i], cd = c[i];

		#pragma unroll
		for(int i = 0; i < 512; i++) {
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

			r += aa * ab + ac * ad;
			r += aa * ab + ac * ad;
			r += aa * ab + ac * ad;
			r += aa * ab + ac * ad;
		}

		res[i] = r;
	}
}
