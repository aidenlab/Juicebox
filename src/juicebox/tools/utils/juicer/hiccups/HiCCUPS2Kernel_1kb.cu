/**
 * Returns the incomplete gamma function P(a,x), evaluated by its series
 * representation. Assumes a > 0 and x >= 0.
 */
__device__ float gser(float a, float x){
   int GAMMA_ITMAX = 200;
   float GAMMA_EPS = 2.22e-16;
   float ap, del, sum;
   int i;

   ap = a;
   del = 1.0/a;
   sum = del;
   for (i = 1; i <= GAMMA_ITMAX; ++ i)
      {
      ap += 1.0;
      del *= x/ap;
      sum += del;
      if (fabsf(del) < fabsf(sum)*GAMMA_EPS)
         {
         return sum*expf(-x + a*logf(x) - lgammaf(a));
         }
      }
   return 1.0; // Too many iterations
}

/**
 * Returns the complementary incomplete gamma function Q(a,x), evaluated by its
 * continued fraction representation. Assumes a > 0 and x >= 0.
 */
__device__ float gcf(float a, float x){
   int GAMMA_ITMAX = 200;
   float GAMMA_EPS = 2.22e-16;
   float GAMMA_FPMIN = (1.18e-38/GAMMA_EPS);
   float b, c, d, h, an, del;
   int i;

   b = x + 1.0 - a;
   c = 1.0/GAMMA_FPMIN;
   d = 1.0/b;
   h = d;
   for (i = 1; i <= GAMMA_ITMAX; ++ i)
      {
      an = -i*(i - a);
      b += 2.0;
      d = an*d + b;
      if (fabsf(d) < GAMMA_FPMIN) d = GAMMA_FPMIN;
      c = b + an/c;
      if (fabsf(c) < GAMMA_FPMIN) c = GAMMA_FPMIN;
      d = 1.0/d;
      del = d*c;
      h *= del;
      if (fabsf(del - 1.0) < GAMMA_EPS)
         {
         return expf(-x + a*logf(x) - lgammaf(a))*h;
         }
      }
   return 0.0; // Too many iterations
}

/**
 * Returns the incomplete gamma function P(a,x).
 */
__device__ float gammp(float a, float x){
   //if (a <= 0.0) die ("gammp(): a illegal");
   //if (x < 0.0) die ("gammp(): x illegal");
   return x == 0.0 ? 0.0 : x < a + 1.0 ? gser(a,x) : 1.0 - gcf(a,x);
}

/**
 * Returns the complementary incomplete gamma function Q(a,x) = 1 - P(a,x).
 */
__device__ float gammq(float a, float x){
   //if (a <= 0.0) die ("gammq(): a illegal");
   //if (x < 0.0) die ("gammq(): x illegal");
   if (x <= powf(2.0,-3.33)) {
   return x == 0.0 ? 1.0 : x < a + 1.0 ? 1.0 - gser(a,powf(2.0,-3.33)) : gcf(a, powf(2.0,-3.33));
   } else {
   return x == 0.0 ? 1.0 : x < a + 1.0 ? 1.0 - gser(a,x) : gcf(a,x);
   }
}


__device__ void ensure_appropriate_values(float e_value, float lognormval, float* bvalue){
	if (!isnan(e_value) && !isinf(e_value)) {
		if (e_value<=powf(2.0,-3.33)) {
			*bvalue = 0;
		}
		else {
			*bvalue = floorf(logf(e_value)/lognormval)+11;
		}
	}

	*bvalue = (float) min((int) *bvalue,  HiCCUPS_W1_MAX_INDX );
}

__device__ void process_masks_lr(int i_start, int i_max_p1, int msize, int t_col, float *c,float *d, int diff,
	float* evalue_d, float* evalue_dist_d, float* evalue_v, float* evalue_dist_v){

	for (int i = i_start; i < i_max_p1; i++) {
		int index = i * msize + t_col;
		if (!isnan(c[index])) {
			*evalue_d -= c[index];
			*evalue_dist_d -= d[abs(i+diff-t_col)];
		}
		for (int j = -1; j < 2; j++) {
			*evalue_v += c[index + j];
			*evalue_dist_v += d[abs(i+diff-t_col-j)];
		}
	}
}

__device__ void process_masks_tb(int j_start, int j_max_p1, int msize, int t_row, float *c,float *d, int diff,
	float* evalue_d, float* evalue_dist_d, float* evalue_h, float* evalue_dist_h){

	for (int j = j_start; j < j_max_p1; j++) {
		int index = t_row * msize + j;
		if (!isnan(c[index])) {
			*evalue_d -= c[index];
			*evalue_dist_d -= d[abs(t_row+diff-j)];
		}
		for (int i = -1; i < 2; i++) {
			*evalue_h += c[(t_row+i) * msize + j];
			*evalue_dist_h += d[abs(t_row+i+diff-j)];
		}
	}
}

extern "C"
__global__ void BasicPeakCallingKernel(float *c, float *expectedbl, float *expecteddonut, float *expectedh,
	float *expectedv, float *observed, float *b_bl, float *b_donut, float *b_h, float *b_v, float *p,
	float *p_bl, float *p_donut, float *p_h, float *p_v,
	float *tbl, float *td, float *th, float *tv, float *d, float *kr1, float *kr2, float *bound1, float *bound3)
{
    // 2D Thread ID
	int t_col = threadIdx.x + blockIdx.x * blockDim.x;
	int t_row = threadIdx.y + blockIdx.y * blockDim.y;

    // Evalue is used to store the element of the matrix
    // that is computed by the thread
	float Evalue_bl =  0;
	float Edistvalue_bl = 0;
	float Evalue_donut =  0;
	float Edistvalue_donut = 0;
	float Evalue_h =  0;
	float Edistvalue_h = 0;
	float Evalue_v =  0;
	float Edistvalue_v = 0;
	float e_bl = 0;
	float e_donut = 0;
	float e_h = 0;
	float e_v = 0;
	float o = 0;
	float bvalue_bl = 0;
	float bvalue_donut = 0;
	float bvalue_h = 0;
	float bvalue_v = 0;
	float pvalue_bl = 1;
	float pvalue_donut = 1;
	float pvalue_h = 1;
	float pvalue_v = 1;
	int wsize =  HiCCUPS_WINDOW;
	int msize =  HiCCUPS_MATRIX_SIZE;
	int pwidth =  HiCCUPS_PEAK_WIDTH;
	int buffer_width =  HiCCUPS_REGION_MARGIN;
	int diff = bound1[0] - bound3[0];
	int diagDist = abs(t_row+diff-t_col);
	int maxIndex = msize-buffer_width;

	wsize = min(wsize, (abs(t_row+diff-t_col)-1)/2);
	if (wsize <= pwidth) {
		wsize = pwidth + 1;
	}
	wsize = min(wsize, buffer_width);

  // only run if within central window (not in data buffer margins)
	if (t_row >= buffer_width && t_row<maxIndex && t_col>= buffer_width && t_col<maxIndex) {

		// calculate initial bottom left box
		for (int i = t_row+1; i <= t_row+wsize; i++) {
			for (int j = t_col-wsize; j < t_col; j++) {
				int index = i * msize + j;
				if (!isnan(c[index])) {
					if (i+diff-j<0) {
						Evalue_bl += c[index];
						Edistvalue_bl += d[abs(i+diff-j)];
					}
				}
			}
		}

		//Subtract off the middle peak
		for (int i = t_row+1; i <= t_row+pwidth; i++) {
			for (int j = t_col-pwidth; j < t_col; j++) {
				int index = i * msize + j;
				if (!isnan(c[index])) {
					if (i+diff-j<0) {
						Evalue_bl -= c[index];
						Edistvalue_bl -= d[abs(i+diff-j)];
					}
				}
			}
		}

		//fix box dimensions
		while (Evalue_bl<16) {
			Evalue_bl =0;
			Edistvalue_bl =0;
			wsize+=1;
			for (int i = t_row+1; i <= t_row+wsize; i++) {
				for (int j = t_col-wsize; j < t_col; j++) {
					int index = i * msize + j;
					if (!isnan(c[index]) && i+diff-j<0) {
						Evalue_bl += c[index];
						Edistvalue_bl += d[abs(i+diff-j)];
						if (i > t_row && i < t_row+pwidth+1 && j > t_col-pwidth-1 && j < t_col) {
							Evalue_bl -= c[index];
							Edistvalue_bl -= d[abs(i+diff-j)];
						}
					}
				}
			}

			if (wsize >= buffer_width) {
				break;
			}
			if (2*wsize>= abs(t_row+diff-t_col)) {
				break;
			}
		}

		// calculate donut
		for (int i = t_row-wsize; i <= t_row+wsize; ++i) {
			for (int j = t_col-wsize; j <= t_col+wsize; ++j) {
				int index = i * msize + j;
				if (!isnan(c[index])) {
					if (i+diff-j<0) {
						Evalue_donut += c[index];
						Edistvalue_donut += d[abs(i+diff-j)];
					}
				}
			}
		}
		//Subtract off the middle peak
		for (int i = t_row-pwidth; i <= t_row+pwidth; ++i) {
			for (int j = t_col-pwidth; j <= t_col+pwidth; ++j) {
				int index = i * msize + j;
				if (!isnan(c[index])) {
					if (i+diff-j<0) {
						Evalue_donut -= c[index];
						Edistvalue_donut -= d[abs(i+diff-j)];
					}
				}
			}
		}

		//Subtract off the cross hairs left side
		process_masks_lr(t_row-wsize, t_row-pwidth, msize, t_col, c, d, diff, &Evalue_donut, &Edistvalue_donut, &Evalue_v, &Edistvalue_v);

		//Subtract off the cross hairs right side
		process_masks_lr(t_row+pwidth+1, t_row+wsize+1, msize, t_col, c, d, diff, &Evalue_donut, &Edistvalue_donut, &Evalue_v, &Edistvalue_v);

		//Subtract off the cross hairs top side
		process_masks_tb(t_col-wsize, t_col-pwidth, msize, t_row, c, d, diff, &Evalue_donut, &Edistvalue_donut, &Evalue_h, &Edistvalue_h);

		//Subtract off the cross hairs bottom side
		process_masks_tb(t_col+pwidth+1, t_col+wsize+1, msize, t_row, c, d, diff, &Evalue_donut, &Edistvalue_donut, &Evalue_h, &Edistvalue_h);


		e_bl = ((Evalue_bl*d[diagDist])/Edistvalue_bl)*kr1[t_row]*kr2[t_col];
		e_donut = ((Evalue_donut*d[diagDist])/Edistvalue_donut)*kr1[t_row]*kr2[t_col];
		e_h = ((Evalue_h*d[diagDist])/Edistvalue_h)*kr1[t_row]*kr2[t_col];
		e_v = ((Evalue_v*d[diagDist])/Edistvalue_v)*kr1[t_row]*kr2[t_col];

		float lognorm = logf(powf(2.0,0.33));

		ensure_appropriate_values(e_bl, lognorm, &bvalue_bl);
		ensure_appropriate_values(e_donut, lognorm, &bvalue_donut);
		ensure_appropriate_values(e_h, lognorm, &bvalue_h);
		ensure_appropriate_values(e_v, lognorm, &bvalue_v);

		int val_index = t_row * msize + t_col;
		o = roundf(c[val_index]*kr1[t_row]*kr2[t_col]);
        pvalue_bl = 1 - gammq(o, e_bl);
        pvalue_donut = 1 - gammq(o, e_donut);
        pvalue_h = 1 - gammq(o, e_h);
        pvalue_v = 1 - gammq(o, e_v);


  		// Write the matrix to device memory;
  		// each thread writes one element

		expectedbl[val_index] = e_bl;
		expecteddonut[val_index] = e_donut;
		expectedh[val_index] = e_h;
		expectedv[val_index] = e_v;

		observed[val_index] = o;
		b_bl[val_index] = bvalue_bl;
		b_donut[val_index] = bvalue_donut;
		b_h[val_index] = bvalue_h;
		b_v[val_index] = bvalue_v;

		if (pvalue_bl <= tbl[(int) bvalue_bl] && pvalue_donut <= td[(int) bvalue_donut] && pvalue_h <= th[(int) bvalue_h] && pvalue_v <= tv[(int) bvalue_v]) {
		    p[val_index] = 1;
		} else {
		    p[val_index] = 0;
		}
		p_bl[val_index] = pvalue_bl;
		p_donut[val_index] = pvalue_donut;
		p_h[val_index] = pvalue_h;
		p_v[val_index] = pvalue_v;
	}
}