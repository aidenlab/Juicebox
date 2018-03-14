extern "C"
__global__ void BasicPeakCallingKernel(float *c, float *expectedbl, float *expecteddonut, float *expectedh, float *expectedv, float *observed, float *b_bl, float *b_donut, float *b_h, float *b_v, float *p, float *tbl, float *td, float *th, float *tv, float *d, float *kr1, float *kr2, float *bound1, float *bound3)
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
	float sbtrkt = 0;
	float bvalue_bl = 0;
	float bvalue_donut = 0;
	float bvalue_h = 0;
	float bvalue_v = 0;
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
	    //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);
			for (int i = t_row+1; i <= t_row+wsize; i++) {
				for (int j = t_col-wsize; j < t_col; j++) {
					int index = i * msize + j;
					if (!isnan(c[index])) {
						if (i+diff-j<0) {
							Evalue_bl += c[index];
							Edistvalue_bl += d[abs(i+diff-j)];
							if (i>= t_row+1) {
								if (i<t_row+pwidth+1) {
									if (j>= t_col-pwidth) {
										if (j<t_col) {
											Evalue_bl -= c[index];
											Edistvalue_bl -= d[abs(i+diff-j)];
										}
									}
								}
							}
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
		for (int i = t_row-wsize; i < t_row-pwidth; i++) {
			int index = i * msize + t_col;
			if (!isnan(c[index])) {
				Evalue_donut -= c[index];
				Edistvalue_donut -= d[abs(i+diff-t_col)];
			}
			for (int j = -1; j <=1; j++) {
				Evalue_v += c[index + j];
				Edistvalue_v += d[abs(i+diff-t_col-j)];
			}
		}
		//Subtract off the cross hairs right side
		for (int i = t_row+pwidth+1; i <= t_row+wsize; ++i) {
			int index = i * msize + t_col;
			if (!isnan(c[index])) {
				Evalue_donut -= c[index];
				Edistvalue_donut -= d[abs(i+diff-t_col)];
			}
			for (int j = -1; j <=1 ; ++j) {
				Evalue_v += c[index + j];
				Edistvalue_v += d[abs(i+diff-t_col-j)];
			}
		}
		//Subtract off the cross hairs top side
		for (int j = t_col-wsize; j < t_col-pwidth; ++j) {
			int index = t_row * msize + j;
			if (!isnan(c[index])) {
				Evalue_donut -= c[index];
				Edistvalue_donut -= d[abs(t_row+diff-j)];
			}
			for (int i = -1; i <=1 ; ++i) {
				Evalue_h += c[(t_row+i) * msize + j];
				Edistvalue_h += d[abs(t_row+i+diff-j)];
			}
		}
		//Subtract off the cross hairs bottom side
  	for (int j = t_col+pwidth+1; j <= t_col+wsize; ++j) {
			int index = t_row * msize + j;
			if (!isnan(c[index])) {
				Evalue_donut -= c[index];
				Edistvalue_donut -= d[abs(t_row+diff-j)];
			}
			for (int i = -1; i <=1 ; ++i) {
				Evalue_h += c[(t_row+i) * msize + j];
				Edistvalue_h += d[abs(t_row+i+diff-j)];
			}
		}
	}

	e_bl = ((Evalue_bl*d[diagDist])/Edistvalue_bl)*kr1[t_row]*kr2[t_col];
	e_donut = ((Evalue_donut*d[diagDist])/Edistvalue_donut)*kr1[t_row]*kr2[t_col];
	e_h = ((Evalue_h*d[diagDist])/Edistvalue_h)*kr1[t_row]*kr2[t_col];
	e_v = ((Evalue_v*d[diagDist])/Edistvalue_v)*kr1[t_row]*kr2[t_col];

	float lognorm = logf(powf(2.0,.33));
	if (!isnan(e_bl) && !isinf(e_bl)) {
		if (e_bl<=1) {
			bvalue_bl = 0;
		}
		else {
			bvalue_bl = floorf(logf(e_bl)/lognorm);
		}
	}
	if (!isnan(e_donut) && !isinf(e_donut)) {
		if (e_donut<=1) {
			bvalue_donut = 0;
		}
		else {
			bvalue_donut = floorf(logf(e_donut)/lognorm);
		}
	}
	if (!isnan(e_h) && !isinf(e_h)) {
		if (e_h<=1) {
			bvalue_h = 0;
		}
		else {
			bvalue_h = floorf(logf(e_h)/lognorm);
		}
	}
	if (!isnan(e_v) && !isinf(e_v)) {
		if (e_v<=1) {
			bvalue_v = 0;
		}
		else {
			bvalue_v = floorf(logf(e_v)/lognorm);
		}
	}

	bvalue_bl = min((int)bvalue_bl,  HiCCUPS_W1_MAX_INDX );
	bvalue_donut = min((int)bvalue_donut,  HiCCUPS_W1_MAX_INDX );
	bvalue_h = min((int)bvalue_h,  HiCCUPS_W1_MAX_INDX );
	bvalue_v = min((int)bvalue_v,  HiCCUPS_W1_MAX_INDX );

  // Write the matrix to device memory;
  // each thread writes one element
	int val_index = t_row * msize + t_col;
	expectedbl[val_index] = e_bl;
	expecteddonut[val_index] = e_donut;
	expectedh[val_index] = e_h;
	expectedv[val_index] = e_v;
	o = roundf(c[val_index]*kr1[t_row]*kr2[t_col]);
  observed[val_index] = o; //roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);
  b_bl[val_index] = bvalue_bl;
  b_donut[val_index] = bvalue_donut;
  b_h[val_index] = bvalue_h;
  b_v[val_index] = bvalue_v;
  sbtrkt = fmaxf(tbl[(int) bvalue_bl],td[(int) bvalue_donut]);
  sbtrkt = fmaxf(sbtrkt, th[(int) bvalue_h]);
  sbtrkt = fmaxf(sbtrkt, tv[(int) bvalue_v]);
  p[val_index] = o-sbtrkt;
}