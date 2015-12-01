/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.juicer.hiccups;

import juicebox.tools.clt.juicer.HiCCUPS;

/**
 * Created by muhammadsaadshamim on 5/11/15.
 */
class GPUKernel {

    public static final String kernelName = "BasicPeakCallingKernel";

    /**
     * hiccups Kernel Code
     *
     * @param window
     * @param matrixSize
     * @param peakWidth
     * @param divisor
     * @return CUDA kernel code as string
     */
    public static String kernelCode(int window, int matrixSize, int peakWidth, int divisor) {
        return "extern \"C\"\n" +
                "__global__ void BasicPeakCallingKernel(float *c, float *expectedbl, float *expecteddonut, float *expectedh, float *expectedv, float *observed, float *b_bl, float *b_donut, float *b_h, float *b_v, float *p, float *tbl, float *td, float *th, float *tv, float *d, float *kr1, float *kr2, float *bound1, float *bound3)\n" +
                "{\n" +
                "    // 2D Thread ID \n" +
                "    int t_col = threadIdx.x + blockIdx.x * blockDim.x;\n" +
                "    int t_row = threadIdx.y + blockIdx.y * blockDim.y;\n" +

                "    // Evalue is used to store the element of the matrix\n" +
                "    // that is computed by the thread\n" +
                "    float Evalue_bl =  0;\n" +
                "    float Edistvalue_bl = 0;\n" +
                "    float Evalue_donut =  0;\n" +
                "    float Edistvalue_donut = 0;\n" +
                "    float Evalue_h =  0;\n" +
                "    float Edistvalue_h = 0;\n" +
                "    float Evalue_v =  0;\n" +
                "    float Edistvalue_v = 0;\n" +
                "    float e_bl = 0;\n" +
                "    float e_donut = 0;\n" +
                "    float e_h = 0;\n" +
                "    float e_v = 0;\n" +
                "    float o = 0;\n" +
                "    float sbtrkt = 0;\n" +
                "    float bvalue_bl = 0;\n" +
                "    float bvalue_donut = 0;\n" +
                "    float bvalue_h = 0;\n" +
                "    float bvalue_v = 0;\n" +
                "    int wsize = " + window + ";\n" +
                "    int msize = " + matrixSize + ";\n" +
                "    int pwidth = " + peakWidth + ";\n" +
                "    int buffer_width = " + HiCCUPS.regionMargin + ";\n" +
                "    //int dvisor = " + divisor + ";\n" +
                "    int diff = bound1[0] - bound3[0];\n" +
                "    int diagDist = abs(t_row+diff-t_col);\n" +
                "    int maxIndex = msize-buffer_width;" +

                "    wsize = min(wsize, (abs(t_row+diff-t_col)-1)/2);\n" +
                "    if (wsize <= pwidth) {\n" +
                "        wsize = pwidth + 1;\n" +
                "    }\n" +
                "    wsize = min(wsize, buffer_width);\n" +

                "    // only run if within central window (not in data buffer margins)\n" +
                "    if (t_row >= buffer_width && t_row<maxIndex && t_col>= buffer_width && t_col<maxIndex) {\n" +
                "        \n" +
                "        // calculate initial bottom left box\n" +
                "        for (int i = t_row+1; i <= t_row+wsize; i++) {\n" +
                "            for (int j = t_col-wsize; j < t_col; j++) {\n" +
                "                int index = i * msize + j;\n" +
                "                if (!isnan(c[index])) {     \n" +
                "                    if (i+diff-j<0) {\n" +
                "                        Evalue_bl += c[index];\n" +
                "                        Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the middle peak\n" +
                "        for (int i = t_row+1; i <= t_row+pwidth; i++) {\n" +
                "            for (int j = t_col-pwidth; j < t_col; j++) {\n" +
                "                int index = i * msize + j;\n" +
                "                if (!isnan(c[index])) {     \n" +
                "                    if (i+diff-j<0) {\n" +
                "                        Evalue_bl -= c[index];\n" +
                "                        Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +
                "        }\n" +

                "        //fix box dimensions\n" +
                "        while (Evalue_bl<16) {\n" +
                "            Evalue_bl =0;\n" +
                "            Edistvalue_bl =0;\n" +
                "            wsize+=1;\n" +
                "            //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);\n" +
                "            for (int i = t_row+1; i <= t_row+wsize; i++) {\n" +
                "                for (int j = t_col-wsize; j < t_col; j++) {\n" +
                "                    int index = i * msize + j;\n" +
                "                    if (!isnan(c[index])) {     \n" +
                "                        if (i+diff-j<0) {\n" +
                "                            Evalue_bl += c[index];\n" +
                "                            Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                            if (i>= t_row+1) {\n" +
                "                                if (i<t_row+pwidth+1) {\n" +
                "                                    if (j>= t_col-pwidth) {\n" +
                "                                        if (j<t_col) {\n" +
                "                                            Evalue_bl -= c[index];\n" +
                "                                            Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                                        }\n" +
                "                                    }\n" +
                "                                }\n" +
                "                            }\n" +
                "                        }\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +

                "            if (wsize >= buffer_width) {\n" +
                "                break;\n" +
                "            }\n" +
                "            if (2*wsize>= abs(t_row+diff-t_col)) {\n" +
                "                break;\n" +
                "            }\n" +
                "        }\n" +

                "        // calculate donut\n" +
                "        for (int i = t_row-wsize; i <= t_row+wsize; ++i) {\n" +
                "            for (int j = t_col-wsize; j <= t_col+wsize; ++j) {\n" +
                "                int index = i * msize + j;\n" +
                "                if (!isnan(c[index])) {     \n" +
                "                    if (i+diff-j<0) {\n" +
                "                        Evalue_donut += c[index];\n" +
                "                        Edistvalue_donut += d[abs(i+diff-j)];\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the middle peak\n" +
                "        for (int i = t_row-pwidth; i <= t_row+pwidth; ++i) {\n" +
                "            for (int j = t_col-pwidth; j <= t_col+pwidth; ++j) {\n" +
                "                int index = i * msize + j;\n" +
                "                if (!isnan(c[index])) {     \n" +
                "                    if (i+diff-j<0) {\n" +
                "                        Evalue_donut -= c[index];\n" +
                "                        Edistvalue_donut -= d[abs(i+diff-j)];\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the cross hairs left side\n" +
                "        for (int i = t_row-wsize; i < t_row-pwidth; i++) {\n" +
                "            int index = i * msize + t_col;\n" +
                "            if (!isnan(c[index])) {     \n" +
                "                Evalue_donut -= c[index];\n" +
                "                Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "            }\n" +
                "            for (int j = -1; j <=1; j++) {\n" +
                "                Evalue_v += c[index + j];\n" +
                "                Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the cross hairs right side\n" +
                "        for (int i = t_row+pwidth+1; i <= t_row+wsize; ++i) {\n" +
                "            int index = i * msize + t_col;\n" +
                "            if (!isnan(c[index])) {     \n" +
                "                Evalue_donut -= c[index];\n" +
                "                Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "            }\n" +
                "            for (int j = -1; j <=1 ; ++j) {\n" +
                "                Evalue_v += c[index + j];\n" +
                "                Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the cross hairs top side\n" +
                "        for (int j = t_col-wsize; j < t_col-pwidth; ++j) {\n" +
                "            int index = t_row * msize + j;\n" +
                "            if (!isnan(c[index])) {     \n" +
                "                Evalue_donut -= c[index];\n" +
                "                Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "            }\n" +
                "            for (int i = -1; i <=1 ; ++i) {\n" +
                "                Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "            }\n" +
                "        }\n" +
                "        //Subtract off the cross hairs bottom side\n" +
                "        for (int j = t_col+pwidth+1; j <= t_col+wsize; ++j) {\n" +
                "            int index = t_row * msize + j;\n" +
                "            if (!isnan(c[index])) {     \n" +
                "                Evalue_donut -= c[index];\n" +
                "                Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "            }\n" +
                "            for (int i = -1; i <=1 ; ++i) {\n" +
                "                Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "            }\n" +
                "        }\n" +
                "    }\n" +

                "    e_bl = ((Evalue_bl*d[diagDist])/Edistvalue_bl)*kr1[t_row]*kr2[t_col];\n" +
                "    e_donut = ((Evalue_donut*d[diagDist])/Edistvalue_donut)*kr1[t_row]*kr2[t_col];\n" +
                "    e_h = ((Evalue_h*d[diagDist])/Edistvalue_h)*kr1[t_row]*kr2[t_col];\n" +
                "    e_v = ((Evalue_v*d[diagDist])/Edistvalue_v)*kr1[t_row]*kr2[t_col];\n" +

                "    float lognorm = logf(powf(2.0,.33));\n" +
                "    if (!isnan(e_bl)) {\n" +
                "        if (e_bl<=1) {\n" +
                "            bvalue_bl = 0;\n" +
                "        }\n" +
                "        else {\n" +
                "            bvalue_bl = floorf(logf(e_bl)/lognorm);\n" +
                "        }\n" +
                "    }\n" +
                "    if (!isnan(e_donut)) {\n" +
                "        if (e_donut<=1) {\n" +
                "            bvalue_donut = 0;\n" +
                "        }\n" +
                "        else {\n" +
                "            bvalue_donut = floorf(logf(e_donut)/lognorm);\n" +
                "        }\n" +
                "    }\n" +
                "    if (!isnan(e_h)) {\n" +
                "        if (e_h<=1) {\n" +
                "            bvalue_h = 0;\n" +
                "        }\n" +
                "        else {\n" +
                "            bvalue_h = floorf(logf(e_h)/lognorm);\n" +
                "        }\n" +
                "    }\n" +
                "    if (!isnan(e_v)) {\n" +
                "        if (e_v<=1) {\n" +
                "            bvalue_v = 0;\n" +
                "        }\n" +
                "        else {\n" +
                "            bvalue_v = floorf(logf(e_v)/lognorm);\n" +
                "        }\n" +
                "    }\n" +

                "    // Write the matrix to device memory;\n" +
                "    // each thread writes one element\n" +
                "    int val_index = t_row * msize + t_col;\n" +
                "    expectedbl[val_index] = e_bl;\n" +
                "    expecteddonut[val_index] = e_donut;\n" +
                "    expectedh[val_index] = e_h;\n" +
                "    expectedv[val_index] = e_v;\n" +
                "    o = roundf(c[val_index]*kr1[t_row]*kr2[t_col]);\n" +
                "    observed[val_index] = o; //roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);\n" +
                "    b_bl[val_index] = bvalue_bl;\n" +
                "    b_donut[val_index] = bvalue_donut;\n" +
                "    b_h[val_index] = bvalue_h;\n" +
                "    b_v[val_index] = bvalue_v;\n" +
                "    sbtrkt = fmaxf(tbl[(int) bvalue_bl],td[(int) bvalue_donut]);\n" +
                "    sbtrkt = fmaxf(sbtrkt, th[(int) bvalue_h]);\n" +
                "    sbtrkt = fmaxf(sbtrkt, tv[(int) bvalue_v]);\n" +
                "    p[val_index] = o-sbtrkt;\n" +
                "}";
    }
}
