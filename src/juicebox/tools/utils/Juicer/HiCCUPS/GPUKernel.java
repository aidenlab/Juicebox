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

package juicebox.tools.utils.Juicer.HiCCUPS;

/**
 * Created by muhammadsaadshamim on 5/11/15.
 */
public class GPUKernel {

    public static String kernelName = "BasicPeakCallingKernel";

    /**
     * HiCCUPS Kernel Code
     *
     * @param window
     * @param matrixSize
     * @param peakWidth
     * @param divisor
     * @return CUDA kernel code as string
     */
    public static String kernelCode(int window, int matrixSize, int peakWidth, int divisor) {
        return "extern \"C\"\n" +
                "__global__ void BasicPeakCallingKernel(float *c, float *expectedbl, float *expecteddonut," +
                "float *expectedh, float *expectedv, float *observed, float *b_bl, float *b_donut," +
                "float *b_h, float *b_v, float *p, float *tbl, float *td, float *th, float *tv," +
                "float *d, float *kr1, float *kr2, float *bound1, float *bound3)\n" +
                "{\n" +
                "       // 2D Thread ID \n" +
                "       int t_col = threadIdx.x + blockIdx.x * blockDim.x;\n" +
                "       int t_row = threadIdx.y + blockIdx.y * blockDim.y;\n" +
                "" +
                "       // Evalue is used to store the element of the matrix\n" +
                "       // that is computed by the thread\n" +
                "       float Evalue_bl =  0;\n" +
                "" +
                "       float Edistvalue_bl = 0;\n" +
                "       float Evalue_donut =  0;\n" +
                "       float Edistvalue_donut = 0;\n" +
                "       float Evalue_h =  0;\n" +
                "       float Edistvalue_h = 0;\n" +
                "       float Evalue_v =  0;\n" +
                "       float Edistvalue_v = 0;\n" +
                "       float e_bl = 0;\n" +
                "       float e_donut = 0;\n" +
                "       float e_h = 0;\n" +
                "       float e_v = 0;\n" +
                "       float o = 0;\n" +
                "       float sbtrkt = 0;\n" +
                "       float bvalue_bl = 0;\n" +
                "       float bvalue_donut = 0;\n" +
                "       float bvalue_h = 0;\n" +
                "       float bvalue_v = 0;\n" +
                "       int wsize = " + window + ";\n" +
                "       int msize = " + matrixSize + ";\n" +
                "       int pwidth = " + peakWidth + ";\n" +
                "       //int dvisor = " + divisor + ";\n" +  // TODO remove?
                "       int diff = bound1[0] - bound3[0];\n" +
                "" +
                "       while (abs(t_row+diff-t_col)<=(2*wsize)) {\n" +
                "               wsize = wsize - 1;\n" +
                "       }\n" +
                "" +
                "       if (wsize<=pwidth) {\n" +
                "               wsize = pwidth + 1;\n" +
                "       }\n" +
                "" +
                "       if (t_row>=20&&t_row<=(msize-20)&&t_col>=20&&t_col<=(msize-20)) {\n" +
                "               // calculate initial bottom left box\n" +
                "               for (int i = max(0,t_row+1); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-wsize);\n" +
                "                       for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_bl += c[i * msize +j];\n" +
                "                                               Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //Subtract off the middle peak\n" +
                "               for (int i = max(0,t_row+1); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-pwidth);\n" +
                "                       for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_bl -= c[i * msize +j];\n" +
                "                                               Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //fix box dimensions\n" +
                "               while (Evalue_bl<16) {\n" +
                "                       Evalue_bl=0;\n" +
                "                       Edistvalue_bl=0;\n" +
                "                       wsize+=1;\n" +
                "                       //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);\n" + //TODO remove?
                "                       for (int i = max(0,t_row+1); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                               int test=max(0,t_col-wsize);\n" +
                "                               for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                                       if (!isnan(c[i * msize + j])) {  \n" +
                "                                               if (i+diff-j<0) {\n" +
                "                                                       Evalue_bl += c[i * msize +j];\n" +
                "                                                       Edistvalue_bl += d[abs(i+diff-j)];\n" +
                "                                                       if (i>=t_row+1) {\n" +
                "                                                               if (i<t_row+pwidth+1) {\n" +
                "                                                                       if (j>=t_col-pwidth) {\n" +
                "                                                                               if (j<t_col) {\n" +
                "                                                                                       Evalue_bl -= c[i * msize +j];\n" +
                "                                                                                       Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                                                                               }\n" +
                "                                                                       }\n" +
                "                                                                }\n" +
                "                                                       }\n" +
                "                                               }\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "" +
                "                       //Subtact off the middle peak\n" +
                "                       //for (int i = max(0,t_row+1); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       //int test=max(0,t_col-pwidth);\n" +
                "                       //for (int j = test; j < min(t_col, msize); ++j) {\n" +
                "                       //if (!isnan(c[i * msize + j])) {  \n" +
                "                       //if (i+diff-j<0) {\n" +
                "                       //Evalue_bl -= c[i * msize +j];\n" +
                "                       //Edistvalue_bl -= d[abs(i+diff-j)];\n" +
                "                       //}\n" +
                "                       //}\n" +
                "                       //}\n" +
                "                       //}\n" +
                "" +
                "                       if (wsize == 20) {\n" +
                "                               break;\n" +
                "                       }\n" +
                "                       if (2*wsize>=abs(t_row+diff-t_col)) {\n" +
                "                               break;\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               // calculate donut\n" +
                "               for (int i = max(0,t_row-wsize); i < min(t_row+wsize+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-wsize);\n" +
                "                       for (int j = test; j < min(t_col+wsize+1, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_donut += c[i * msize +j];\n" +
                "                                               Edistvalue_donut += d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "" +
                "               //Subtract off the middle peak\n" +
                "               for (int i = max(0,t_row-pwidth); i < min(t_row+pwidth+1, msize); ++i) {\n" +
                "                       int test=max(0,t_col-pwidth);\n" +
                "                       for (int j = test; j < min(t_col+pwidth+1, msize); ++j) {\n" +
                "                               if (!isnan(c[i * msize + j])) {  \n" +
                "                                       if (i+diff-j<0) {\n" +
                "                                               Evalue_donut -= c[i * msize +j];\n" +
                "                                               Edistvalue_donut -= d[abs(i+diff-j)];\n" +
                "                                       }\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               //Subtract off the cross hairs\n" +
                "               if ((t_row-pwidth)>0) {\n" +
                "                       for (int i = max(0,t_row-wsize); i < (t_row-pwidth); ++i) {\n" +
                "                               if (!isnan(c[i * msize + t_col])) {  \n" +
                "                                       Evalue_donut -= c[i * msize + t_col];\n" +
                "                                       Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "                               }\n" +
                "                               for (int j = -1; j <=1 ; ++j) {\n" +
                "                                       Evalue_v += c[i * msize + t_col + j];\n" +
                "                                       Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_row+pwidth)<msize) {\n" +
                "                       for (int i = (t_row+pwidth+1); i < min(t_row+wsize+1,msize); ++i) {\n" +
                "                               if (!isnan(c[i * msize + t_col])) {  \n" +
                "                                       Evalue_donut -= c[i * msize + t_col];\n" +
                "                                       Edistvalue_donut -= d[abs(i+diff-t_col)];\n" +
                "                               }\n" +
                "                               for (int j = -1; j <=1 ; ++j) {\n" +
                "                                       Evalue_v += c[i * msize + t_col + j];\n" +
                "                                       Edistvalue_v += d[abs(i+diff-t_col-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_col-pwidth)>0) {\n" +
                "                       for (int j = max(0,t_col-wsize); j < (t_col-pwidth); ++j) {\n" +
                "                               if (!isnan(c[t_row * msize + j])) {  \n" +
                "                                       Evalue_donut -= c[t_row * msize + j];\n" +
                "                                       Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "                               }\n" +
                "                               for (int i = -1; i <=1 ; ++i) {\n" +
                "                                       Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                                       Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "               if ((t_col+pwidth)<msize) {\n" +
                "                       for (int j = (t_col+pwidth+1); j < min(t_col+wsize+1,msize); ++j) {\n" +
                "                               if (!isnan(c[t_row * msize + j])) {  \n" +
                "                                       Evalue_donut -= c[t_row * msize + j];\n" +
                "                                       Edistvalue_donut -= d[abs(t_row+diff-j)];\n" +
                "                               }\n" +
                "                               for (int i = -1; i <=1 ; ++i) {\n" +
                "                                       Evalue_h += c[(t_row+i) * msize + j];\n" +
                "                                       Edistvalue_h += d[abs(t_row+i+diff-j)];\n" +
                "                               }\n" +
                "                       }\n" +
                "               }\n" +
                "       }\n" +
                "" +
                "       //if (t_row+diff-t_col<(-1*pwidth)-2) {\n" +
                "       e_bl = ((Evalue_bl*d[abs(t_row+diff-t_col)])/Edistvalue_bl)*kr1[t_row]*kr2[t_col];\n" +
                "       e_donut = ((Evalue_donut*d[abs(t_row+diff-t_col)])/Edistvalue_donut)*kr1[t_row]*kr2[t_col];\n" +
                "       e_h = ((Evalue_h*d[abs(t_row+diff-t_col)])/Edistvalue_h)*kr1[t_row]*kr2[t_col];\n" +
                "       e_v = ((Evalue_v*d[abs(t_row+diff-t_col)])/Edistvalue_v)*kr1[t_row]*kr2[t_col];\n" +
                "" +
                "       if (!isnan(e_bl)) {\n" +
                "               if (e_bl<=1) {\n" +
                "                       bvalue_bl = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_bl = floorf(logf(e_bl)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_donut)) {\n" +
                "               if (e_donut<=1) {\n" +
                "                       bvalue_donut = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_donut = floorf(logf(e_donut)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_h)) {\n" +
                "               if (e_h<=1) {\n" +
                "                       bvalue_h = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_h = floorf(logf(e_h)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "       if (!isnan(e_v)) {\n" +
                "               if (e_v<=1) {\n" +
                "                       bvalue_v = 0;\n" +
                "               }\n" +
                "               else {\n" +
                "                       bvalue_v = floorf(logf(e_v)/logf(powf(2.0,.33)));\n" +
                "               }\n" +
                "       }\n" +
                "" +
                "       // Write the matrix to device memory;\n" +
                "       // each thread writes one element\n" +
                "       expectedbl[t_row * msize + t_col] = e_bl;\n" +
                "       expecteddonut[t_row * msize + t_col] = e_donut;\n" +
                "       expectedh[t_row * msize + t_col] = e_h;\n" +
                "       expectedv[t_row * msize + t_col] = e_v;\n" +
                "       o = roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);\n" +
                "       observed[t_row * msize + t_col] = o; //roundf(c[t_row * msize + t_col]*kr1[t_row]*kr2[t_col]);\n" +
                "       b_bl[t_row * msize + t_col] = bvalue_bl;\n" +
                "       b_donut[t_row * msize + t_col] = bvalue_donut;\n" +
                "       b_h[t_row * msize + t_col] = bvalue_h;\n" +
                "       b_v[t_row * msize + t_col] = bvalue_v;\n" +
                "       sbtrkt = fmaxf(tbl[(int) bvalue_bl],td[(int) bvalue_donut]);\n" +
                "       sbtrkt = fmaxf(sbtrkt, th[(int) bvalue_h]);\n" +
                "       sbtrkt = fmaxf(sbtrkt, tv[(int) bvalue_v]);\n" +
                "       p[t_row * msize + t_col] = o-sbtrkt;\n" +
                "}";
    }
}
