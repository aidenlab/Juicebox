/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.utils.KernelLauncher;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.tools.clt.juicer.HiCCUPS;
import juicebox.tools.clt.juicer.HiCCUPSRegionHandler;
import juicebox.tools.utils.common.ArrayTools;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class GPUController {

    private static final int blockSize = 16;  //number of threads in block

    private final KernelLauncher kernelLauncher;
    private final boolean useCPUVersionHiCCUPS;
    private final int windowCPU;
    private final int matrixSizeCPU;
    private final int peakWidthCPU;

    public GPUController(int window, int matrixSize, int peakWidth, boolean useCPUVersionHiCCUPS) {

        windowCPU = window;
        matrixSizeCPU = matrixSize;
        peakWidthCPU = peakWidth;
        this.useCPUVersionHiCCUPS = useCPUVersionHiCCUPS;

        if (useCPUVersionHiCCUPS) {
            kernelLauncher = null;
        } else {
            String kernelCode = readCuFile("HiCCUPSKernel.cu", window, matrixSize, peakWidth);
            kernelLauncher = KernelLauncher.compile(kernelCode, "BasicPeakCallingKernel");
            //KernelLauncher.create()

            //threads per block = block_size*block_size
            kernelLauncher.setBlockSize(blockSize, blockSize, 1);

            // for grid of blocks
            int gridSize = (int) Math.ceil(matrixSize * 1.0 / blockSize);
            kernelLauncher.setGridSize(gridSize, gridSize);
        }

    }

    private static String readCuFile(String fileName, int window, int matrixSize, int peakWidth) {
        StringBuilder contentBuilder = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(GPUController.class.getResourceAsStream(fileName)))) {
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                contentBuilder.append(sCurrentLine).append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        String cuFileText = contentBuilder.toString();

        cuFileText = cuFileText.replaceAll("HiCCUPS_WINDOW", "" + window);
        cuFileText = cuFileText.replaceAll("HiCCUPS_MATRIX_SIZE", "" + matrixSize);
        cuFileText = cuFileText.replaceAll("HiCCUPS_PEAK_WIDTH", "" + peakWidth);
        cuFileText = cuFileText.replaceAll("HiCCUPS_REGION_MARGIN", "" + (HiCCUPS.regionMargin));
        cuFileText = cuFileText.replaceAll("HiCCUPS_W1_MAX_INDX", "" + (HiCCUPS.w1 - 1));

        return cuFileText;
    }

    public GPUOutputContainer process(HiCCUPSRegionHandler regionHandler, HiCCUPSRegionContainer regionContainer, int matrixSize,
                                      float[] thresholdBL, float[] thresholdDonut, float[] thresholdH, float[] thresholdV,
                                      NormalizationType normalizationType, HiCZoom zoom)
            throws NegativeArraySizeException, IOException {

        MatrixZoomData zd = regionHandler.getZoomData(regionContainer, zoom);
        double[] normalizationVector = regionHandler.getNormalizationVector(regionContainer, zoom);
        double[] expectedVector = regionHandler.getExpectedVector(regionContainer, zoom);
        int[] rowBounds = regionContainer.getRowBounds();
        int[] columnBounds = regionContainer.getColumnBounds();

        RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd, rowBounds[0], rowBounds[1],
                columnBounds[0], columnBounds[1], matrixSize, matrixSize, normalizationType, false);


        float[] observedVals = Floats.toArray(Doubles.asList(MatrixTools.flattenedRowMajorOrderMatrix(localizedRegionData)));

        // slice KR vector to localized region
        float[] distanceExpectedKRVector = Floats.toArray(Doubles.asList(expectedVector));


        float[] kr1CPU = Floats.toArray(Doubles.asList(Arrays.copyOfRange(normalizationVector, rowBounds[0], rowBounds[1])));
        float[] kr2CPU = Floats.toArray(Doubles.asList(Arrays.copyOfRange(normalizationVector, columnBounds[0], columnBounds[1])));

        if (kr1CPU.length < matrixSize)
            kr1CPU = ArrayTools.padEndOfArray(kr1CPU, matrixSize, Float.NaN);
        if (kr2CPU.length < matrixSize)
            kr2CPU = ArrayTools.padEndOfArray(kr2CPU, matrixSize, Float.NaN);

        float[] boundRowIndex = new float[1];
        boundRowIndex[0] = rowBounds[0];
        float[] boundColumnIndex = new float[1];
        boundColumnIndex[0] = columnBounds[0];

        if (useCPUVersionHiCCUPS) {
            return runCPUVersion(localizedRegionData.getData(), distanceExpectedKRVector, kr1CPU, kr2CPU,
                    boundRowIndex, boundColumnIndex, thresholdBL, thresholdDonut, thresholdH, thresholdV,
                    rowBounds, columnBounds);
        }

        // transfer host (CPU) memory to device (GPU) memory
        CUdeviceptr observedKRGPU = GPUHelper.allocateInput(observedVals);
        CUdeviceptr expectedDistanceVectorGPU = GPUHelper.allocateInput(distanceExpectedKRVector);
        CUdeviceptr kr1GPU = GPUHelper.allocateInput(kr1CPU);
        CUdeviceptr kr2GPU = GPUHelper.allocateInput(kr2CPU);
        CUdeviceptr thresholdBLGPU = GPUHelper.allocateInput(thresholdBL);
        CUdeviceptr thresholdDonutGPU = GPUHelper.allocateInput(thresholdDonut);
        CUdeviceptr thresholdHGPU = GPUHelper.allocateInput(thresholdH);
        CUdeviceptr thresholdVGPU = GPUHelper.allocateInput(thresholdV);
        CUdeviceptr boundRowIndexGPU = GPUHelper.allocateInput(boundRowIndex);
        CUdeviceptr boundColumnIndexGPU = GPUHelper.allocateInput(boundColumnIndex);

        // create empty gpu arrays for the results
        int flattenedSize = matrixSize * matrixSize;
        CUdeviceptr expectedBLGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr expectedDonutGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr expectedHGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr expectedVGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr binBLGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr binDonutGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr binHGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr binVGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr observedGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);
        CUdeviceptr peakGPU = GPUHelper.allocateOutput(flattenedSize, Sizeof.FLOAT);

        // call the kernel on the card
        kernelLauncher.call(
                // inputs
                observedKRGPU,
                // output
                expectedBLGPU, expectedDonutGPU, expectedHGPU, expectedVGPU,
                observedGPU,
                binBLGPU, binDonutGPU, binHGPU, binVGPU,
                peakGPU,
                // thresholds
                thresholdBLGPU, thresholdDonutGPU, thresholdHGPU, thresholdVGPU,
                // distance expected
                expectedDistanceVectorGPU,
                // kr
                kr1GPU, kr2GPU,
                // bounds
                boundRowIndexGPU,
                boundColumnIndexGPU);

        // initialize memory to store GPU results
        float[] expectedBLResult = new float[flattenedSize];
        float[] expectedDonutResult = new float[flattenedSize];
        float[] expectedHResult = new float[flattenedSize];
        float[] expectedVResult = new float[flattenedSize];
        float[] binBLResult = new float[flattenedSize];
        float[] binDonutResult = new float[flattenedSize];
        float[] binHResult = new float[flattenedSize];
        float[] binVResult = new float[flattenedSize];
        float[] observedResult = new float[flattenedSize];
        float[] peakResult = new float[flattenedSize];

        // transfer device (GPU) memory to host (CPU) memory
        cuMemcpyDtoH(Pointer.to(expectedBLResult), expectedBLGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(expectedDonutResult), expectedDonutGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(expectedHResult), expectedHGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(expectedVResult), expectedVGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(binBLResult), binBLGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(binDonutResult), binDonutGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(binHResult), binHGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(binVResult), binVGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(observedResult), observedGPU, flattenedSize * Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(peakResult), peakGPU, flattenedSize * Sizeof.FLOAT);

        GPUHelper.freeUpMemory(new CUdeviceptr[]{observedKRGPU, expectedDistanceVectorGPU,
                kr1GPU, kr2GPU, thresholdBLGPU, thresholdDonutGPU, thresholdHGPU,
                thresholdVGPU, boundRowIndexGPU, boundColumnIndexGPU,
                expectedBLGPU, expectedDonutGPU, expectedHGPU, expectedVGPU,
                binBLGPU, binDonutGPU, binHGPU, binVGPU,
                observedGPU, peakGPU});

        //long gpu_time2 = System.currentTimeMillis();
        //System.out.println("GPU Time: " + (gpu_time2-gpu_time1));

        int finalWidthX = rowBounds[5] - rowBounds[4];
        int finalWidthY = columnBounds[5] - columnBounds[4];

        // x2, y2 not inclusive here
        int x1 = rowBounds[2];
        int y1 = columnBounds[2];
        int x2 = x1 + finalWidthX;
        int y2 = y1 + finalWidthY;

        //System.out.println("flat = "+flattenedSize+" n = "+matrixSize+" x1 = "+x1+" x2 = "+x2+" y1 = "+y1+" y2 ="+y2);
        float[][] observedDenseCPU = GPUHelper.GPUArraytoCPUMatrix(observedResult, matrixSize, x1, x2, y1, y2);
        float[][] peakDenseCPU = GPUHelper.GPUArraytoCPUMatrix(peakResult, matrixSize, x1, x2, y1, y2);
        float[][] binBLDenseCPU = GPUHelper.GPUArraytoCPUMatrix(binBLResult, matrixSize, x1, x2, y1, y2);
        float[][] binDonutDenseCPU = GPUHelper.GPUArraytoCPUMatrix(binDonutResult, matrixSize, x1, x2, y1, y2);
        float[][] binHDenseCPU = GPUHelper.GPUArraytoCPUMatrix(binHResult, matrixSize, x1, x2, y1, y2);
        float[][] binVDenseCPU = GPUHelper.GPUArraytoCPUMatrix(binVResult, matrixSize, x1, x2, y1, y2);
        float[][] expectedBLDenseCPU = GPUHelper.GPUArraytoCPUMatrix(expectedBLResult, matrixSize, x1, x2, y1, y2);
        float[][] expectedDonutDenseCPU = GPUHelper.GPUArraytoCPUMatrix(expectedDonutResult, matrixSize, x1, x2, y1, y2);
        float[][] expectedHDenseCPU = GPUHelper.GPUArraytoCPUMatrix(expectedHResult, matrixSize, x1, x2, y1, y2);
        float[][] expectedVDenseCPU = GPUHelper.GPUArraytoCPUMatrix(expectedVResult, matrixSize, x1, x2, y1, y2);

        return new GPUOutputContainer(observedDenseCPU, peakDenseCPU,
                binBLDenseCPU, binDonutDenseCPU, binHDenseCPU, binVDenseCPU,
                expectedBLDenseCPU, expectedDonutDenseCPU, expectedHDenseCPU, expectedVDenseCPU);
    }

    /**
     * just a direct implementation from the CUDA kernel
     * todo clean up and optimize after debugging and testing
     */
    private GPUOutputContainer runCPUVersion(double[][] c, float[] d,
                                             float[] kr1, float[] kr2,
                                             float[] bound1, float[] bound3,
                                             float[] thresholdBL, float[] thresholdDonut,
                                             float[] thresholdH, float[] thresholdV,
                                             int[] rowBounds, int[] columnBounds) {

        float[][] observedDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] peakDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] binBLDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] binDonutDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] binHDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] binVDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] expectedBLDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] expectedDonutDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] expectedHDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];
        float[][] expectedVDenseCPU = new float[matrixSizeCPU][matrixSizeCPU];

        //int t_col = threadIdx.x + blockIdx.x * blockDim.x;\n" +
        //"    int t_row = threadIdx.y + blockIdx.y * blockDim.y;\n" +

        for (int t_row = HiCCUPS.regionMargin; t_row < matrixSizeCPU - HiCCUPS.regionMargin; t_row++) {
            for (int t_col = HiCCUPS.regionMargin; t_col < matrixSizeCPU - HiCCUPS.regionMargin; t_col++) {

                // Evalue is used to store the element of the matrix


                // that is computed by the thread
                float Evalue_bl = 0;
                float Edistvalue_bl = 0;
                float Evalue_donut = 0;
                float Edistvalue_donut = 0;
                float Evalue_h = 0;
                float Edistvalue_h = 0;
                float Evalue_v = 0;
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

                int wsize = windowCPU;
                int msize = matrixSizeCPU;
                int pwidth = peakWidthCPU;
                int buffer_width = HiCCUPS.regionMargin;

                int diff = (int) (bound1[0] - bound3[0]);
                int diagDist = Math.abs(t_row + diff - t_col);
                int maxIndex = msize - buffer_width;

                wsize = Math.min(wsize, (diagDist - 1) / 2);
                if (wsize <= pwidth) {
                    wsize = pwidth + 1;
                }
                wsize = Math.min(wsize, buffer_width);

                // only run if within central window (not in data buffer margins)
                if (t_row >= buffer_width && t_row < maxIndex && t_col >= buffer_width && t_col < maxIndex) {

                    // calculate initial bottom left box
                    for (int i = t_row + 1; i <= t_row + wsize; i++) {
                        for (int j = t_col - wsize; j < t_col; j++) {
                            if (!Double.isNaN(c[i][j])) {
                                if (i + diff - j < 0) {
                                    Evalue_bl += c[i][j];
                                    Edistvalue_bl += d[Math.abs(i + diff - j)];
                                }
                            }
                        }
                    }
                    //Subtract off the middle peak
                    for (int i = t_row + 1; i <= t_row + pwidth; i++) {
                        for (int j = t_col - pwidth; j < t_col; j++) {
                            if (!Double.isNaN(c[i][j])) {
                                if (i + diff - j < 0) {
                                    Evalue_bl -= c[i][j];
                                    Edistvalue_bl -= d[Math.abs(i + diff - j)];
                                }
                            }
                        }
                    }

                    //fix box dimensions
                    while (Evalue_bl < 16) {
                        Evalue_bl = 0;
                        Edistvalue_bl = 0;
                        wsize += 1;
                        //dvisor = powf(wsize,2.0) - powf(pwidth,2.0);
                        for (int i = t_row + 1; i <= t_row + wsize; i++) {
                            for (int j = t_col - wsize; j < t_col; j++) {
                                if (!Double.isNaN(c[i][j])) {
                                    if (i + diff - j < 0) {
                                        Evalue_bl += c[i][j];
                                        int distVal = Math.abs(i + diff - j);
                                        Edistvalue_bl += d[distVal];
                                        if (i >= t_row + 1) {
                                            if (i < t_row + pwidth + 1) {
                                                if (j >= t_col - pwidth) {
                                                    if (j < t_col) {
                                                        Evalue_bl -= c[i][j];
                                                        Edistvalue_bl -= d[distVal];
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
                        if (2 * wsize >= diagDist) {
                            break;
                        }
                    }

                    // calculate donut
                    for (int i = t_row - wsize; i <= t_row + wsize; ++i) {
                        for (int j = t_col - wsize; j <= t_col + wsize; ++j) {
                            if (!Double.isNaN(c[i][j])) {
                                if (i + diff - j < 0) {
                                    Evalue_donut += c[i][j];
                                    Edistvalue_donut += d[Math.abs(i + diff - j)];
                                }
                            }
                        }
                    }
                    //Subtract off the middle peak
                    for (int i = t_row - pwidth; i <= t_row + pwidth; ++i) {
                        for (int j = t_col - pwidth; j <= t_col + pwidth; ++j) {
                            if (!Double.isNaN(c[i][j])) {
                                if (i + diff - j < 0) {
                                    Evalue_donut -= c[i][j];
                                    Edistvalue_donut -= d[Math.abs(i + diff - j)];
                                }
                            }
                        }
                    }
                    //Subtract off the cross hairs left side
                    for (int i = t_row - wsize; i < t_row - pwidth; i++) {
                        if (!Double.isNaN(c[i][t_col])) {
                            Evalue_donut -= c[i][t_col];
                            Edistvalue_donut -= d[Math.abs(i + diff - t_col)];
                        }
                        for (int j = -1; j <= 1; j++) {
                            Evalue_v += c[i][t_col + j];
                            Edistvalue_v += d[Math.abs(i + diff - t_col - j)];
                        }
                    }
                    //Subtract off the cross hairs right side
                    for (int i = t_row + pwidth + 1; i <= t_row + wsize; ++i) {
                        if (!Double.isNaN(c[i][t_col])) {
                            Evalue_donut -= c[i][t_col];
                            Edistvalue_donut -= d[Math.abs(i + diff - t_col)];
                        }
                        for (int j = -1; j <= 1; ++j) {
                            Evalue_v += c[i][t_col + j];
                            Edistvalue_v += d[Math.abs(i + diff - t_col - j)];
                        }
                    }
                    //Subtract off the cross hairs top side
                    for (int j = t_col - wsize; j < t_col - pwidth; ++j) {
                        if (!Double.isNaN(c[t_row][j])) {
                            Evalue_donut -= c[t_row][j];
                            Edistvalue_donut -= d[Math.abs(t_row + diff - j)];
                        }
                        for (int i = -1; i <= 1; ++i) {
                            Evalue_h += c[t_row + i][j];
                            Edistvalue_h += d[Math.abs(t_row + i + diff - j)];
                        }
                    }
                    //Subtract off the cross hairs bottom side
                    for (int j = t_col + pwidth + 1; j <= t_col + wsize; ++j) {
                        if (!Double.isNaN(c[t_row][j])) {
                            Evalue_donut -= c[t_row][j];
                            Edistvalue_donut -= d[Math.abs(t_row + diff - j)];
                        }
                        for (int i = -1; i <= 1; ++i) {
                            Evalue_h += c[t_row + i][j];
                            Edistvalue_h += d[Math.abs(t_row + i + diff - j)];
                        }
                    }
                }

                e_bl = ((Evalue_bl * d[diagDist]) / Edistvalue_bl) * kr1[t_row] * kr2[t_col];
                e_donut = ((Evalue_donut * d[diagDist]) / Edistvalue_donut) * kr1[t_row] * kr2[t_col];
                e_h = ((Evalue_h * d[diagDist]) / Edistvalue_h) * kr1[t_row] * kr2[t_col];
                e_v = ((Evalue_v * d[diagDist]) / Edistvalue_v) * kr1[t_row] * kr2[t_col];

                float lognorm = (float) Math.log(Math.pow(2.0, .33));
                if (!Float.isNaN(e_bl) && !Float.isInfinite(e_bl)) {
                    if (e_bl <= 1) {
                        bvalue_bl = 0;
                    } else {
                        bvalue_bl = (float) Math.floor(Math.log(e_bl) / lognorm);
                    }
                }
                if (!Float.isNaN(e_donut) && !Float.isInfinite(e_donut)) {
                    if (e_donut <= 1) {
                        bvalue_donut = 0;
                    } else {
                        bvalue_donut = (float) Math.floor(Math.log(e_donut) / lognorm);
                    }
                }
                if (!Float.isNaN(e_h) && !Float.isInfinite(e_h)) {
                    if (e_h <= 1) {
                        bvalue_h = 0;
                    } else {
                        bvalue_h = (float) Math.floor(Math.log(e_h) / lognorm);
                    }
                }
                if (!Float.isNaN(e_v) && !Float.isInfinite(e_v)) {
                    if (e_v <= 1) {
                        bvalue_v = 0;
                    } else {
                        bvalue_v = (float) Math.floor(Math.log(e_v) / lognorm);
                    }
                }

                // todo why are bin values exceeding w1 in cpu version?
                // do they exceed in gpu version as well
                bvalue_bl = Math.min(bvalue_bl, HiCCUPS.w1 - 1);
                bvalue_donut = Math.min(bvalue_donut, HiCCUPS.w1 - 1);
                bvalue_h = Math.min(bvalue_h, HiCCUPS.w1 - 1);
                bvalue_v = Math.min(bvalue_v, HiCCUPS.w1 - 1);

                // Write the matrix to device memory;
                // each thread writes one element
                expectedBLDenseCPU[t_row][t_col] = e_bl;
                expectedDonutDenseCPU[t_row][t_col] = e_donut;
                expectedHDenseCPU[t_row][t_col] = e_h;
                expectedVDenseCPU[t_row][t_col] = e_v;
                o = Math.round(c[t_row][t_col] * kr1[t_row] * kr2[t_col]);
                observedDenseCPU[t_row][t_col] = o;
                binBLDenseCPU[t_row][t_col] = bvalue_bl;
                binDonutDenseCPU[t_row][t_col] = bvalue_donut;
                binHDenseCPU[t_row][t_col] = bvalue_h;
                binVDenseCPU[t_row][t_col] = bvalue_v;
                //System.out.println("thresholdBL "+thresholdBL.length+" thresholdDonut "+thresholdDonut.length);
                //System.out.println("a "+bvalue_bl+" b "+bvalue_donut);

                sbtrkt = Math.max(thresholdBL[(int) bvalue_bl], thresholdDonut[(int) bvalue_donut]);
                sbtrkt = Math.max(sbtrkt, thresholdH[(int) bvalue_h]);
                sbtrkt = Math.max(sbtrkt, thresholdV[(int) bvalue_v]);
                peakDenseCPU[t_row][t_col] = o - sbtrkt;
            }
        }

        // x2, y2 not inclusive here
        int finalWidthX = rowBounds[5] - rowBounds[4];
        int finalWidthY = columnBounds[5] - columnBounds[4];

        // x2, y2 not inclusive here
        int x1 = rowBounds[2];
        int y1 = columnBounds[2];
        int x2 = x1 + finalWidthX;
        int y2 = y1 + finalWidthY;

        //System.out.println("flat = "+flattenedSize+" n = "+matrixSize+" x1 = "+x1+" x2 = "+x2+" y1 = "+y1+" y2 ="+y2);

        observedDenseCPU = MatrixTools.extractLocalMatrixRegion(observedDenseCPU, x1, x2, y1, y2);
        peakDenseCPU = MatrixTools.extractLocalMatrixRegion(peakDenseCPU, x1, x2, y1, y2);
        binBLDenseCPU = MatrixTools.extractLocalMatrixRegion(binBLDenseCPU, x1, x2, y1, y2);
        binDonutDenseCPU = MatrixTools.extractLocalMatrixRegion(binDonutDenseCPU, x1, x2, y1, y2);
        binHDenseCPU = MatrixTools.extractLocalMatrixRegion(binHDenseCPU, x1, x2, y1, y2);
        binVDenseCPU = MatrixTools.extractLocalMatrixRegion(binVDenseCPU, x1, x2, y1, y2);
        expectedBLDenseCPU = MatrixTools.extractLocalMatrixRegion(expectedBLDenseCPU, x1, x2, y1, y2);
        expectedDonutDenseCPU = MatrixTools.extractLocalMatrixRegion(expectedDonutDenseCPU, x1, x2, y1, y2);
        expectedHDenseCPU = MatrixTools.extractLocalMatrixRegion(expectedHDenseCPU, x1, x2, y1, y2);
        expectedVDenseCPU = MatrixTools.extractLocalMatrixRegion(expectedVDenseCPU, x1, x2, y1, y2);

        return new GPUOutputContainer(observedDenseCPU, peakDenseCPU,
                binBLDenseCPU, binDonutDenseCPU, binHDenseCPU, binVDenseCPU,
                expectedBLDenseCPU, expectedDonutDenseCPU, expectedHDenseCPU, expectedVDenseCPU);
    }

}
