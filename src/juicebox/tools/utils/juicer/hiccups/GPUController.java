/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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
import juicebox.tools.utils.common.ArrayTools;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;

import java.io.IOException;
import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class GPUController {

    private static final int blockSize = 16;  //number of threads in block

    private final KernelLauncher kernelLauncher;

    public GPUController(int window, int matrixSize, int peakWidth, int divisor) {
        String kernelCode = GPUKernel.kernelCode(window, matrixSize, peakWidth, divisor);
        kernelLauncher =
                KernelLauncher.compile(kernelCode, GPUKernel.kernelName);

        //threads per block = block_size*block_size
        kernelLauncher.setBlockSize(blockSize, blockSize, 1);

        // for grid of blocks
        int gridSize = (int) Math.ceil(matrixSize * 1.0 / blockSize);
        kernelLauncher.setGridSize(gridSize, gridSize);
    }

    public GPUOutputContainer process(MatrixZoomData zd, double[] normalizationVector, double[] expectedVector,
                                      int[] rowBounds, int[] columnBounds, int matrixSize,
                                      float[] thresholdBL, float[] thresholdDonut, float[] thresholdH, float[] thresholdV,
                                      NormalizationType normalizationType)
            throws NegativeArraySizeException, IOException {

        RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd, rowBounds[0], rowBounds[1],
                columnBounds[0], columnBounds[1], matrixSize, matrixSize, normalizationType);


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

        //long gpu_time1 = System.currentTimeMillis();

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

        GPUHelper.freeUpMemory(new CUdeviceptr[]{observedKRGPU, expectedDistanceVectorGPU,
                kr1GPU, kr2GPU, thresholdBLGPU, thresholdDonutGPU, thresholdHGPU,
                thresholdVGPU, boundRowIndexGPU, boundColumnIndexGPU,
                expectedBLGPU, expectedDonutGPU, expectedHGPU, expectedVGPU,
                binBLGPU, binDonutGPU, binHGPU, binVGPU,
                observedGPU, peakGPU});

        return new GPUOutputContainer(observedDenseCPU, peakDenseCPU,
                binBLDenseCPU, binDonutDenseCPU, binHDenseCPU, binVDenseCPU,
                expectedBLDenseCPU, expectedDonutDenseCPU, expectedHDenseCPU, expectedVDenseCPU);
    }
}
