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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.utils.KernelLauncher;
import juicebox.data.MatrixZoomData;
import juicebox.data.NormalizationVector;
import juicebox.tools.utils.Common.ArrayTools;
import juicebox.tools.utils.Common.HiCFileTools;
import juicebox.tools.utils.Common.MatrixTools;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class GPUController {

    private String kernelCode;
    private int blockSize = 16;  //number of threads in block
    private int[] block = new int[]{blockSize, blockSize, 1}; //threads per block = block_size*block_size
    //private int[] grid; // for grid of blocks

    private KernelLauncher kernelLauncher;

    public GPUController(int window, int matrixSize, int peakWidth, int divisor) {
        kernelCode = GPUKernel.kernelCode(window, matrixSize, peakWidth, divisor);
        kernelLauncher =
                KernelLauncher.compile(kernelCode, GPUKernel.kernelName);

        kernelLauncher.setBlockSize(blockSize, blockSize, 1);
        int gridSize = (matrixSize / blockSize) + 1;
        kernelLauncher.setGridSize(gridSize, gridSize);

        //grid = new int[]{, (matrixSize / blockSize) + 1};
    }

    public GPUOutputContainer process(MatrixZoomData zd, NormalizationVector krNormalizationVector, double[] expectedKRVector,
                                      int[] rowBounds, int[] columnBounds, int matrixSize,
                                      float[] thresholdBL, float[] thresholdDonut, float[] thresholdH,float[] thresholdV,
                                      float[] boundRowIndex, float[] boundColumnIndex, NormalizationType normalizationType) {

        RealMatrix localizedRegionData = HiCFileTools.extractLocalBoundedRegion(zd, rowBounds[0], rowBounds[1],
                columnBounds[0], columnBounds[1], matrixSize, matrixSize, normalizationType);
        
        float[] observedVals = ArrayTools.doubleArrayToFloatArray(
                MatrixTools.flattenedRowMajorOrderMatrix(localizedRegionData));

        // slice KR vector to localized region

        float[] distanceExpectedKRVector = ArrayTools.doubleArrayToFloatArray(expectedKRVector);



        float[] kr1CPU = ArrayTools.doubleArrayToFloatArray(
                MatrixTools.sliceFromVector(krNormalizationVector, rowBounds[0], rowBounds[1]));
        float[] kr2CPU = ArrayTools.doubleArrayToFloatArray(
                MatrixTools.sliceFromVector(krNormalizationVector, columnBounds[0], columnBounds[1]));
        boundRowIndex[0] = rowBounds[0];
        boundColumnIndex[0] = columnBounds[0];

        long gpu_time1 = System.currentTimeMillis();



        // transfer host (CPU) memory to device (GPU) memory
        CUdeviceptr observedKRGPU = GPUHelper.allocateInput(observedVals);
        CUdeviceptr expectedKRVectorGPU = GPUHelper.allocateInput(distanceExpectedKRVector);
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
                expectedBLGPU,
                expectedDonutGPU,
                expectedHGPU,
                expectedVGPU,
                observedGPU,
                binBLGPU,
                binDonutGPU,
                binHGPU,
                binVGPU,
                peakGPU,
                // thresholds
                thresholdBLGPU,
                thresholdDonutGPU,
                thresholdHGPU,
                thresholdVGPU,
                // distance expected
                expectedKRVectorGPU,
                // kr
                kr1GPU,
                kr2GPU,
                // bounds
                boundRowIndexGPU,
                boundColumnIndexGPU);


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





        long gpu_time2 = System.currentTimeMillis();

        //System.out.println("GPU Time: " + (gpu_time2-gpu_time1));

        // x2, y2 not inclusive here
        int x1 = rowBounds[2], x2 = matrixSize - rowBounds[3];
        int y1 = columnBounds[2], y2 = matrixSize - columnBounds[3];

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


        //MatrixTools.extractLocalMatrixRegion(, x1, x2, y1, y2);
        //float[][] temp12 = MatrixTools.reshapeFlatMatrix(observedResult, matrixSize);


        /** testing **/
        //dumpToFile(expectedBLResult, "expectedBLResult+"+)
        /*System.out.println("diag observedResult = ");
        int tempI = (int)Math.sqrt(flattenedSize);
        for(int i = 0; i < flattenedSize; i+= 1+ tempI){
            System.out.print(observedResult[i]+" ");
        }
        System.out.println("");

        System.out.println("diag2 observedDenseCPU = ");
        tempI = Math.min(observedDenseCPU.length,observedDenseCPU[0].length);
        for(int i = 0; i < tempI; i++){
            System.out.print(observedDenseCPU[i][i]+" ");
        }
        System.out.println("");
        */



        // free up GPU inputs and outputs
        GPUHelper.freeUpMemory(new CUdeviceptr[]{observedKRGPU, expectedKRVectorGPU,
                kr1GPU, kr2GPU, thresholdBLGPU, thresholdDonutGPU, thresholdHGPU,
                thresholdVGPU, boundRowIndexGPU, boundColumnIndexGPU});
        GPUHelper.freeUpMemory(new CUdeviceptr[]{expectedBLGPU, expectedDonutGPU,
                expectedHGPU, expectedVGPU, binBLGPU, binDonutGPU, binHGPU, binVGPU,
                observedGPU, peakGPU});

        return new GPUOutputContainer(observedDenseCPU, peakDenseCPU,
                binBLDenseCPU, binDonutDenseCPU, binHDenseCPU, binVDenseCPU,
                expectedBLDenseCPU, expectedDonutDenseCPU, expectedHDenseCPU, expectedVDenseCPU);
    }
}
