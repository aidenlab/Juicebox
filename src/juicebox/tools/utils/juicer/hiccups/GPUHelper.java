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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import juicebox.tools.utils.common.MatrixTools;

import static jcuda.driver.JCudaDriver.*;

/**
 * Created by muhammadsaadshamim on 5/8/15.
 */
class GPUHelper {

    public static CUdeviceptr allocateOutput(int size, int typeSize) {
        CUdeviceptr dOutput = new CUdeviceptr();
        cuMemAlloc(dOutput, size * typeSize);
        return dOutput;
    }

    public static CUdeviceptr allocateInput(float[] input) {
        int typeSize = Sizeof.FLOAT;
        Pointer ptr = Pointer.to(input);
        int size = input.length;
        CUdeviceptr dInput = new CUdeviceptr();
        cuMemAlloc(dInput, size * Sizeof.FLOAT);
        cuMemcpyHtoD(dInput, ptr, size * typeSize);
        return dInput;
    }

    public static void freeUpMemory(CUdeviceptr[] pointers) {
        for (CUdeviceptr pointer : pointers) {
            cuMemFree(pointer);
        }
    }

    public static float[][] GPUArraytoCPUMatrix(float[] result, int n, int x1, int x2, int y1, int y2) {
        return MatrixTools.extractLocalMatrixRegion(
                MatrixTools.reshapeFlatMatrix(result, n)
                , x1, x2, y1, y2);
    }
}

