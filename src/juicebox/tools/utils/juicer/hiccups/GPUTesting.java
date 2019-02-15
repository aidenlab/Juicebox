/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2019 Broad Institute, Aiden Lab
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
import jcuda.driver.JCudaDriver;
import jcuda.utils.KernelLauncher;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

/**
 * Created by muhammadsaadshamim on 5/7/15.
 */
class GPUTesting {

    public static void test() {
        JCudaDriver.setExceptionsEnabled(true);

        String sourceCode = "extern \"C\"" + "\n" +
                "__global__ void add(float *result, float *a, float *b)" +
                "\n" +
                "{" + "\n" +
                "    int i = threadIdx.x;" + "\n" +
                "    result[i] = a[i] + b[i];" + "\n" +
                "}";

        // Prepare the kernel
        System.out.println("Preparing the KernelLauncher...");
        KernelLauncher kernelLauncher =
                KernelLauncher.compile(sourceCode, "add");

        // Create the input data
        System.out.println("Creating input data...");
        int size = 10;
        float[] result = new float[size];
        float[] a = new float[size];
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = i;
        }

        // Allocate the device memory and copy the input
        // data to the device
        System.out.println("Initializing device memory...");

        CUdeviceptr dResult = GPUHelper.allocateOutput(size, Sizeof.FLOAT);
        CUdeviceptr dA = GPUHelper.allocateInput(a);
        CUdeviceptr dB = GPUHelper.allocateInput(b);

        System.out.println("Calling the kernel...");
        kernelLauncher.setBlockSize(size, 1, 1);
        kernelLauncher.call(dResult, dA, dB);

        // Copy the result from the device to the host
        System.out.println("Obtaining results...");

        cuMemcpyDtoH(Pointer.to(result), dResult, size * Sizeof.FLOAT);

        System.out.println("Result: " + Arrays.toString(result));

        // Clean up
        cuMemFree(dA);
        cuMemFree(dB);
        cuMemFree(dResult);
    }
}
