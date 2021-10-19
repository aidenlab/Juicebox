/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2021 Broad Institute, Aiden Lab, Rice University, Baylor College of Medicine
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
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools.utils.common;

import java.util.List;

public class QuickMedian {

    public static float fastMedian(List<Number> list) {
        int size = list.size();
        if (size < 1) {
            return Float.NaN;
        } else if (size == 1) {
            return list.get(0).floatValue();
        }

        float[] arr = new float[size];
        for (int k = 0; k < arr.length; k++) {
            arr[k] = list.get(k).floatValue();
        }

        return fastMedian(arr);
    }

    public static float fastMedian(float[] arr) {
        int len = arr.length;
        if (len % 2 == 1) {
            return kSelection(arr, 0, len - 1, len / 2);
        } else {
            float a = kSelection(arr, 0, len - 1, len / 2);
            float b = kSelection(arr, 0, len - 1, len / 2 - 1);
            return (a + b) / 2;
        }
    }

    public static float kSelection(float[] arr, int low, int high, int k) {
        int localLow = low;
        int localHigh = high;

        int partitionSortingValue = partition(arr, localLow, localHigh);
        while (partitionSortingValue != k) {
            if (partitionSortingValue < k) {
                localLow = partitionSortingValue + 1;
            } else {
                localHigh = partitionSortingValue - 1;
            }
            partitionSortingValue = partition(arr, localLow, localHigh);
        }
        return arr[partitionSortingValue];
    }

    static int partition(float[] arr, int low, int high) {
        float pivot = arr[high];
        int z = (low - 1);
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                z++;
                float temp = arr[z];
                arr[z] = arr[j];
                arr[j] = temp;
            }
        }
        float temp = arr[z + 1];
        arr[z + 1] = arr[high];
        arr[high] = temp;
        return z + 1;
    }
}
