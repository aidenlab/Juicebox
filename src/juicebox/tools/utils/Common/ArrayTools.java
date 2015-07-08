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

package juicebox.tools.utils.common;

import org.apache.commons.math.distribution.PoissonDistributionImpl;

import java.util.Arrays;

/**
 * Created by muhammadsaadshamim on 5/12/15.
 */
public class ArrayTools {

    public static float[][] deepCopy(float[][] original) {
        float[][] copy = new float[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            System.arraycopy(original[i], 0, copy[i], 0, original[i].length);
        }
        return copy;
    }

    public static int arrayMax(int[] array) {
        int max = array[0];
        for (int val : array)
            if (val > max)
                max = val;
        return max;
    }

    public static double arrayMax(double[] array) {
        double max = array[0];
        for (double val : array)
            if (val > max)
                max = val;
        return max;
    }

    public static int arrayMin(int[] array) {
        int min = array[0];
        for (int val : array)
            if (val < min)
                min = val;
        return min;
    }

    /**
     * poisson.pdf(k) = exp(-mu) * mu**k / k!
     *
     * @param index
     * @param width
     * @return
     */
    public static double[] generatePoissonPMF(int index, int width) {
        double mu = Math.pow(2.0,(index + 1.0) / 3.0);
        double[] poissonPMF = new double[width];

        PoissonDistributionImpl poissonDistribution = new PoissonDistributionImpl(mu);

        // use dynamic programming to grow poisson PMF
        for (int k = 0; k < width; k++){
            poissonPMF[k] = poissonDistribution.probability(k); // the total is for scaling
        }

        return poissonPMF;
    }

    public static float[] doubleArrayToFloatArray(double[] doubleArray){
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0 ; i < doubleArray.length; i++)
        {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }

    public static float[] makeReverseCumulativeArray(float[] inputArray) {
        float[] outputArray = new float[inputArray.length];
        float total = 0f;
        for (int i = inputArray.length - 1; i > -1; i--) {
            total += inputArray[i];
            outputArray[i] = total;
        }
        return outputArray;
    }

    public static int[] makeReverseCumulativeArray(int[] inputArray) {
        int[] outputArray = new int[inputArray.length];
        int total = 0;
        for (int i = inputArray.length - 1; i > -1; i--) {
            total += inputArray[i];
            outputArray[i] = total;
        }
        return outputArray;
    }

    /**
     * inclusive both ends
     * @param array
     * @param start index
     * @param end index (inclusive)
     */
    public static double[] extractArray(double[] array, int start, int end) {
        double[] array2 = new double[end - start + 1];
        System.arraycopy(array, start, array2, 0, array2.length);
        return array2;
    }

    public static double[] flipArray(double[] array) {
        int n = array.length;
        double[] array2 = new double[n];
        for(int i = 0; i < n; i++){
            array2[i] = array[n-i];
        }
        return array2;
    }

    public static float[] newValueInitializedFloatArray(int n, float val) {
        float[] array = new float[n];
        Arrays.fill(array, val);
        return array;
    }

    public static int[][] makeReverse2DCumulativeArray(int[][] hist) {
        int[][] rcsHist = new int[hist.length][hist[0].length];
        for (int i = 0; i < hist.length; i++) {
            rcsHist[i] = ArrayTools.makeReverseCumulativeArray(hist[i]);
        }
        return rcsHist;
    }

    public static float[] scalarMultiplyArray(int scaleFactor, float[] array) {
        float[] scaledArray = newValueInitializedFloatArray(array.length, scaleFactor);
        for (int i=0; i<array.length; i++) {
            scaledArray[i] *= array[i];
        }
        return scaledArray;
    }

    /**
     * Assumes array passed in is <= length, otherwise indexOutOfBounds error will be thrown
     * @param original
     * @param length
     * @param val
     * @return
     */
    public static float[] padEndOfArray(float[] original, int length, float val) {
        float[] paddedArray = new float[length];
        for(int i = 0; i < original.length; i++){
            paddedArray[i] = original[i];
        }
        for(int i = original.length; i < length; i++){
            paddedArray[i] = val;
        }
        return paddedArray;
    }
}
