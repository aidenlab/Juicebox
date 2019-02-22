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

package juicebox.tools.utils.common;

import juicebox.HiCGlobals;
import org.apache.commons.math.distribution.PoissonDistributionImpl;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

/**
 * Helper methods for handling arrays
 */
public class ArrayTools {

    /**
     * @return deep copy of 2D float array
     */
    public static float[][] deepCopy(float[][] original) {
        float[][] copy = new float[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            System.arraycopy(original[i], 0, copy[i], 0, original[i].length);
        }
        return copy;
    }

    /**
     * @return mean of given array
     */
    public static double mean(double[] doubles) {
        double sum = 0;
        for (double d : doubles) {
            sum += d;
        }
        return sum / doubles.length;
    }

    /**
     * poisson.pdf(k) = exp(-mu) * mu**k / k!
     *
     * @param index
     * @param width
     * @return
     */
    public static double[] generatePoissonPMF(int index, int width) {
        double mu = Math.pow(2.0, (index + 1.0) / 3.0);
        double[] poissonPMF = new double[width];

        PoissonDistributionImpl poissonDistribution = new PoissonDistributionImpl(mu);

        // use dynamic programming to grow poisson PMF
        for (int k = 0; k < width; k++) {
            poissonPMF[k] = poissonDistribution.probability(k); // the total is for scaling
        }

        return poissonPMF;
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

    private static long[] makeReverseCumulativeArray(long[] inputArray) {
        long[] outputArray = new long[inputArray.length];
        int total = 0;
        for (int i = inputArray.length - 1; i > -1; i--) {
            total += inputArray[i];
            outputArray[i] = total;
        }
        return outputArray;
    }

    public static long[][] makeReverse2DCumulativeArray(long[][] data) {
        long[][] rcsData = new long[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            rcsData[i] = ArrayTools.makeReverseCumulativeArray(data[i]);
            if (HiCGlobals.printVerboseComments && data[i][0] < 0) {
                System.out.println("poss source2: i " + i + "  " + data[i][0]);
            }
        }
        return rcsData;
    }

    public static float[] newValueInitializedFloatArray(int n, float val) {
        float[] array = new float[n];
        Arrays.fill(array, val);
        return array;
    }

    public static float[] scalarMultiplyArray(long scaleFactor, float[] array) {
        float[] scaledArray = newValueInitializedFloatArray(array.length, scaleFactor);
        for (int i = 0; i < array.length; i++) {
            scaledArray[i] *= array[i];
        }
        return scaledArray;
    }

    /**
     * Assumes array passed in is <= length, otherwise indexOutOfBounds error will be thrown
     *
     * @param original
     * @param length
     * @param val
     * @return
     */
    public static float[] padEndOfArray(float[] original, int length, float val) {
        float[] paddedArray = new float[length];
        System.arraycopy(original, 0, paddedArray, 0, original.length);
        for (int i = original.length; i < length; i++) {
            paddedArray[i] = val;
        }
        return paddedArray;
    }

    public static int[] extractIntegers(List<String> stringList) {
        int[] array = new int[stringList.size()];

        int index = 0;
        for (String val : stringList) {
            array[index] = Integer.parseInt(val);
            index++;
        }
        return array;
    }

    public static float[] extractFloats(List<String> stringList) {
        float[] array = new float[stringList.size()];

        int index = 0;
        for (String val : stringList) {
            array[index] = Float.parseFloat(val);
            index++;
        }
        return array;
    }

    public static int[] preInitializeIntArray(int val, int n) {
        int[] array = new int[n];
        for (int i = 0; i < n; i++) {
            array[i] = val;
        }
        return array;
    }

    public static float[] preInitializeFloatArray(float val, int n) {
        float[] array = new float[n];
        for (int i = 0; i < n; i++) {
            array[i] = val;
        }
        return array;
    }

    public static float[] inverseArrayValues(float[] array) {
        float[] inverses = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            inverses[i] = 1f / array[i];
        }
        return inverses;
    }

    public static double[] preInitializeDoubleArray(double val, int n) {
        double[] array = new double[n];
        for (int i = 0; i < n; i++) {
            array[i] = val;
        }
        return array;
    }

    public static double[] extractDoubles(List<String> stringList) {
        double[] array = new double[stringList.size()];

        int index = 0;
        for (String val : stringList) {
            array[index] = Double.parseDouble(val);
            index++;
        }
        return array;
    }

    public static double[] inverseArrayValues(double[] array) {
        double[] inverses = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            inverses[i] = 1. / array[i];
        }
        return inverses;
    }

    /**
     * Print out a given vector in wig style format to the specified writer
     * @param vector
     * @param pw
     * @param name
     * @param res
     */
    public static void exportChr1DArrayToWigFormat(double[] vector, PrintWriter pw, String name, int res) {
        pw.println("fixedStep chrom=chr" + name.replace("chr", "") + " start=1 step=" + res + " span=" + res);
        for(double val : vector){
            pw.println(val);
        }
    }

    /**
     * confirm before calling that both vectors are the same size
     *
     * @param vector1
     * @param vector2
     * @return
     */
    public static double euclideanDistance(double[] vector1, double[] vector2) {
        double distance = 0;
        for (int i = 0; i < vector1.length; i++) {
            double diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return Math.sqrt(distance);
    }
}
