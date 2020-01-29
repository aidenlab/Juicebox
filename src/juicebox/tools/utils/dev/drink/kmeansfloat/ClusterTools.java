/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2020 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.dev.drink.kmeansfloat;

import juicebox.tools.utils.common.MatrixTools;

import java.io.File;

public class ClusterTools {

    public static void saveDistComparisonBetweenClusters(File directory, String filename, Cluster[] clusters, int[] ids) {
        int n = clusters.length;
        double[][] distances = new double[n][n];
        double[][] distancesNormalized = new double[n][n];
        for (int i = 0; i < n; i++) {
            Cluster expected = clusters[i];
            for (int j = 0; j < n; j++) {
                distances[i][j] = getDistance(clusters[j], expected);
                distancesNormalized[i][j] = distances[i][j] / clusters[j].getCenter().length;
            }
        }

        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".ids.npy").getAbsolutePath(), ids);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), distances);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + "_normed.npy").getAbsolutePath(), distancesNormalized);
    }

    public static double getDistance(Cluster observed, Cluster expected) {
        return Math.sqrt(getVectorMSEDifference(expected.getCenter(), observed.getCenter()));
    }

    public static double getDistance(float[] expectedArray, float[] obsArray) {
        return Math.sqrt(getVectorMSEDifference(expectedArray, obsArray));
    }

    public static double getVectorMSEDifference(float[] expectedArray, float[] obsArray) {

        double val = 0;

        for (int k = 0; k < obsArray.length; k++) {
            double v = expectedArray[k] - obsArray[k];
            val += v * v;
        }

        return val;
    }

    public static float[] normalize(float[] vector, Integer total) {
        float[] newVector = new float[vector.length];
        for (int k = 0; k < vector.length; k++) {
            newVector[k] = vector[k] / total;
        }
        return newVector;
    }
}

