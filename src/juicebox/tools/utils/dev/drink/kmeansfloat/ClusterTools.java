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
import org.apache.commons.math.stat.inference.ChiSquareTest;
import org.apache.commons.math.stat.inference.ChiSquareTestImpl;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ClusterTools {

    public static Cluster[] getSortedClusters(Cluster[] unsortedClusters) {
        List<Cluster> tempClusters = new ArrayList<>();
        for (Cluster item : unsortedClusters) {
            tempClusters.add(item);
        }

        Collections.sort(tempClusters, new Comparator<Cluster>() {
            @Override
            public int compare(Cluster o1, Cluster o2) {
                Integer size1 = o1.getMemberIndexes().length;
                Integer size2 = o2.getMemberIndexes().length;

                int comparison = -size1.compareTo(size2);
                if (comparison == 0) {
                    Integer indx1 = o1.getMemberIndexes()[0];
                    Integer indx2 = o2.getMemberIndexes()[0];
                    comparison = indx1.compareTo(indx2);
                }

                return comparison;
            }
        });

        Cluster[] sortedClusters = new Cluster[unsortedClusters.length];
        for (int i = 0; i < unsortedClusters.length; i++) {
            sortedClusters[i] = tempClusters.get(i);
        }


        return sortedClusters;
    }



    public static void performStatisticalAnalysisBetweenClusters(File directory, String description, Cluster[] clusters, int[] ids) {

        File statsFolder = new File(directory, description + "_cluster_stats");
        statsFolder.mkdir();

        MatrixTools.saveMatrixTextNumpy(new File(statsFolder, description + "cluster.ids.npy").getAbsolutePath(), ids);

        saveClusterSizes(statsFolder, "sizes", clusters);
        saveDistComparisonBetweenClusters(statsFolder, "distances", clusters);
        saveComparisonBetweenClusters(statsFolder, "num.differences", clusters);
        saveChiSquarePvalComparisonBetweenClusters(statsFolder, "chi2.pval", clusters);
        saveChiSquareValComparisonBetweenClusters(statsFolder, "chi2", clusters);

    }

    private static void saveDistComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
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

        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), distances);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + "_normed.npy").getAbsolutePath(), distancesNormalized);
    }

    private static void saveChiSquarePvalComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
        int n = clusters.length;
        double[][] pvalues = new double[n][n];
        for (int i = 0; i < n; i++) {
            Cluster expected = clusters[i];
            for (int j = 0; j < n; j++) {
                pvalues[i][j] = getPvalueChiSquared(clusters[j], expected);
            }
        }
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), pvalues);
    }


    private static void saveComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
        int n = clusters.length;
        double[][] numDiffEntries = new double[n][n];
        double[][] numDiffEntriesNormalized = new double[n][n];
        for (int i = 0; i < n; i++) {
            Cluster expected = clusters[i];
            for (int j = 0; j < n; j++) {
                numDiffEntries[i][j] = getNumDiffEntries(clusters[j], expected);
                numDiffEntriesNormalized[i][j] = numDiffEntries[i][j] / clusters[j].getCenter().length;
            }
        }

        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), numDiffEntries);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + "_normed.npy").getAbsolutePath(), numDiffEntriesNormalized);
    }

    private static void saveChiSquareValComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
        int n = clusters.length;
        double[][] chi2Val = new double[n][n];
        for (int i = 0; i < n; i++) {
            Cluster expected = clusters[i];
            for (int j = 0; j < n; j++) {
                chi2Val[i][j] = getValueChiSquared(clusters[j], expected);
            }
        }
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), chi2Val);

    }

    private static void saveClusterSizes(File directory, String filename, Cluster[] clusters) {
        int n = clusters.length;

        int[][] sizeClusters = new int[1][n];
        for (int i = 0; i < n; i++) {
            sizeClusters[0][i] = clusters[i].getMemberIndexes().length;
        }

        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), sizeClusters);
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

    private static double getValueChiSquared(Cluster observed, Cluster expected) {
        ChiSquareTest test = new ChiSquareTestImpl();
        try {
            return test.chiSquare(toHalfDoubleArray(expected), toHalfLongArray(observed));
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static double getPvalueChiSquared(Cluster observed, Cluster expected) {
        ChiSquareTest test = new ChiSquareTestImpl();
        try {
            return test.chiSquareTest(toHalfDoubleArray(expected), toHalfLongArray(observed));
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static int getNumDiffEntries(Cluster observed, Cluster expected) {
        float[] expectedArray = expected.getCenter();
        float[] obsArray = observed.getCenter();
        int count = 0;

        for (int k = 0; k < obsArray.length; k++) {
            double v = expectedArray[k] - obsArray[k];
            v = (v * v) / Math.abs(expectedArray[k]);
            if (v < .05) {
                count++;
            }
        }

        return count;
    }

    private static long[] toHalfLongArray(Cluster cluster) {
        float[] clusterData = cluster.getCenter();
        int n = (clusterData.length + 1) / 2; // trim derivative
        long[] result = new long[n];
        for (int i = 0; i < n; i++) {
            result[i] = Math.round(clusterData[i]);
        }
        return result;
    }

    private static double[] toHalfDoubleArray(Cluster cluster) {
        float[] clusterData = cluster.getCenter();
        int n = (clusterData.length + 1) / 2; // trim derivative
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = clusterData[i];
        }
        return result;
    }

    public static Cluster[] clone(Cluster[] input) {
        Cluster[] clone = new Cluster[input.length];
        for (int k = 0; k < input.length; k++) {
            clone[k] = input[k].getClone();
        }

        return clone;
    }
}

