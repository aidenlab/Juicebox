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

package juicebox.tools.utils.dev.drink;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.tools.utils.dev.drink.kmeansfloat.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeansfloat.KMeansListener;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.MathException;
import org.apache.commons.math.stat.inference.ChiSquareTest;
import org.apache.commons.math.stat.inference.ChiSquareTestImpl;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class OddAndEvenClusterer {

    private final Dataset ds;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;
    private final NormalizationType norm;
    private final int numClusters;
    private final int maxIters;
    private final GenomeWideList<SubcompartmentInterval> origIntraSubcompartments;
    private final AtomicInteger numCompleted = new AtomicInteger(0);

    public OddAndEvenClusterer(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                               int numClusters, int maxIters, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments) {
        this.ds = ds;
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
        this.norm = norm;
        this.numClusters = numClusters;
        this.maxIters = maxIters;
        this.origIntraSubcompartments = origIntraSubcompartments;
    }

    public GenomeWideList<SubcompartmentInterval> extractFinalGWSubcompartments(File outputDirectory, Random generator,
                                                                                CompositeInterchromDensityMatrix.InterMapType mapType) {

        final CompositeInterchromDensityMatrix interMatrix = new CompositeInterchromDensityMatrix(
                chromosomeHandler, ds, norm, resolution, origIntraSubcompartments, mapType);

        //File outputFile = new File(outputDirectory, isOddsVsEvenType + "inter_Odd_vs_Even_matrix_data.txt");
        //MatrixTools.exportData(interMatrix.getCleanedData(), outputFile);

        Map<Integer, Integer> subcompartment1IDsToSize = new HashMap<>();
        Map<Integer, Integer> subcompartment2IDsToSize = new HashMap<>();

        GenomeWideList<SubcompartmentInterval> finalCompartments = new GenomeWideList<>(chromosomeHandler);
        launchKmeansInterMatrix(outputDirectory, mapType + "_" + false, interMatrix, finalCompartments, false, generator.nextLong(), subcompartment1IDsToSize);
        launchKmeansInterMatrix(outputDirectory, mapType + "_" + true, interMatrix, finalCompartments, true, generator.nextLong(), subcompartment2IDsToSize);

        while (numCompleted.get() < 1) {
            System.out.println("So far portion completed is " + numCompleted.get() + "/2");
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        interMatrix.stitchTogetherResults(finalCompartments, ds, outputDirectory, "" + mapType);

        return finalCompartments;
    }


    private void launchKmeansInterMatrix(File directory, String description, final CompositeInterchromDensityMatrix matrix,
                                         final GenomeWideList<SubcompartmentInterval> interSubcompartments,
                                         final boolean isTransposed, final long seed, Map<Integer, Integer> subcompartmentIDsToSize) {

        if (matrix.getLength() > 0 && matrix.getWidth() > 0) {
            float[][] cleanDataWithDeriv;
            if (isTransposed) {
                cleanDataWithDeriv = matrix.getCleanedTransposedData();
            } else {
                cleanDataWithDeriv = matrix.getCleanedData();
            }
            //cleanDataWithDeriv = MatrixTools.getMainAppendedDerivativeDownColumn(cleanDataWithDeriv, 10);
            cleanDataWithDeriv = MatrixTools.getNormalizedThresholdedAndAppendedDerivativeDownColumn(cleanDataWithDeriv, 2, 10, 5);
            //cleanDataWithDeriv = MatrixTools.getRelevantDerivativeScaledPositive(cleanDataWithDeriv);
            MatrixTools.saveMatrixTextNumpy(new File(directory, description + "." + isTransposed + "clusterdata.npy").getAbsolutePath(), cleanDataWithDeriv);

            ConcurrentKMeans kMeans = new ConcurrentKMeans(cleanDataWithDeriv, numClusters, maxIters, seed);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    synchronized (numCompleted) {
                        numCompleted.incrementAndGet();
                        //saveChiSquareComparisonBetweenClusters(directory, description + "_pval", clusters);
                        //saveChiSquareValComparisonBetweenClusters(directory, description + "_chi", clusters);
                        //saveDistComparisonBetweenClusters(directory, description + "_dist", clusters);
                        matrix.processGWKmeansResult(clusters, interSubcompartments, isTransposed, subcompartmentIDsToSize);
                    }
                }

                @Override
                public void kmeansError(Throwable throwable) {
                    throwable.printStackTrace();
                    System.err.println("gw drink - err - " + throwable.getLocalizedMessage());
                    System.exit(98);
                }
            };
            kMeans.addKMeansListener(kMeansListener);
            kMeans.run();
        }
    }

    private void saveChiSquareComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
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


    private void saveComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
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
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + "num_diff_entries.npy").getAbsolutePath(), numDiffEntries);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + "num_diff_entries_normed.npy").getAbsolutePath(), numDiffEntriesNormalized);
    }

    private void saveChiSquareValComparisonBetweenClusters(File directory, String filename, Cluster[] clusters) {
        int n = clusters.length;
        double[][] chi2Val = new double[n][n];
        for (int i = 0; i < n; i++) {
            Cluster expected = clusters[i];
            for (int j = 0; j < n; j++) {
                chi2Val[i][j] = getValueChiSquared(clusters[j], expected);
            }
        }
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".npy").getAbsolutePath(), chi2Val);


        int[][] sizeClusters = new int[1][n];
        for (int i = 0; i < n; i++) {
            sizeClusters[0][i] = clusters[i].getMemberIndexes().length;
        }

        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".sizes.npy").getAbsolutePath(), sizeClusters);
    }

    private double getValueChiSquared(Cluster observed, Cluster expected) {
        ChiSquareTest test = new ChiSquareTestImpl();
        return test.chiSquare(toHalfDoubleArray(expected), toHalfLongArray(observed));
    }



    private int getNumDiffEntries(Cluster observed, Cluster expected) {
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

    private double getPvalueChiSquared(Cluster observed, Cluster expected) {
        ChiSquareTest test = new ChiSquareTestImpl();
        try {
            return test.chiSquareTest(toHalfDoubleArray(expected), toHalfLongArray(observed));
        } catch (MathException e) {
            e.printStackTrace();
        }
        return Double.NaN;
    }

    private long[] toHalfLongArray(Cluster cluster) {
        float[] clusterData = cluster.getCenter();
        int n = (clusterData.length + 1) / 2; // trim derivative
        long[] result = new long[n];
        for (int i = 0; i < n; i++) {
            result[i] = Math.round(clusterData[i]);
        }
        return result;
    }

    private double[] toHalfDoubleArray(Cluster cluster) {
        float[] clusterData = cluster.getCenter();
        int n = (clusterData.length + 1) / 2; // trim derivative
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = clusterData[i];
        }
        return result;
    }
}
