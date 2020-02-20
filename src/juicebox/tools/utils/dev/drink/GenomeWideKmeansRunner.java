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

import com.google.common.util.concurrent.AtomicDouble;
import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.tools.utils.dev.drink.kmeansfloat.ClusterTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeansfloat.KMeansListener;
import org.broad.igv.util.Pair;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class GenomeWideKmeansRunner {

    private static Cluster[] recentClusters;
    private static int[] recentIDs;

    private final CompositeGenomeWideDensityMatrix matrix;
    private final ChromosomeHandler chromosomeHandler;
    private final Random generator;
    private final AtomicInteger numActualClusters = new AtomicInteger(0);
    private final AtomicDouble meanSquaredErrorForRun = new AtomicDouble(0);
    private final int maxIters = 20000;


    private GenomeWideList<SubcompartmentInterval> finalCompartments;
    private int numClusters = 0;

    public GenomeWideKmeansRunner(ChromosomeHandler chromosomeHandler, CompositeGenomeWideDensityMatrix interMatrix, Random generator) {
        matrix = interMatrix;
        this.generator = generator;
        this.chromosomeHandler = chromosomeHandler;
    }

    public void prepareForNewRun(int numClusters) {
        recentClusters = null;
        recentIDs = null;
        this.numClusters = numClusters;
        numActualClusters.set(0);
        meanSquaredErrorForRun.set(0);
        finalCompartments = new GenomeWideList<>(chromosomeHandler);
    }

    public void launchKmeansGWMatrix() {

        if (matrix.getLength() > 0 && matrix.getWidth() > 0) {

            ConcurrentKMeans kMeans = new ConcurrentKMeans(matrix.getCleanedData(), numClusters, maxIters, generator.nextLong());

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println(s);
                    }
                }

                @Override
                public void kmeansComplete(Cluster[] preSortedClusters, long l) {

                    Cluster[] clusters = ClusterTools.getSortedClusters(preSortedClusters);

                    System.out.print(".");
                    Pair<Double, int[]> mseAndIds = matrix.processGWKmeansResult(clusters, finalCompartments);
                    recentClusters = ClusterTools.clone(clusters);
                    recentIDs = mseAndIds.getSecond();
                    numActualClusters.set(clusters.length);
                    meanSquaredErrorForRun.set(mseAndIds.getFirst());
                }

                @Override
                public void kmeansError(Throwable throwable) {
                    throwable.printStackTrace();
                    System.err.println("gw full drink - err - " + throwable.getLocalizedMessage());
                    System.exit(98);
                }
            };
            kMeans.addKMeansListener(kMeansListener);
            kMeans.run();
        }

        waitUntilDone();
    }

    private void waitUntilDone() {
        while (numActualClusters.get() < 1 && meanSquaredErrorForRun.get() == 0.0) {
            System.out.print(".");
            try {
                TimeUnit.SECONDS.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public int getNumActualClusters() {
        return numActualClusters.get();
    }

    public double getMeanSquaredError() {
        return meanSquaredErrorForRun.get();
    }

    public Cluster[] getRecentClustersClone() {
        return ClusterTools.clone(recentClusters);
    }

    public int[] getRecentIDsClone() {
        return Arrays.copyOf(recentIDs, recentIDs.length);
    }

    public GenomeWideList<SubcompartmentInterval> getFinalCompartments() {
        return finalCompartments.deepClone();
    }
}
