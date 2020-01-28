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

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.tools.utils.dev.drink.kmeansfloat.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeansfloat.KMeansListener;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class FullGenomeOEWithinClusters {
    private final Dataset ds;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;
    private final NormalizationType norm;
    private final int maxIters;
    private final GenomeWideList<SubcompartmentInterval> origIntraSubcompartments;
    private final AtomicInteger numCompleted = new AtomicInteger(0);

    public FullGenomeOEWithinClusters(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                                      int maxIters, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments) {
        this.ds = ds;
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
        this.norm = norm;
        this.maxIters = maxIters;
        DrinkUtils.collapseGWList(origIntraSubcompartments);
        this.origIntraSubcompartments = origIntraSubcompartments;
    }

    public Map<Integer, GenomeWideList<SubcompartmentInterval>> extractFinalGWSubcompartments(File outputDirectory, Random generator) {

        Map<Integer, GenomeWideList<SubcompartmentInterval>> numItersToResults = new HashMap<>();

        final CompositeGenomeWideDensityMatrix interMatrix = new CompositeGenomeWideDensityMatrix(
                chromosomeHandler, ds, norm, resolution, origIntraSubcompartments);
        System.out.println(interMatrix.getLength() + " -v- " + interMatrix.getWidth());

        MatrixTools.saveMatrixTextNumpy(new File(outputDirectory, "data_matrix.npy").getAbsolutePath(), interMatrix.getCleanedData());


        Map<Integer, Integer> subcompartment1IDsToSize = new HashMap<>();

        for (int k = 2; k < 20; k++) {

            GenomeWideList<SubcompartmentInterval> finalCompartments = new GenomeWideList<>(chromosomeHandler);
            launchKmeansGWMatrix(outputDirectory, "final_gw", interMatrix, finalCompartments, generator.nextLong(),
                    subcompartment1IDsToSize, k);
            while (numCompleted.get() < 1) {
                System.out.print(".");
                try {
                    TimeUnit.SECONDS.sleep(10);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println(".");
            numCompleted.set(0);

            numItersToResults.put(k, finalCompartments);
        }


        return numItersToResults;
    }


    private void launchKmeansGWMatrix(File directory, String description, final CompositeGenomeWideDensityMatrix matrix,
                                      final GenomeWideList<SubcompartmentInterval> interSubcompartments,
                                      final long seed, Map<Integer, Integer> subcompartmentIDsToSize,
                                      final int numClusters) {

        if (matrix.getLength() > 0 && matrix.getWidth() > 0) {


            ConcurrentKMeans kMeans = new ConcurrentKMeans(matrix.getCleanedData(), numClusters, maxIters, seed);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                    if (HiCGlobals.printVerboseComments) {
                        System.out.println(s);
                    }
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    numCompleted.incrementAndGet();
                    matrix.processGWKmeansResult(clusters, interSubcompartments, subcompartmentIDsToSize);
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
    }
}
