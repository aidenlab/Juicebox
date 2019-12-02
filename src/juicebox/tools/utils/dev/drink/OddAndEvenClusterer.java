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

package juicebox.tools.utils.dev.drink;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeans.Cluster;
import juicebox.tools.utils.dev.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;

import java.io.File;
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

    public GenomeWideList<SubcompartmentInterval> extractFinalGWSubcompartments(File outputDirectory, long[] seeds, CompositeInterchromDensityMatrix.InterMapType mapType) {

        final CompositeInterchromDensityMatrix interMatrix = new CompositeInterchromDensityMatrix(
                chromosomeHandler, ds, norm, resolution, origIntraSubcompartments, mapType);

        //File outputFile = new File(outputDirectory, isOddsVsEvenType + "inter_Odd_vs_Even_matrix_data.txt");
        //MatrixTools.exportData(interMatrix.getCleanedData(), outputFile);

        GenomeWideList<SubcompartmentInterval> finalCompartments = new GenomeWideList<>(chromosomeHandler);
        launchKmeansInterMatrix(interMatrix, finalCompartments, false, seeds[0]);
        launchKmeansInterMatrix(interMatrix, finalCompartments, true, seeds[1]);

        while (numCompleted.get() < 1) {
            System.out.println("So far portion completed is " + numCompleted.get() + "/2");
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return finalCompartments;
    }


    private void launchKmeansInterMatrix(final CompositeInterchromDensityMatrix matrix,
                                         final GenomeWideList<SubcompartmentInterval> interSubcompartments, final boolean isTransposed, final long seed) {

        if (matrix.getLength() > 0 && matrix.getWidth() > 0) {
            double[][] cleanDataWithDeriv;
            if (isTransposed) {
                cleanDataWithDeriv = matrix.getCleanedTransposedData();
            } else {
                cleanDataWithDeriv = matrix.getCleanedData();
            }
            cleanDataWithDeriv = MatrixTools.getMainAppendedDerivativeDownColumn(cleanDataWithDeriv);

            ConcurrentKMeans kMeans = new ConcurrentKMeans(cleanDataWithDeriv, numClusters, maxIters, seed);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    synchronized (numCompleted) {
                        numCompleted.incrementAndGet();
                        matrix.processGWKmeansResult(clusters, interSubcompartments, isTransposed);
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
}
