/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2018 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.curse;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.curse.kmeans.Cluster;
import juicebox.tools.utils.juicer.curse.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.juicer.curse.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class SecondGWApproach {

    public static GenomeWideList<SubcompartmentInterval>
    extractFinalGWSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                                  File outputDirectory, int numClusters, int maxIters, double logThreshold,
                                  GenomeWideList<SubcompartmentInterval> origIntraSubcompartments) {

        Chromosome[] chromosomes = chromosomeHandler.getChromosomeArrayWithoutAllByAll();

        AtomicInteger numToExpect = new AtomicInteger(0);
        AtomicInteger numCompleted = new AtomicInteger(0);

        Map<Integer, GenomeWideList<SubcompartmentInterval>> interSubcompartmentMap = new HashMap<>();

        for (Chromosome chromosome : chromosomes) {
            final ScaledInterchromosomalMatrix interMatrix = new ScaledInterchromosomalMatrix(chromosome,
                    chromosomeHandler, ds, norm, resolution, origIntraSubcompartments, logThreshold);

            File outputFile = new File(outputDirectory, "inter_" + chromosome.getIndex() + "_matrix_data.txt");
            MatrixTools.exportData(interMatrix.getCleanedData(), outputFile);


            GenomeWideList<SubcompartmentInterval> interSubcompartments = new GenomeWideList<>(chromosomeHandler);
            launchKmeansInterMatrix(interMatrix, interSubcompartments, numClusters, maxIters, numToExpect, numCompleted);
            interSubcompartmentMap.put(chromosome.getIndex(), interSubcompartments);


        }

        while (numToExpect.get() > numCompleted.get()) {
            System.out.println("So far portion completed is " + numCompleted.get() + "/" + numToExpect.get());
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        for (Integer key : interSubcompartmentMap.keySet()) {
            File outputFile = new File(outputDirectory, "inter_kmeans_" + key + "_clusters.bed");
            interSubcompartmentMap.get(key).simpleExport(outputFile);
        }

        GenomeWideList<SubcompartmentInterval> finalSubcompartments = mergeIntraAndInterAnnotations(chromosomes, origIntraSubcompartments, interSubcompartmentMap);

        return finalSubcompartments;
    }

    private static GenomeWideList<SubcompartmentInterval>
    mergeIntraAndInterAnnotations(Chromosome[] chromosomes, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments,
                                  Map<Integer, GenomeWideList<SubcompartmentInterval>> interSubcompartmentMap) {

        for (Chromosome chromosome : chromosomes) {

            List<SubcompartmentInterval> origIntervals = origIntraSubcompartments.getFeatures("" + chromosome.getIndex());


        }


        return null;
    }

    private static void launchKmeansInterMatrix(final ScaledInterchromosomalMatrix matrix,
                                                final GenomeWideList<SubcompartmentInterval> interSubcompartments, int numClusters,
                                                int maxIters, AtomicInteger numToExpect, final AtomicInteger numCompleted) {

        if (matrix.getLength() > 0) {

            ConcurrentKMeans kMeans = new ConcurrentKMeans(matrix.getCleanedData(), numClusters,
                    maxIters, 128971L);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    numCompleted.incrementAndGet();
                    matrix.processGWKmeansResult(clusters, interSubcompartments);
                }

                @Override
                public void kmeansError(Throwable throwable) {
                    throwable.printStackTrace();
                    System.err.println("gw curse - err - " + throwable.getLocalizedMessage());
                    System.exit(98);
                }
            };
            kMeans.addKMeansListener(kMeansListener);
            numToExpect.incrementAndGet();
            kMeans.run();

        }
    }
}

