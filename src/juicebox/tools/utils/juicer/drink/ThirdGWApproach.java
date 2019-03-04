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

package juicebox.tools.utils.juicer.drink;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import juicebox.tools.utils.juicer.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.juicer.drink.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class ThirdGWApproach {

    public static GenomeWideList<SubcompartmentInterval>
    extractFinalGWSubcompartments(Dataset ds, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                                  File outputDirectory, int numClusters, int maxIters, double logThreshold,
                                  GenomeWideList<SubcompartmentInterval> origIntraSubcompartments, int connectedComponentThreshold) {

        AtomicInteger numCompleted = new AtomicInteger(0);

        final ScaledCompositeOddVsEvenInterchromosomalMatrix interMatrix = new ScaledCompositeOddVsEvenInterchromosomalMatrix(
                chromosomeHandler, ds, norm, resolution, origIntraSubcompartments, logThreshold, true);

        File outputFile = new File(outputDirectory, "inter_Odd_vs_Even_matrix_data.txt");
        MatrixTools.exportData(interMatrix.getCleanedData(), outputFile);

        File outputFile2 = new File(outputDirectory, "inter_Even_vs_Odd_matrix_data.txt");
        MatrixTools.exportData(interMatrix.getCleanedTransposedData(), outputFile2);

        GenomeWideList<SubcompartmentInterval> interOddSubcompartments = new GenomeWideList<>(chromosomeHandler);
        launchKmeansInterMatrix(interMatrix, interOddSubcompartments, numClusters, maxIters, numCompleted, false);

        GenomeWideList<SubcompartmentInterval> interEvenSubcompartments = new GenomeWideList<>(chromosomeHandler);
        launchKmeansInterMatrix(interMatrix, interEvenSubcompartments, numClusters, maxIters, numCompleted, true);


        while (numCompleted.get() < 1) {
            System.out.println("So far portion completed is " + numCompleted.get() + "/2");
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        File outputFile3 = new File(outputDirectory, "inter_odd_kmeans_clusters.bed");
        interOddSubcompartments.simpleExport(outputFile3);

        outputFile3 = new File(outputDirectory, "inter_even_kmeans_clusters.bed");
        interEvenSubcompartments.simpleExport(outputFile3);

        GenomeWideList<SubcompartmentInterval> finalSubcompartments = mergeIntraAndInterAnnotations(outputDirectory,
                origIntraSubcompartments, interOddSubcompartments, interEvenSubcompartments, connectedComponentThreshold);

        return finalSubcompartments;
    }

    private static GenomeWideList<SubcompartmentInterval>
    mergeIntraAndInterAnnotations(File outputDirectory, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments,
                                  GenomeWideList<SubcompartmentInterval> interOddSubcompartments,
                                  GenomeWideList<SubcompartmentInterval> interEvenSubcompartments, int connectedComponentThreshold) {

        final Map<SimpleInterval, Set<Integer>> intervalToClusterIDs = new HashMap<>();

        System.out.println("Start Intra List Processing");
        // set the initial set with cluster val
        origIntraSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    Set<Integer> clusterIDs = new HashSet<>();
                    clusterIDs.add(interval.getClusterID());
                    intervalToClusterIDs.put(interval.getSimpleIntervalKey(), clusterIDs);
                }
            }
        });
        System.out.println("End Intra List Processing");

        System.out.println("Start Inter List Processing");

        //odds first
        interOddSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    intervalToClusterIDs.get(interval.getSimpleIntervalKey()).add(interval.getClusterID());
                }
            }
        });

        //even next
        interEvenSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    intervalToClusterIDs.get(interval.getSimpleIntervalKey()).add(interval.getClusterID());
                }
            }
        });


        System.out.println("End Inter List Processing");

        int[][] adjacencyMatrix = ConnectedComponents.generateAdjacencyMatrix(intervalToClusterIDs);

        File outputFile = new File(outputDirectory, "subcompartment_adj_matrix_data.txt");
        //MatrixTools.exportData(MatrixTools.convertToDoubleMatrix(adjacencyMatrix), outputFile);
        MatrixTools.exportData(MatrixTools.convertToDoubleMatrix(adjacencyMatrix), outputFile);

        Set<Set<Integer>> connectedComponents = ConnectedComponents.calculateConnectedComponents(adjacencyMatrix, connectedComponentThreshold);

        return ConnectedComponents.stitchSubcompartments(connectedComponents, origIntraSubcompartments);
    }


    private static void launchKmeansInterMatrix(final ScaledCompositeOddVsEvenInterchromosomalMatrix matrix,
                                                final GenomeWideList<SubcompartmentInterval> interSubcompartments, int numClusters,
                                                int maxIters, final AtomicInteger numCompleted, final boolean isTranspose) {

        if (matrix.getLength() > 0 && matrix.getWidth() > 0) {

            double[][] cleanData;
            if (isTranspose) {
                cleanData = matrix.getCleanedTransposedData();
            } else {
                cleanData = matrix.getCleanedData();
            }
            ConcurrentKMeans kMeans = new ConcurrentKMeans(cleanData, numClusters, maxIters, 128971L);

            KMeansListener kMeansListener = new KMeansListener() {
                @Override
                public void kmeansMessage(String s) {
                }

                @Override
                public void kmeansComplete(Cluster[] clusters, long l) {
                    numCompleted.incrementAndGet();
                    matrix.processGWKmeansResult(clusters, interSubcompartments, isTranspose);
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
