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

package juicebox.tools.dev;

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.dev.drink.DataCleaner;
import juicebox.tools.utils.dev.drink.DataCleanerV2;
import juicebox.tools.utils.dev.drink.DrinkUtils;
import juicebox.tools.utils.dev.drink.SubcompartmentInterval;
import juicebox.tools.utils.dev.drink.kmeans.Cluster;
import juicebox.tools.utils.dev.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class DRINKSUtils {
    /**
     * version with multiple hic files
     */

    public static void extractAllComparativeIntraSubcompartments(
            List<Dataset> datasets, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm, double logThreshold,
            double maxPercentAllowedToBeZeroThreshold, int numClusters, int maxIters, File outputDirectory, List<String> inputHicFilePaths) {

        final AtomicInteger numRunsToExpect = new AtomicInteger();
        final AtomicInteger numRunsDone = new AtomicInteger();

        // save the kmeans centroid of each cluster
        final Map<Integer, double[]> idToCentroidMap = new HashMap<>();

        // each ds will need a respective list of assigned subcompartments
        final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments = new ArrayList<>();
        for (int i = 0; i < datasets.size(); i++) {
            comparativeSubcompartments.add(new GenomeWideList<>(chromosomeHandler));
        }

        for (final Chromosome chromosome : chromosomeHandler.getChromosomeArrayWithoutAllByAll()) {

            try {
                List<double[][]> matrices = new ArrayList<>();

                for (Dataset ds : datasets) {

                    RealMatrix localizedRegionData = HiCFileTools.getRealOEMatrixForChromosome(ds, chromosome, resolution, norm, logThreshold);
                    if (localizedRegionData != null) {
                        matrices.add(localizedRegionData.getData());
                    }
                }

                // e.g. diploid Y chromosome; can't assess vs non existent map
                if (matrices.size() != datasets.size()) continue;

                final DataCleanerV2 dataCleanerV2 = new DataCleanerV2(matrices, chromosome.getIndex(),
                        maxPercentAllowedToBeZeroThreshold, resolution, outputDirectory, new ArrayList<>());

                if (dataCleanerV2.getLength() > 0) {
                    launchKMeansForInitialSubcompartment(chromosome, dataCleanerV2, null, numClusters,
                            maxIters, numRunsToExpect, numRunsDone, comparativeSubcompartments, idToCentroidMap);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        waitWhileCodeRuns(numRunsDone, numRunsToExpect, -1);

        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            DrinkUtils.reSort(gwList);
        }

        // process differences for diff vector
        DrinkUtils.writeDiffVectorsRelativeToBaselineToFiles(comparativeSubcompartments, idToCentroidMap, outputDirectory,
                inputHicFilePaths, chromosomeHandler, resolution, "drink_r_" + resolution + "_k_" + numClusters + "_diffs");

        DrinkUtils.writeConsensusSubcompartmentsToFile(comparativeSubcompartments, outputDirectory);

        DrinkUtils.writeFinalSubcompartmentsToFiles(comparativeSubcompartments, outputDirectory, inputHicFilePaths);
    }

    private static void launchKMeansForInitialSubcompartment(Chromosome chromosome, DataCleaner dataCleaner,
                                                             GenomeWideList<SubcompartmentInterval> subcompartments,
                                                             int numClusters, int maxIters,
                                                             AtomicInteger numRunsToExpect, AtomicInteger numRunsDone,
                                                             List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments,
                                                             Map<Integer, double[]> idToCentroidMap) {
        ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleaner.getCleanedData(), numClusters,
                maxIters, 128971L);

        numRunsToExpect.incrementAndGet();
        KMeansListener kMeansListener = new KMeansListener() {
            @Override
            public void kmeansMessage(String s) {
            }

            @Override
            public void kmeansComplete(Cluster[] clusters, long l) {
                numRunsDone.incrementAndGet();
                if (dataCleaner instanceof DataCleanerV2) {
                    ((DataCleanerV2) dataCleaner).processKmeansResultV2(chromosome, comparativeSubcompartments, clusters,
                            idToCentroidMap);
                } else {
                    dataCleaner.processKmeansResult(chromosome, subcompartments, clusters);
                }
            }

            @Override
            public void kmeansError(Throwable throwable) {
                System.err.println("drink chr " + chromosome.getName() + " - err - " + throwable.getLocalizedMessage());
                throwable.printStackTrace();
                System.exit(98);
            }
        };
        kMeans.addKMeansListener(kMeansListener);
        kMeans.run();
    }

    private static void waitWhileCodeRuns(AtomicInteger numRunsDone, AtomicInteger numRunsToExpect, int size) {
        while (numRunsDone.get() < numRunsToExpect.get()) {
            if (size > 0) {
                System.out.println("So far size is " + size);
            }
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
