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
import juicebox.data.HiCFileTools;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.dev.drink.kmeans.Cluster;
import juicebox.tools.utils.dev.drink.kmeans.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeans.KMeansListener;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Clusterer {

    /**
     * version with multiple hic files
     */
    final AtomicInteger numRunsToExpect = new AtomicInteger();
    final AtomicInteger numRunsDone = new AtomicInteger();
    // save the kmeans centroid of each cluster
    final Map<Integer, double[]> idToCentroidMap = new HashMap<>();
    private final double maxPercentAllowBeZero = 0.5;
    private final List<Dataset> datasets;
    private final int numDatasets;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;
    private final NormalizationType norm;
    private final int maxIters = 20000;
    private final double logThreshold = 4;
    private final int numClusters;
    private final List<Map<Chromosome, Map<Integer, List<Integer>>>> mapPosIndexToCluster = new ArrayList<>();
    private final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments = new ArrayList<>();
    private final long[] randomSeeds = new long[]{128971L, 22871L};

    public Clusterer(List<Dataset> datasets, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                     int numClusters) {
        this.datasets = datasets;
        numDatasets = datasets.size();
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
        this.norm = norm;
        this.numClusters = numClusters;

        for (int i = 0; i < numDatasets; i++) {
            comparativeSubcompartments.add(new GenomeWideList<>(chromosomeHandler));
            mapPosIndexToCluster.add(new HashMap<>());
        }
    }

    /**
     * @param outputDirectory
     * @param inputHicFilePaths
     */
    public void extractAllComparativeIntraSubcompartmentsTo(File outputDirectory, List<String> inputHicFilePaths) {

        // each ds will need a respective list of assigned subcompartments

        Map<Chromosome, Pair<DataCleanerV2, DataCleanerV2>> dataCleanerV2MapForChrom = getCleanedDatasets();
        for (Chromosome chromosome : dataCleanerV2MapForChrom.keySet()) {
            Pair<DataCleanerV2, DataCleanerV2> dataPair = dataCleanerV2MapForChrom.get(chromosome);
            for (long seed : randomSeeds) {
                launchKMeansClustering(chromosome, dataPair.getFirst(), seed);
                launchKMeansClustering(chromosome, dataPair.getSecond(), seed);
            }
        }

        waitWhileCodeRuns();

        collapseClustersAcrossRuns();

        // process differences for diff vector
        DrinkUtils.writeDiffVectorsRelativeToBaselineToFiles(comparativeSubcompartments, idToCentroidMap, outputDirectory,
                inputHicFilePaths, chromosomeHandler, resolution, "drink_r_" + resolution + "_k_" + numClusters + "_diffs");

        DrinkUtils.writeConsensusSubcompartmentsToFile(comparativeSubcompartments, outputDirectory);

        DrinkUtils.writeFinalSubcompartmentsToFiles(comparativeSubcompartments, outputDirectory, inputHicFilePaths);
    }

    /**
     *
     * @return
     */
    private Map<Chromosome, Pair<DataCleanerV2, DataCleanerV2>> getCleanedDatasets() {
        Map<Chromosome, Pair<DataCleanerV2, DataCleanerV2>> dataCleanerV2MapForChrom = new HashMap<>();

        for (final Chromosome chromosome : chromosomeHandler.getAutosomalChromosomesArray()) {
            try {
                List<double[][]> matrices = new ArrayList<>();

                for (Dataset ds : datasets) {
                    RealMatrix localizedRegionData = HiCFileTools.getRealOEMatrixForChromosome(ds, chromosome, resolution,
                            norm, logThreshold, ExtractingOEDataUtils.ThresholdType.LINEAR_INVERSE_OE_BOUNDED);
                    if (localizedRegionData != null) {
                        matrices.add(localizedRegionData.getData());
                    }
                }

                // can't assess vs non existent map
                if (matrices.size() != datasets.size()) continue;

                DataCleanerV2 dataCleaner = new DataCleanerV2(matrices, maxPercentAllowBeZero, resolution, false);
                DataCleanerV2 dataDerivative = new DataCleanerV2(matrices, maxPercentAllowBeZero, resolution, true);

                if (dataCleaner.getLength() > 0 && dataDerivative.getLength() > 0) {
                    dataCleanerV2MapForChrom.put(chromosome, new Pair<>(dataCleaner, dataDerivative));
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return dataCleanerV2MapForChrom;
    }


    private void launchKMeansClustering(Chromosome chromosome, DataCleanerV2 dataCleaner, long randomSeed) {
        ConcurrentKMeans kMeans = new ConcurrentKMeans(dataCleaner.getCleanedData(), numClusters,
                maxIters, randomSeed);

        numRunsToExpect.incrementAndGet();
        KMeansListener kMeansListener = new KMeansListener() {
            @Override
            public void kmeansMessage(String s) {
            }

            @Override
            public void kmeansComplete(Cluster[] clusters, long l) {
                System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");
                List<Map<Integer, List<Integer>>> mapOfClusterIDForIndexForChrom = dataCleaner.postProcessKmeansResultV2(clusters, idToCentroidMap);
                mapIterationRunToGlobalMap(chromosome, mapOfClusterIDForIndexForChrom);
                numRunsDone.incrementAndGet();
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

    private synchronized void mapIterationRunToGlobalMap(Chromosome chromosome, List<Map<Integer, List<Integer>>> mapOfClusterIDForIndexForChrom) {
        for (int i = 0; i < numDatasets; i++) {
            if (mapPosIndexToCluster.get(i).containsKey(chromosome)) {
                Map<Integer, List<Integer>> mapToUpdate = mapPosIndexToCluster.get(i).get(chromosome);

                Map<Integer, List<Integer>> input = mapOfClusterIDForIndexForChrom.get(i);
                for (Integer key : input.keySet()) {
                    List<Integer> values = mapToUpdate.getOrDefault(key, new ArrayList<>());
                    values.addAll(input.get(key));
                    mapToUpdate.put(key, values);
                }

                mapPosIndexToCluster.get(i).put(chromosome, mapToUpdate);
            } else {
                mapPosIndexToCluster.get(i).put(chromosome, mapOfClusterIDForIndexForChrom.get(i));
            }
        }
    }

    private void waitWhileCodeRuns() {
        int counter = -1;
        while (numRunsDone.get() < numRunsToExpect.get()) {
            if (counter > 0) {
                System.out.println("So far counter is " + counter);
            }
            System.out.println("Wait another minute");
            try {
                TimeUnit.MINUTES.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }


    private void collapseClustersAcrossRuns() {

        int metaCounter = 1;
        Map<String, Integer> stringToMetaID = new HashMap<>();
        List<List<SubcompartmentInterval>> subcompartmentIntervals = new ArrayList<>();

        for (int i = 0; i < numDatasets; i++) {
            Map<Chromosome, Map<Integer, List<Integer>>> chromToIDs = mapPosIndexToCluster.get(i);
            subcompartmentIntervals.add(new ArrayList<>());
            for (Chromosome chromosome : chromToIDs.keySet()) {
                for (Integer index : chromToIDs.get(chromosome).keySet()) {
                    List<Integer> clusterIDs = chromToIDs.get(chromosome).get(index);
                    String idString = getUniqueStringFromIDs(clusterIDs);
                    Integer metaID;
                    if (stringToMetaID.containsKey(idString)) {
                        metaID = stringToMetaID.get(idString);
                    } else {
                        metaID = metaCounter++;
                        stringToMetaID.put(idString, metaID);
                    }

                    int x1 = index * resolution;
                    int x2 = x1 + resolution;

                    subcompartmentIntervals.get(i).add(
                            new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x1, x2, metaID));

                }
            }
        }

        for (int i = 0; i < numDatasets; i++) {
            comparativeSubcompartments.get(i).addAll(subcompartmentIntervals.get(i));
        }

        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            DrinkUtils.reSort(gwList);
        }
    }

    private String getUniqueStringFromIDs(List<Integer> clusterIDs) {
        String idString = "";
        Collections.sort(clusterIDs);
        for (Integer id : clusterIDs) {
            idString += id + ".";
        }
        return idString;
    }
}
