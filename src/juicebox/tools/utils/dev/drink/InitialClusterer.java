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
import juicebox.data.HiCFileTools;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.tools.utils.dev.drink.kmeansfloat.ConcurrentKMeans;
import juicebox.tools.utils.dev.drink.kmeansfloat.KMeansListener;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class InitialClusterer {

    /**
     * version with multiple hic files
     */
    private final AtomicInteger numRunsToExpect = new AtomicInteger();
    private final AtomicInteger numRunsDone = new AtomicInteger();
    private final double maxPercentAllowBeZero = 0.75;
    private final int maxIters = 20000;
    private final List<Dataset> datasets;
    private final int numDatasets;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;
    private final NormalizationType norm;
    private final double logThreshold;
    private final long[] randomSeeds;
    private final int numClusters;
    private final List<Map<Chromosome, Map<Integer, List<Integer>>>> mapPosIndexToCluster = new ArrayList<>();
    private final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments = new ArrayList<>();
    private Map<Integer, float[]> idToCentroidMap = new HashMap<>();
    private double[] convolution1d;

    public InitialClusterer(List<Dataset> datasets, ChromosomeHandler chromosomeHandler, int resolution, NormalizationType norm,
                            int numClusters, Random generator, float logThreshold, double[] convolution1d, int numIters) {
        this.datasets = datasets;
        numDatasets = datasets.size();
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
        this.norm = norm;
        this.numClusters = numClusters;
        this.randomSeeds = new long[numIters];
        for (int i = 0; i < numIters; i++) {
            randomSeeds[i] = generator.nextLong();
        }

        this.logThreshold = logThreshold;
        this.convolution1d = convolution1d;

        for (int i = 0; i < numDatasets; i++) {
            comparativeSubcompartments.add(new GenomeWideList<>(chromosomeHandler));
            mapPosIndexToCluster.add(new HashMap<>());
        }
    }

    private static Map<Integer, float[]> generateMetaCentroidMap(Map<Integer, float[]> originalIDToCentroidMap, Map<Integer, List<Integer>> metaCIDtoOriginalCIDs) {

        Map<Integer, float[]> metaIDToCentroidMap = new HashMap<>();
        for (Integer metaID : metaCIDtoOriginalCIDs.keySet()) {
            List<Integer> origIDs = metaCIDtoOriginalCIDs.get(metaID);
            int comboLength = 0;
            for (Integer origID : origIDs) {
                comboLength += originalIDToCentroidMap.get(origID).length;
            }

            float[] comboVector = new float[comboLength];
            int offsetIndex = 0;
            for (Integer origID : origIDs) {
                float[] currentVector = originalIDToCentroidMap.get(origID);
                System.arraycopy(currentVector, 0, comboVector, offsetIndex, currentVector.length);
                offsetIndex += currentVector.length;
            }

            metaIDToCentroidMap.put(metaID, comboVector);
        }

        return metaIDToCentroidMap;
    }

    public Pair<List<GenomeWideList<SubcompartmentInterval>>, Map<Integer, float[]>> extractAllComparativeIntraSubcompartmentsTo(File outputDirectory) {

        // each ds will need a respective list of assigned subcompartments

        Map<Chromosome, DataCleanerV2> dataCleanerV2MapForChrom = getCleanedDatasets(outputDirectory);
        for (long seed : randomSeeds) {
            if (HiCGlobals.printVerboseComments) {
                System.out.println("** Cluster with seed " + seed);
            }
            for (Chromosome chromosome : dataCleanerV2MapForChrom.keySet()) {
                DataCleanerV2 cleanedData = dataCleanerV2MapForChrom.get(chromosome);
                launchKMeansClustering(chromosome, cleanedData, seed);
            }
            waitWhileCodeRuns();
        }

        Map<Integer, List<Integer>> metaCIDtoOriginalCIDs = collapseClustersAcrossRuns();

        Map<Integer, float[]> metaIDToCentroidMap = generateMetaCentroidMap(idToCentroidMap, metaCIDtoOriginalCIDs);

        return new Pair<>(comparativeSubcompartments, metaIDToCentroidMap);
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
                if (HiCGlobals.printVerboseComments) {
                    System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");
                }
                System.out.print(".");

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

    /**
     * @return
     */
    private Map<Chromosome, DataCleanerV2> getCleanedDatasets(File outputDirectory) {
        Map<Chromosome, DataCleanerV2> dataCleanerV2MapForChrom = new HashMap<>();

        for (final Chromosome chromosome : chromosomeHandler.getAutosomalChromosomesArray()) {
            try {
                List<double[][]> matrices = new ArrayList<>();

                for (Dataset ds : datasets) {
                    RealMatrix localizedRegionData = HiCFileTools.getRealOEMatrixForChromosome(ds, chromosome, resolution,
                            norm, logThreshold, ExtractingOEDataUtils.ThresholdType.LOG_OE_BOUNDED, true);
                    if (localizedRegionData != null) {
                        matrices.add(localizedRegionData.getData());
                        if (HiCGlobals.printVerboseComments) {
                            MatrixTools.saveMatrixTextNumpy(new File(outputDirectory, chromosome.getName() + "_matrix.npy").getAbsolutePath(), localizedRegionData.getData());
                        }
                    }
                }

                // can't assess vs non existent map
                if (matrices.size() != datasets.size() || matrices.size() < 1) continue;

                DataCleanerV2 dataCleaner = new DataCleanerV2(matrices, maxPercentAllowBeZero, resolution, convolution1d);

                if (dataCleaner.getLength() > 0) {
                    dataCleanerV2MapForChrom.put(chromosome, dataCleaner);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return dataCleanerV2MapForChrom;
    }

    private Map<Integer, List<Integer>> collapseClustersAcrossRuns() {

        int metaCounter = 1;
        Map<String, Integer> stringToMetaID = new HashMap<>();
        List<List<SubcompartmentInterval>> subcompartmentIntervals = new ArrayList<>();

        Map<Integer, List<Integer>> metaIDtoPriorIDs = new HashMap<>();

        for (int i = 0; i < numDatasets; i++) {
            Map<Chromosome, Map<Integer, List<Integer>>> chromToIDs = mapPosIndexToCluster.get(i);
            subcompartmentIntervals.add(new ArrayList<>());
            for (Chromosome chromosome : chromToIDs.keySet()) {
                for (Integer index : chromToIDs.get(chromosome).keySet()) {
                    List<Integer> clusterIDs = chromToIDs.get(chromosome).get(index);
                    String idString = getUniqueStringFromIDs(clusterIDs);
                    Collections.sort(clusterIDs);
                    Integer metaID;
                    if (stringToMetaID.containsKey(idString)) {
                        metaID = stringToMetaID.get(idString);
                    } else {
                        metaID = metaCounter++;
                        stringToMetaID.put(idString, metaID);
                        metaIDtoPriorIDs.put(metaID, clusterIDs);
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

        return metaIDtoPriorIDs;
    }

    private String getUniqueStringFromIDs(List<Integer> clusterIDs) {
        String idString = "";
        for (Integer id : clusterIDs) {
            idString += id + ".";
        }
        return idString;
    }
}
