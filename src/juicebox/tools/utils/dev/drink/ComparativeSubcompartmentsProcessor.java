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

import com.google.common.primitives.Ints;
import juicebox.data.ChromosomeHandler;
import juicebox.data.feature.FeatureFilter;
import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.dev.drink.kmeansfloat.ClusterTools;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ComparativeSubcompartmentsProcessor {

    private final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments;
    private final Map<String, Double> distanceMatrix;
    private final ChromosomeHandler chromosomeHandler;
    private final int resolution;

    public ComparativeSubcompartmentsProcessor(Pair<List<GenomeWideList<SubcompartmentInterval>>, Map<Integer, float[]>> initialSubcompartmentsData, ChromosomeHandler chromosomeHandler, int resolution) {
        this.comparativeSubcompartments = initialSubcompartmentsData.getFirst();
        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            DrinkUtils.reSort(gwList);
        }
        distanceMatrix = calculateVectorDifferencesMatrix(initialSubcompartmentsData.getSecond());
        this.chromosomeHandler = chromosomeHandler;
        this.resolution = resolution;
    }

    private static double calculateTopThreshold(double[] data) {
        int numNonZero = 0;
        double total = 0;
        for (double val : data) {
            if (val > 0.001) {
                total += val;
                numNonZero += 1;
            }
        }

        return total / numNonZero;
    }

    private static Map<String, Double> calculateVectorDifferencesMatrix(Map<Integer, float[]> idToCentroidMap) {
        Map<String, Double> differences = new HashMap<>();
        for (Integer indx1 : idToCentroidMap.keySet()) {

            String key0 = getClusterPairID(indx1, indx1);
            differences.put(key0, 0.0);

            int n1 = idToCentroidMap.get(indx1).length;
            for (Integer indx2 : idToCentroidMap.keySet()) {
                int n2 = idToCentroidMap.get(indx2).length;
                String key1 = getClusterPairID(indx1, indx2);
                String key2 = getClusterPairID(indx2, indx1);
                if (n1 == n2 && !indx1.equals(indx2) && !differences.containsKey(key1)) {
                    double distance = ClusterTools.getDistance(idToCentroidMap.get(indx1), idToCentroidMap.get(indx2));
                    differences.put(key1, distance);
                    differences.put(key2, distance);
                }
            }
        }
        return differences;
    }

    // todo mss
    // variableStep chrom=chr2 span=5
    // 300701  12.5
    private static void writeClusterCenterToWigAndBed(Chromosome chromosome, double[] data, List<SubcompartmentInterval> featureList,
                                                      final FileWriter fwWig, final FileWriter fwBed, int resolution, double threshold) {
        try {
            // write to wig file
            fwWig.write("fixedStep chrom=chr" + chromosome.getName() + " start=1" + " step=" + resolution + " span=" + resolution + "\n");
            for (double d : data) {
                if (d > threshold) {
                    fwWig.write(d + "\n");
                } else {
                    fwWig.write("0.0\n");
                }
            }

            // write to bed file
            try {
                for (int k = 0; k < data.length; k++) {
                    SubcompartmentInterval interval = featureList.get(k);
                    if (interval.getDifferenceFromControl() > threshold) {
                        fwBed.write("chr" + chromosome.getName() + "\t" + interval.getX1() + "\t" + interval.getX2() + "\n");
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static GenomeWideList<SubcompartmentInterval> calculateConsensus(
            final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments) {

        GenomeWideList<SubcompartmentInterval> control = comparativeSubcompartments.get(0);
        GenomeWideList<SubcompartmentInterval> consensus = control.deepClone();

        final Map<SimpleInterval, Integer> modeOfClusterIdsInInterval = new HashMap<>();

        control.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> controlList) {

                Map<SimpleInterval, List<Integer>> clusterIdsInInterval = new HashMap<>();
                for (SubcompartmentInterval sInterval : controlList) {
                    List<Integer> ids = new ArrayList<>();
                    ids.add(sInterval.getClusterID());
                    clusterIdsInInterval.put(sInterval.getSimpleIntervalKey(), ids);
                }

                if (comparativeSubcompartments.size() > 1) {
                    for (int i = 1; i < comparativeSubcompartments.size(); i++) {
                        for (SubcompartmentInterval sInterval : comparativeSubcompartments.get(i).getFeatures(chr)) {
                            if (clusterIdsInInterval.containsKey(sInterval.getSimpleIntervalKey())) {
                                clusterIdsInInterval.get(sInterval.getSimpleIntervalKey()).add(sInterval.getClusterID());
                            } else {
                                List<Integer> ids = new ArrayList<>();
                                ids.add(sInterval.getClusterID());
                                clusterIdsInInterval.put(sInterval.getSimpleIntervalKey(), ids);
                            }
                        }
                    }
                }

                for (SimpleInterval sInterval : clusterIdsInInterval.keySet()) {
                    int mode = mode(clusterIdsInInterval.get(sInterval));
                    if (mode >= 0) {
                        modeOfClusterIdsInInterval.put(sInterval, mode);
                    }
                }
            }
        });

        consensus.filterLists(new FeatureFilter<SubcompartmentInterval>() {
            @Override
            public List<SubcompartmentInterval> filter(String chr, List<SubcompartmentInterval> featureList) {

                List<SubcompartmentInterval> newIntervals = new ArrayList<>();

                for (SubcompartmentInterval sInterval : featureList) {
                    SimpleInterval key = sInterval.getSimpleIntervalKey();
                    if (modeOfClusterIdsInInterval.containsKey(key)) {
                        sInterval.setClusterID(modeOfClusterIdsInInterval.get(key));
                        newIntervals.add(sInterval);
                    }
                }
                return newIntervals;
            }
        });

        return consensus;
    }

    private static void calculateDifferences(
            final List<GenomeWideList<SubcompartmentInterval>> comparativeSubcompartments, final Map<String, Double> distanceMatrix) {

        GenomeWideList<SubcompartmentInterval> control = comparativeSubcompartments.get(0);

        control.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> controlList) {

                Map<SimpleInterval, Integer> controlClusterIdsForInterval = new HashMap<>();
                for (SubcompartmentInterval sInterval : controlList) {
                    controlClusterIdsForInterval.put(sInterval.getSimpleIntervalKey(), sInterval.getClusterID());
                }

                if (comparativeSubcompartments.size() > 1) {
                    for (int i = 1; i < comparativeSubcompartments.size(); i++) {
                        for (SubcompartmentInterval sInterval : comparativeSubcompartments.get(i).getFeatures(chr)) {
                            if (controlClusterIdsForInterval.containsKey(sInterval.getSimpleIntervalKey())) {

                                int controlID = controlClusterIdsForInterval.get(sInterval.getSimpleIntervalKey());
                                int expID = sInterval.getClusterID();
                                String key = getClusterPairID(controlID, expID);
                                if (distanceMatrix.containsKey(key)) {
                                    double diff = distanceMatrix.get(key);
                                    sInterval.setDifferenceFromControl(diff);
                                } else {
                                    System.err.println("Missing cluster pair key " + key);
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * @param integers
     * @return mode only if there is exactly one value repeated the max number of times
     */
    private static Integer mode(List<Integer> integers) {

        Map<Integer, Integer> counts = new HashMap<>();
        for (Integer i : integers) {
            if (counts.containsKey(i)) {
                counts.put(i, counts.get(i) + 1);
            } else {
                counts.put(i, 1);
            }
        }

        int maxNumTimesCount = Ints.max(Ints.toArray(counts.values()));
        int howManyTimesDoesMaxAppear = 0;
        int potentialModeVal = -1;
        for (Integer val : counts.keySet()) {
            if (counts.get(val).equals(maxNumTimesCount)) {
                potentialModeVal = val;
                howManyTimesDoesMaxAppear++;
            }
        }

        if (howManyTimesDoesMaxAppear == 1) {
            return potentialModeVal;
        }

        return -1;
    }

    private static String getClusterPairID(Integer indx1, Integer indx2) {
        return indx1 + "_" + indx2;
    }

    public void writeDiffVectorsRelativeToBaselineToFiles(File outputDirectory, final List<String> inputHicFilePaths, String uniqueString) {
        // get all differences
        calculateDifferences(comparativeSubcompartments, distanceMatrix);

        for (int i = 1; i < comparativeSubcompartments.size(); i++) {

            String outputFileStem = uniqueString + DrinkUtils.cleanUpPath(inputHicFilePaths.get(i)) + "__VS__" + DrinkUtils.cleanUpPath(inputHicFilePaths.get(0));

            File outputWigFile = new File(outputDirectory, outputFileStem + ".wig");
            File outputBEDFile = new File(outputDirectory, outputFileStem + ".bed");
            File outputStringentWigFile = new File(outputDirectory, "top_" + outputFileStem + ".wig");
            File outputStringentBEDFile = new File(outputDirectory, "top_" + outputFileStem + ".bed");

            try {
                final FileWriter fwWIG = new FileWriter(outputWigFile);
                final FileWriter fwBED = new FileWriter(outputBEDFile);
                final FileWriter fwStringentWIG = new FileWriter(outputStringentWigFile);
                final FileWriter fwStringentBED = new FileWriter(outputStringentBEDFile);

                comparativeSubcompartments.get(i).processLists(new FeatureFunction<SubcompartmentInterval>() {
                    @Override
                    public void process(String chr, List<SubcompartmentInterval> featureList) {

                        double[] differences = new double[featureList.size()];
                        for (int k = 0; k < differences.length; k++) {
                            differences[k] = featureList.get(k).getDifferenceFromControl();
                        }
                        Chromosome chromosome = chromosomeHandler.getChromosomeFromIndex(Integer.parseInt(chr));

                        writeClusterCenterToWigAndBed(chromosome, differences, featureList, fwWIG, fwBED, resolution, 0);
                        double avgThreshold = calculateTopThreshold(differences);
                        writeClusterCenterToWigAndBed(chromosome, differences, featureList, fwStringentWIG, fwStringentBED, resolution, avgThreshold);

                    }
                });
                fwWIG.close();
                fwBED.close();
                fwStringentWIG.close();
                fwStringentBED.close();

            } catch (IOException e) {
                System.err.println("Unable to open file for exporting GWList");
            }
        }
    }

    public void writeConsensusSubcompartmentsToFile(File outputDirectory) {
        GenomeWideList<SubcompartmentInterval> consensus = calculateConsensus(comparativeSubcompartments);
        File outputFile3 = new File(outputDirectory, "consensus_result_intra_compare_file.bed");
        consensus.simpleExport(outputFile3);
    }

    public void writeFinalSubcompartmentsToFiles(File outputDirectory, List<String> inputHicFilePaths) {

        for (GenomeWideList<SubcompartmentInterval> gwList : comparativeSubcompartments) {
            DrinkUtils.collapseGWList(gwList);
        }

        for (int i = 0; i < comparativeSubcompartments.size(); i++) {
            File outputFile2 = new File(outputDirectory, "intra_compare_file_" + DrinkUtils.cleanUpPath(inputHicFilePaths.get(i)) + ".bed");
            comparativeSubcompartments.get(i).simpleExport(outputFile2);
        }
    }

}
