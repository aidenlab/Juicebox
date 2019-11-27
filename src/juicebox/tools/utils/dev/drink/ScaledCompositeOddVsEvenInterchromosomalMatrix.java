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
import juicebox.data.MatrixZoomData;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeans.Cluster;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.util.*;

class ScaledCompositeOddVsEvenInterchromosomalMatrix {

    private final ChromosomeHandler chromosomeHandler;
    private final NormalizationType norm;
    private final int resolution;
    private final GenomeWideList<SubcompartmentInterval> intraSubcompartments;
    private final double threshold;
    private final double[][] gwCleanMatrix, transposedGWCleanMatrix;
    private final Map<Integer, SubcompartmentInterval> indexToInterval1Map = new HashMap<>();
    private final Map<Integer, SubcompartmentInterval> indexToInterval2Map = new HashMap<>();

    public ScaledCompositeOddVsEvenInterchromosomalMatrix(ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm, int resolution,
                                                          GenomeWideList<SubcompartmentInterval> intraSubcompartments, double threshold,
                                                          boolean isOddVsEven) {
        this.chromosomeHandler = chromosomeHandler;
        this.norm = norm;
        this.resolution = resolution;
        this.intraSubcompartments = intraSubcompartments;
        this.threshold = threshold;

        gwCleanMatrix = makeCleanScaledInterMatrix(ds);
        transposedGWCleanMatrix = MatrixTools.transpose(gwCleanMatrix);
        //System.out.println("Final Size "+gwCleanMatrix.length+" by "+gwCleanMatrix[0].length);
    }

    private double[][] makeCleanScaledInterMatrix(Dataset ds) {

        Chromosome[] oddChromosomes = chromosomeHandler.extractOddOrEvenAutosomes(true);
        Chromosome[] evenChromosomes = chromosomeHandler.extractOddOrEvenAutosomes(false);

        // assuming Odd vs Even
        // height chromosomes
        Pair<Integer, int[]> oddsDimension = calculateDimensionInterMatrix(oddChromosomes);

        // width chromosomes
        Pair<Integer, int[]> evenDimension = calculateDimensionInterMatrix(evenChromosomes);

        double[][] interMatrix = new double[oddsDimension.getFirst()][evenDimension.getFirst()];
        for (int i = 0; i < oddChromosomes.length; i++) {
            Chromosome chr1 = oddChromosomes[i];

            for (int j = 0; j < evenChromosomes.length; j++) {
                Chromosome chr2 = evenChromosomes[j];

                if (chr1.getIndex() == chr2.getIndex()) continue;
                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, resolution);
                if (zd == null) continue;

                // will need to flip across diagonal
                boolean needToFlip = chr2.getIndex() < chr1.getIndex();
                fillInInterChromosomeRegion(interMatrix, zd, chr1, oddsDimension.getSecond()[i], chr2, evenDimension.getSecond()[j], needToFlip);
            }
        }

        return interMatrix;
    }

    private Pair<Integer, int[]> calculateDimensionInterMatrix(Chromosome[] chromosomes) {
        int total = 0;
        int[] indices = new int[chromosomes.length];

        for (int i = 0; i < chromosomes.length; i++) {
            for (SubcompartmentInterval interval : intraSubcompartments.getFeatures("" + chromosomes[i].getIndex())) {
                total += interval.getWidthForResolution(resolution);
            }
            if (i < chromosomes.length - 1) {
                indices[i + 1] = total;
            }
        }

        return new Pair<>(total, indices);
    }

    private void fillInInterChromosomeRegion(double[][] matrix, MatrixZoomData zd, Chromosome chr1, int offsetIndex1,
                                             Chromosome chr2, int offsetIndex2, boolean needToFlip) {

        int chr1Index = chr1.getIndex();
        int chr2Index = chr2.getIndex();
        if (chr1Index == chr2Index) {
            System.err.println("Same chr " + chr1.getName());
            System.exit(989);
        }

        int lengthChr1 = chr1.getLength() / resolution;
        int lengthChr2 = chr2.getLength() / resolution;
        List<SubcompartmentInterval> intervals1 = intraSubcompartments.getFeatures("" + chr1.getIndex());
        List<SubcompartmentInterval> intervals2 = intraSubcompartments.getFeatures("" + chr2.getIndex());

        if (intervals1.size() == 0 || intervals2.size() == 0) return;
        double[][] allDataForRegion = null;
        try {
            if (needToFlip) {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr2, 0, lengthChr1, lengthChr2, lengthChr1, norm, false);
                allDataForRegionMatrix = allDataForRegionMatrix.transpose();
                allDataForRegion = allDataForRegionMatrix.getData();
            } else {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr1, 0, lengthChr2, lengthChr1, lengthChr2, norm, false);
                allDataForRegion = allDataForRegionMatrix.getData();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(99);
        }

        if (allDataForRegion == null) {
            System.err.println("Missing Interchromosomal Data " + zd.getKey());
            return;
        }

        Map<String, Integer> allAreaBetweenClusters = new HashMap<>();
        Map<String, Double> allContactsBetweenClusters = new HashMap<>();

        for (SubcompartmentInterval interv1 : intervals1) {
            Integer id1 = interv1.getClusterID();
            for (SubcompartmentInterval interv2 : intervals2) {
                Integer id2 = interv2.getClusterID();
                String regionKey = id1 + "-" + id2;

                double countsBetweenClusters = getSumTotalCounts(allDataForRegion, interv1, lengthChr1, interv2, lengthChr2);
                int areaBetweenClusters = interv1.getWidthForResolution(resolution) * interv2.getWidthForResolution(resolution);

                if (allAreaBetweenClusters.containsKey(regionKey)) {
                    allAreaBetweenClusters.put(regionKey, allAreaBetweenClusters.get(regionKey) + areaBetweenClusters);
                    allContactsBetweenClusters.put(regionKey, allContactsBetweenClusters.get(regionKey) + countsBetweenClusters);
                } else {
                    allAreaBetweenClusters.put(regionKey, areaBetweenClusters);
                    allContactsBetweenClusters.put(regionKey, countsBetweenClusters);
                }
            }
        }

        Map<String, Double> densityBetweenClusters = new HashMap<>();
        for (String key : allAreaBetweenClusters.keySet()) {
            densityBetweenClusters.put(key, 100. * allContactsBetweenClusters.get(key) / allAreaBetweenClusters.get(key));
        }

        int internalOffset1 = offsetIndex1;
        for (SubcompartmentInterval interv1 : intervals1) {
            Integer id1 = interv1.getClusterID();
            int numRows = interv1.getWidthForResolution(resolution);

            int internalOffset2 = offsetIndex2;
            for (SubcompartmentInterval interv2 : intervals2) {
                Integer id2 = interv2.getClusterID();
                int numCols = interv2.getWidthForResolution(resolution);

                String regionKey = id1 + "-" + id2;
                double density = densityBetweenClusters.get(regionKey);
                updateMasterMatrixWithRegionalDensities(matrix, density, interv1, internalOffset1, numRows, interv2, internalOffset2, numCols);
                internalOffset2 += numCols;
            }
            internalOffset1 += numRows;
        }
    }

    private void updateMasterMatrixWithRegionalDensities(double[][] matrix, double density,
                                                         SubcompartmentInterval interv1, int offsetIndex1, int numRows,
                                                         SubcompartmentInterval interv2, int offsetIndex2, int numCols) {
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                indexToInterval1Map.put(offsetIndex1 + i, interv1);
                indexToInterval2Map.put(offsetIndex2 + j, interv2);
                matrix[offsetIndex1 + i][offsetIndex2 + j] = density;
            }
        }
    }

    private double getSumTotalCounts(double[][] allDataForRegion, SubcompartmentInterval interv1, int lengthChr1,
                                     SubcompartmentInterval interv2, int lengthChr2) {
        double total = 0;
        int binXStart = interv1.getX1() / resolution;
        int binXEnd = Math.min(interv1.getX2() / resolution, lengthChr1);

        int binYStart = interv2.getX1() / resolution;
        int binYEnd = Math.min(interv2.getX2() / resolution, lengthChr2);

        for (int i = binXStart; i < binXEnd; i++) {
            for (int j = binYStart; j < binYEnd; j++) {
                if (!Double.isNaN(allDataForRegion[i][j])) {
                    total += allDataForRegion[i][j];
                }
            }
        }
        return total;
    }

    public synchronized void processGWKmeansResult(Cluster[] clusters, GenomeWideList<SubcompartmentInterval> subcompartments, boolean isTranspose) {

        Set<SubcompartmentInterval> subcompartmentIntervals = new HashSet<>();
        System.out.println("GW Composite data vs clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = UniqueSubcompartmentClusterID.genomewideInitialClusterID.getAndIncrement();

            for (int i : cluster.getMemberIndexes()) {

                try {
                    SubcompartmentInterval interv;
                    if (isTranspose) {
                        interv = indexToInterval2Map.get(i);
                    } else {
                        interv = indexToInterval1Map.get(i);
                    }
                    if (interv == null) continue; // probably a zero row

                    int chrIndex = interv.getChrIndex();
                    String chrName = interv.getChrName();
                    int x1 = interv.getX1();
                    int x2 = interv.getX2();

                    subcompartmentIntervals.add(
                            new SubcompartmentInterval(chrIndex, chrName, x1, x2, currentClusterID));
                } catch (Exception e) {
                    System.err.println(i + " - " + isTranspose);
                    e.printStackTrace();
                    System.exit(87);
                }
            }
        }

        subcompartments.addAll(new ArrayList<>(subcompartmentIntervals));
        DrinkUtils.reSort(subcompartments);
    }

    public int getLength() {
        return gwCleanMatrix.length;
    }

    public double[][] getCleanedData() {
        return gwCleanMatrix;
    }

    public double[][] getCleanedTransposedData() {
        return transposedGWCleanMatrix;
    }

    public int getWidth() {
        return gwCleanMatrix[0].length;
    }
}
