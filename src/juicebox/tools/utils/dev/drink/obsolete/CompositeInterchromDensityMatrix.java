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

package juicebox.tools.utils.dev.drink.obsolete;

import juicebox.HiCGlobals;
import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.HiCFileTools;
import juicebox.data.MatrixZoomData;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.DrinkUtils;
import juicebox.tools.utils.dev.drink.SubcompartmentInterval;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import juicebox.windowui.NormalizationType;
import org.apache.commons.math.linear.RealMatrix;
import org.broad.igv.feature.Chromosome;
import org.broad.igv.util.Pair;

import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class CompositeInterchromDensityMatrix {

    private final NormalizationType norm;
    private final int resolution;
    private final GenomeWideList<SubcompartmentInterval> intraSubcompartments;
    private final float[][] gwCleanMatrix, transposedGWCleanMatrix;
    private final Map<Integer, SubcompartmentInterval> indexToInterval1Map = new HashMap<>();
    private final Map<Integer, SubcompartmentInterval> indexToInterval2Map = new HashMap<>();
    private final Chromosome[] rowsChromosomes;
    private final Chromosome[] colsChromosomes;

    public final static AtomicInteger genomewideInitialClusterID = new AtomicInteger(0);

    public CompositeInterchromDensityMatrix(ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm, int resolution,
                                            GenomeWideList<SubcompartmentInterval> intraSubcompartments,
                                            InterMapType mapType) {
        this.norm = norm;
        this.resolution = resolution;
        this.intraSubcompartments = intraSubcompartments;

        switch (mapType) {
            case SKIP_BY_TWOS: // but start with CHR 1 separate
                Pair<Chromosome[], Chromosome[]> splitByTwos = chromosomeHandler.splitAutosomesAndSkipByTwos();
                rowsChromosomes = splitByTwos.getFirst();
                colsChromosomes = splitByTwos.getSecond();
                break;
            case FIRST_HALF_VS_SECOND_HALF:
                Pair<Chromosome[], Chromosome[]> firstHalfVsSecondHalf = chromosomeHandler.splitAutosomesIntoHalves();
                rowsChromosomes = firstHalfVsSecondHalf.getFirst();
                colsChromosomes = firstHalfVsSecondHalf.getSecond();
                break;
            case ODDS_VS_EVENS:
            default:
                rowsChromosomes = chromosomeHandler.extractOddOrEvenAutosomes(true);
                colsChromosomes = chromosomeHandler.extractOddOrEvenAutosomes(false);
                break;
        }

        gwCleanMatrix = makeCleanScaledInterMatrix(ds);
        transposedGWCleanMatrix = MatrixTools.transpose(gwCleanMatrix);
    }

    private float[][] makeCleanScaledInterMatrix(Dataset ds) {

        // assuming Odd vs Even
        // height chromosomes
        Pair<Integer, int[]> rowsDimension = calculateDimensionInterMatrix(rowsChromosomes);

        // width chromosomes
        Pair<Integer, int[]> colsDimension = calculateDimensionInterMatrix(colsChromosomes);

        float[][] interMatrix = new float[rowsDimension.getFirst()][colsDimension.getFirst()];
        for (int i = 0; i < rowsChromosomes.length; i++) {
            Chromosome chr1 = rowsChromosomes[i];

            for (int j = 0; j < colsChromosomes.length; j++) {
                Chromosome chr2 = colsChromosomes[j];

                if (chr1.getIndex() == chr2.getIndex()) continue;
                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, resolution);
                if (zd == null) continue;

                // will need to flip across diagonal
                boolean needToFlip = chr2.getIndex() < chr1.getIndex();
                fillInInterChromosomeRegion(interMatrix, zd, chr1, rowsDimension.getSecond()[i], chr2, colsDimension.getSecond()[j], needToFlip);
            }
        }

        return interMatrix;
    }

    public enum InterMapType {ODDS_VS_EVENS, FIRST_HALF_VS_SECOND_HALF, SKIP_BY_TWOS}

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

    private void fillInInterChromosomeRegion(float[][] matrix, MatrixZoomData zd, Chromosome chr1, int offsetIndex1,
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
        float[][] allDataForRegion = null;
        try {
            if (needToFlip) {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr2, 0, lengthChr1, lengthChr2, lengthChr1, norm, false);
                allDataForRegionMatrix = allDataForRegionMatrix.transpose();
                allDataForRegion = MatrixTools.convertToFloatMatrix(allDataForRegionMatrix.getData());
            } else {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr1, 0, lengthChr2, lengthChr1, lengthChr2, norm, false);
                allDataForRegion = MatrixTools.convertToFloatMatrix(allDataForRegionMatrix.getData());
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
        Map<String, Float> allContactsBetweenClusters = new HashMap<>();

        for (SubcompartmentInterval interv1 : intervals1) {
            Integer id1 = interv1.getClusterID();
            for (SubcompartmentInterval interv2 : intervals2) {
                Integer id2 = interv2.getClusterID();
                String regionKey = id1 + "-" + id2;

                float countsBetweenClusters = getSumTotalCounts(allDataForRegion, interv1, lengthChr1, interv2, lengthChr2);
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

        Map<String, Float> densityBetweenClusters = new HashMap<>();
        for (String key : allAreaBetweenClusters.keySet()) {
            densityBetweenClusters.put(key, 100f * allContactsBetweenClusters.get(key) / allAreaBetweenClusters.get(key));
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
                float density = densityBetweenClusters.get(regionKey);
                updateMasterMatrixWithRegionalDensities(matrix, density, interv1, internalOffset1, numRows, interv2, internalOffset2, numCols);
                internalOffset2 += numCols;
            }
            internalOffset1 += numRows;
        }
    }

    private void updateMasterMatrixWithRegionalDensities(float[][] matrix, float density,
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

    private float getSumTotalCounts(float[][] allDataForRegion, SubcompartmentInterval interv1, int lengthChr1,
                                    SubcompartmentInterval interv2, int lengthChr2) {
        float total = 0;
        int binXStart = interv1.getX1() / resolution;
        int binXEnd = Math.min(interv1.getX2() / resolution, lengthChr1);

        int binYStart = interv2.getX1() / resolution;
        int binYEnd = Math.min(interv2.getX2() / resolution, lengthChr2);

        for (int i = binXStart; i < binXEnd; i++) {
            for (int j = binYStart; j < binYEnd; j++) {
                if (!Float.isNaN(allDataForRegion[i][j])) {
                    total += allDataForRegion[i][j];
                }
            }
        }
        return total;
    }

    public synchronized void processGWKmeansResult(Cluster[] clusters, GenomeWideList<SubcompartmentInterval> subcompartments,
                                                   boolean isTranspose, Map<Integer, Integer> subcompartmentIDsToSize) {

        Set<SubcompartmentInterval> subcompartmentIntervals = new HashSet<>();
        System.out.println("GW Composite data vs clustered into " + clusters.length + " clusters");

        Map<Integer, SubcompartmentInterval> indexToIntervalMap;
        if (isTranspose) {
            indexToIntervalMap = indexToInterval2Map;
        } else {
            indexToIntervalMap = indexToInterval1Map;
        }


        for (Cluster cluster : clusters) {
            int currentClusterID = genomewideInitialClusterID.getAndIncrement();
            subcompartmentIDsToSize.put(currentClusterID, cluster.getMemberIndexes().length);

            if (HiCGlobals.printVerboseComments) {
                System.out.println("Size of cluster " + currentClusterID + " - " + cluster.getMemberIndexes().length);
            }

            for (int i : cluster.getMemberIndexes()) {

                try {
                    SubcompartmentInterval interv;

                    if (indexToIntervalMap.containsKey(i)) {
                        interv = indexToIntervalMap.get(i);
                        if (interv == null) continue; // probably a zero row

                        int chrIndex = interv.getChrIndex();
                        String chrName = interv.getChrName();
                        int x1 = interv.getX1();
                        int x2 = interv.getX2();

                        subcompartmentIntervals.add(
                                new SubcompartmentInterval(chrIndex, chrName, x1, x2, currentClusterID));
                    } else {
                        System.err.println("is weird error?");
                    }

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


    public void stitchTogetherResults(GenomeWideList<SubcompartmentInterval> preFinalCompartments, Dataset ds, File directory, String filename) {

        Map<String, Float> allContactsBetweenClusters = new HashMap<>();
        Map<String, Integer> allAreasBetweenClusters = new HashMap<>();
        Set<Integer> rowIDs = new HashSet<>();
        Set<Integer> colIDs = new HashSet<>();

        for (int i = 0; i < rowsChromosomes.length; i++) {
            Chromosome chr1 = rowsChromosomes[i];

            for (int j = 0; j < colsChromosomes.length; j++) {
                Chromosome chr2 = colsChromosomes[j];

                if (chr1.getIndex() == chr2.getIndex()) continue;
                final MatrixZoomData zd = HiCFileTools.getMatrixZoomData(ds, chr1, chr2, resolution);
                if (zd == null) continue;

                // will need to flip across diagonal
                boolean needToFlip = chr2.getIndex() < chr1.getIndex();
                getContactsBetweenGroups(zd, chr1, chr2, needToFlip, preFinalCompartments, allContactsBetweenClusters, allAreasBetweenClusters, rowIDs, colIDs);
            }
        }

        double[][] contacts = new double[rowIDs.size() + 1][colIDs.size() + 1];
        double[][] areas = new double[rowIDs.size() + 1][colIDs.size() + 1];
        double[][] interactions = new double[rowIDs.size() + 1][colIDs.size() + 1];
        List<Integer> rowIDsOrdered = new ArrayList<>(rowIDs);
        List<Integer> colIDsOrdered = new ArrayList<>(colIDs);
        for (int i = 1; i < rowIDsOrdered.size() + 1; i++) {
            int rID = rowIDsOrdered.get(i - 1);
            contacts[i][0] = rID;
            areas[i][0] = rID;
            interactions[i][0] = rID;

            for (int j = 1; j < colIDsOrdered.size() + 1; j++) {
                int cID = colIDsOrdered.get(j - 1);
                contacts[0][j] = cID;
                areas[0][j] = cID;
                interactions[0][j] = cID;

                String key = rID + "-" + cID;

                if (allAreasBetweenClusters.containsKey(key)) {
                    contacts[i][j] = allContactsBetweenClusters.get(key);
                    areas[i][j] = allAreasBetweenClusters.get(key);
                    interactions[i][j] = allContactsBetweenClusters.get(key) / allAreasBetweenClusters.get(key);
                }
            }
        }
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".contacts.npy").getAbsolutePath(), contacts);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".areas.npy").getAbsolutePath(), areas);
        MatrixTools.saveMatrixTextNumpy(new File(directory, filename + ".density.npy").getAbsolutePath(), interactions);

    }

    private void getContactsBetweenGroups(MatrixZoomData zd, Chromosome chr1, Chromosome chr2, boolean needToFlip,
                                          GenomeWideList<SubcompartmentInterval> preFinalCompartments,
                                          Map<String, Float> allContactsBetweenClusters,
                                          Map<String, Integer> allAreasBetweenClusters,
                                          Set<Integer> rowIDs, Set<Integer> colIDs) {

        int chr1Index = chr1.getIndex();
        int chr2Index = chr2.getIndex();
        if (chr1Index == chr2Index) {
            System.err.println("Same chr " + chr1.getName());
            System.exit(989);
        }

        int lengthChr1 = chr1.getLength() / resolution;
        int lengthChr2 = chr2.getLength() / resolution;
        List<SubcompartmentInterval> intervals1 = preFinalCompartments.getFeatures("" + chr1.getIndex());
        List<SubcompartmentInterval> intervals2 = preFinalCompartments.getFeatures("" + chr2.getIndex());

        if (intervals1.size() == 0 || intervals2.size() == 0) return;
        float[][] allDataForRegion = null;
        try {
            if (needToFlip) {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr2, 0, lengthChr1, lengthChr2, lengthChr1, norm, false);
                allDataForRegionMatrix = allDataForRegionMatrix.transpose();
                allDataForRegion = MatrixTools.convertToFloatMatrix(allDataForRegionMatrix.getData());
            } else {
                RealMatrix allDataForRegionMatrix = HiCFileTools.extractLocalBoundedRegion(zd, 0, lengthChr1, 0, lengthChr2, lengthChr1, lengthChr2, norm, false);
                allDataForRegion = MatrixTools.convertToFloatMatrix(allDataForRegionMatrix.getData());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(99);
        }

        if (allDataForRegion == null) {
            System.err.println("Missing Interchromosomal Data " + zd.getKey());
            return;
        }

        for (SubcompartmentInterval interv1 : intervals1) {
            Integer id1 = interv1.getClusterID();
            rowIDs.add(id1);
            for (SubcompartmentInterval interv2 : intervals2) {
                Integer id2 = interv2.getClusterID();
                colIDs.add(id2);
                String regionKey = id1 + "-" + id2;

                float countsBetweenClusters = getSumTotalCounts(allDataForRegion, interv1, lengthChr1, interv2, lengthChr2);
                int areaBetweenClusters = interv1.getWidthForResolution(resolution) * interv2.getWidthForResolution(resolution);

                if (allContactsBetweenClusters.containsKey(regionKey)) {
                    allContactsBetweenClusters.put(regionKey, allContactsBetweenClusters.get(regionKey) + countsBetweenClusters);
                    allAreasBetweenClusters.put(regionKey, allAreasBetweenClusters.get(regionKey) + areaBetweenClusters);
                } else {
                    allContactsBetweenClusters.put(regionKey, countsBetweenClusters);
                    allAreasBetweenClusters.put(regionKey, areaBetweenClusters);
                }
            }
        }
    }

    public int getLength() {
        return gwCleanMatrix.length;
    }

    public float[][] getCleanedData() {
        return gwCleanMatrix;
    }

    public float[][] getCleanedTransposedData() {
        return transposedGWCleanMatrix;
    }

    public int getWidth() {
        return gwCleanMatrix[0].length;
    }
}
