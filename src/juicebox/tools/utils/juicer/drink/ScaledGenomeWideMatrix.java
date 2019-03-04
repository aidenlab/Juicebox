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

package juicebox.tools.utils.juicer.drink;

import juicebox.data.*;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.*;


class ScaledGenomeWideMatrix {

    private final ChromosomeHandler chromosomeHandler;
    private final NormalizationType norm;
    private final int resolution;
    private final GenomeWideList<SubcompartmentInterval> intraSubcompartments;
    private final double threshold;
    private final double[][] gwCleanMatrix;


    public ScaledGenomeWideMatrix(ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm, int resolution,
                                  GenomeWideList<SubcompartmentInterval> intraSubcompartments, double threshold) {
        this.chromosomeHandler = chromosomeHandler;
        this.norm = norm;
        this.resolution = resolution;
        this.intraSubcompartments = intraSubcompartments;
        this.threshold = threshold;

        gwCleanMatrix = makeCleanScaledGWMatrix(ds);
    }

    private double[][] makeCleanScaledGWMatrix(Dataset ds) {

        Chromosome[] chromosomes = chromosomeHandler.getAutosomalChromosomesArray();
        int n = calculateSizeGWMatrix(chromosomes);
        int[] indices = calculateOffsetIndex(chromosomes);
        System.out.println("Size " + n);

        double[][] gwMatrix = new double[n][n];

        for (int i = 0; i < chromosomes.length; i++) {
            Chromosome chr1 = chromosomes[i];
            for (int j = i; j < chromosomes.length; j++) {
                Chromosome chr2 = chromosomes[j];

                boolean isIntra = chr1.getIndex() == chr2.getIndex();
                Matrix matrix = ds.getMatrix(chr1, chr2);

                if (matrix == null) continue;

                HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                final MatrixZoomData zd = matrix.getZoomData(zoom);

                if (zd == null) continue;

                ExpectedValueFunction df = ds.getExpectedValues(zd.getZoom(), norm);
                if (isIntra && df == null) {
                    System.err.println("O/E data not available at " + chr1.getName() + " " + zoom + " " + norm);
                    System.exit(14);
                }

                fillInChromosomeRegion(gwMatrix, zd, df, isIntra, chr1, indices[i], chr2, indices[j]);
            }
        }

        return gwMatrix;
    }

    private final Map<Integer, SubcompartmentInterval> indexToIntervalMap = new HashMap<>();

    private int calculateSizeGWMatrix(Chromosome[] chromosomes) {
        int total = 0;
        for (Chromosome chromosome : chromosomes) {
            total += intraSubcompartments.getFeatures("" + chromosome.getIndex()).size();
        }
        return total;
    }

    private int[] calculateOffsetIndex(Chromosome[] chromosomes) {
        int[] indices = new int[chromosomes.length];
        for (int i = 0; i < chromosomes.length - 1; i++) {

            indices[i + 1] = indices[i] + intraSubcompartments.getFeatures("" + chromosomes[i].getIndex()).size();
        }
        return indices;
    }

    private void fillInChromosomeRegion(double[][] matrix, MatrixZoomData zd, ExpectedValueFunction df, boolean isIntra,
                                        Chromosome chr1, int offsetIndex1, Chromosome chr2, int offsetIndex2) {

        int chr1Index = chr1.getIndex();
        int lengthChr1 = chr1.getLength() / resolution;
        int lengthChr2 = chr2.getLength() / resolution;
        List<SubcompartmentInterval> intervals1 = intraSubcompartments.getFeatures("" + chr1.getIndex());
        List<SubcompartmentInterval> intervals2 = intraSubcompartments.getFeatures("" + chr2.getIndex());

        try {
            if (intervals1.size() == 0 || intervals2.size() == 0) return;
            double[][] allDataForRegion = ExtractingOEDataUtils.extractLocalOEBoundedRegion(zd, 0, lengthChr1,
                    0, lengthChr2, lengthChr1, lengthChr2, norm, isIntra, df, chr1Index, threshold);

            for (int i = 0; i < intervals1.size(); i++) {
                SubcompartmentInterval interv1 = intervals1.get(i);
                indexToIntervalMap.put(offsetIndex1 + i, interv1);
                int binXStart = interv1.getX1() / resolution;
                int binXEnd = Math.min(interv1.getX2() / resolution, lengthChr1);

                for (int j = 0; j < intervals2.size(); j++) {
                    SubcompartmentInterval interv2 = intervals2.get(j);
                    int binYStart = interv2.getX1() / resolution;
                    int binYEnd = Math.min(interv2.getX2() / resolution, lengthChr2);

                    double averagedValue = ExtractingOEDataUtils.extractAveragedOEFromRegion(allDataForRegion,
                            binXStart, binXEnd, binYStart, binYEnd, threshold, isIntra);

                    matrix[offsetIndex1 + i][offsetIndex2 + j] = averagedValue;
                    if (!isIntra) {
                        matrix[offsetIndex2 + j][offsetIndex1 + i] = averagedValue;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void processGWKmeansResult(Cluster[] clusters, GenomeWideList<SubcompartmentInterval> subcompartments) {

        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        System.out.println("GW data clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = UniqueSubcompartmentClusterID.tempInitialClusterID.getAndIncrement();
            System.out.println("Cluster " + currentClusterID);
            System.out.println(Arrays.toString(cluster.getMemberIndexes()));
            for (int i : cluster.getMemberIndexes()) {

                SubcompartmentInterval interv1 = indexToIntervalMap.get(i);
                System.out.println(i + " - " + interv1);

                int chrIndex = interv1.getChrIndex();
                String chrName = interv1.getChrName();
                int x1 = interv1.getX1();
                int x2 = interv1.getX2();

                subcompartmentIntervals.add(
                        new SubcompartmentInterval(chrIndex, chrName, x1, x2, currentClusterID));
            }
        }
        System.out.println("Cluster counter0 at " + UniqueSubcompartmentClusterID.tempInitialClusterID.get());

        SubcompartmentInterval.reSort(subcompartments);
        subcompartments.addAll(subcompartmentIntervals);
    }

    public int getLength() {
        return gwCleanMatrix.length;
    }

    public double[][] getCleanedData() {
        return gwCleanMatrix;
    }
}
