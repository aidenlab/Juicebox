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

import juicebox.data.ChromosomeHandler;
import juicebox.data.Dataset;
import juicebox.data.Matrix;
import juicebox.data.MatrixZoomData;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        Chromosome[] heightChromosomes = oddChromosomes;
        int h = calculateDimensionInterMatrix(oddChromosomes);
        int[] heightIndices = calculateOffsetIndex(oddChromosomes);

        Chromosome[] widthChromosomes = evenChromosomes;
        int w = calculateDimensionInterMatrix(evenChromosomes);
        int[] widthIndices = calculateOffsetIndex(evenChromosomes);


        //System.out.println("Size "+h+" by "+w);

        double[][] interMatrix = new double[h][w];

        for (int i = 0; i < heightChromosomes.length; i++) {
            Chromosome chr1 = heightChromosomes[i];

            for (int j = 0; j < widthChromosomes.length; j++) {
                Chromosome chr2 = widthChromosomes[j];

                if (chr1.getIndex() == chr2.getIndex()) continue;

                Matrix matrix = ds.getMatrix(chr1, chr2);

                if (matrix == null) continue;

                HiCZoom zoom = ds.getZoomForBPResolution(resolution);
                final MatrixZoomData zd = matrix.getZoomData(zoom);

                if (zd == null) continue;

                // will need to flip across diagonal
                boolean needToFlip = chr2.getIndex() < chr1.getIndex();
                fillInInterChromosomeRegion(interMatrix, zd, chr1, heightIndices[i], chr2, widthIndices[j], needToFlip);
            }
        }

        if (w > 0 && h > 0)
            return MatrixTools.transpose(interMatrix);

        return interMatrix;
    }

    private int calculateDimensionInterMatrix(Chromosome[] chromosomes) {
        int total = 0;
        for (Chromosome chromosome : chromosomes) {
            total += intraSubcompartments.getFeatures("" + chromosome.getIndex()).size();
        }
        return total;
    }

    private int[] calculateOffsetIndex(Chromosome[] chromosomes) {
        int[] indices = new int[chromosomes.length];
        for (int i = 0; i < chromosomes.length - 1; i++) {
            int chrIndex = chromosomes[i].getIndex();
            indices[i + 1] = indices[i] + intraSubcompartments.getFeatures("" + chrIndex).size();
        }
        return indices;
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

        try {
            if (intervals1.size() == 0 || intervals2.size() == 0) return;
            double[][] allDataForRegion;
            if (needToFlip) {
                allDataForRegion = ExtractingOEDataUtils.extractLocalOEBoundedRegion(zd, 0, lengthChr2,
                        0, lengthChr1, lengthChr2, lengthChr1, norm, false, null, chr1Index, threshold);
                //System.out.println(allDataForRegion.length+" -- - -- "+allDataForRegion[0].length);
                allDataForRegion = MatrixTools.transpose(allDataForRegion);
                //System.out.println(allDataForRegion.length+" -- flip -- "+allDataForRegion[0].length);
            } else {
                allDataForRegion = ExtractingOEDataUtils.extractLocalOEBoundedRegion(zd, 0, lengthChr1,
                        0, lengthChr2, lengthChr1, lengthChr2, norm, false, null, chr1Index, threshold);
            }

            for (int i = 0; i < intervals1.size(); i++) {
                SubcompartmentInterval interv1 = intervals1.get(i);

                int binXStart = interv1.getX1() / resolution;
                int binXEnd = Math.min(interv1.getX2() / resolution, lengthChr1);
                indexToInterval1Map.put(offsetIndex1 + i, interv1);

                for (int j = 0; j < intervals2.size(); j++) {
                    SubcompartmentInterval interv2 = intervals2.get(j);
                    indexToInterval2Map.put(offsetIndex2 + j, interv2);
                    int binYStart = interv2.getX1() / resolution;
                    int binYEnd = Math.min(interv2.getX2() / resolution, lengthChr2);
                    double averagedValue = ExtractingOEDataUtils.extractAveragedOEFromRegion(allDataForRegion,
                            binXStart, binXEnd, binYStart, binYEnd, threshold, false);

                    try {
                        matrix[offsetIndex1 + i][offsetIndex2 + j] = averagedValue;
                    } catch (Exception e) {
                        //System.err.println("err " + i + ", (" + offsetIndex2 + "+" + j + ")");
                        //System.err.println("err interv1 " + interv1);
                        //System.err.println("err interv2 " + interv2);
                        //System.err.println("err region size " + allDataForRegion.length + " by " + allDataForRegion[0].length);
                        //System.err.println("err matrix size " + matrix.length + " by " + matrix[0].length);

                        e.printStackTrace();
                        System.exit(99);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void processGWKmeansResult(Cluster[] clusters, GenomeWideList<SubcompartmentInterval> subcompartments, boolean isTranspose) {

        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        System.out.println("GW Composite data vs clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = UniqueSubcompartmentClusterID.tempInitialClusterID.getAndIncrement();
            //System.out.println("Cluster " + currentClusterID);
            //System.out.println(Arrays.toString(cluster.getMemberIndexes()));
            for (int i : cluster.getMemberIndexes()) {

                try {
                    SubcompartmentInterval interv;
                    if (isTranspose) {
                        interv = indexToInterval1Map.get(i);
                    } else {
                        interv = indexToInterval2Map.get(i);
                    }

                    //System.out.println(i + " - " + interv1);

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

        SubcompartmentInterval.reSort(subcompartments);
        subcompartments.addAll(subcompartmentIntervals);
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
