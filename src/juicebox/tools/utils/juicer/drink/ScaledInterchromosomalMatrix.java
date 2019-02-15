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

class ScaledInterchromosomalMatrix {


    private final Chromosome mainVSChromosome;
    private final ChromosomeHandler chromosomeHandler;
    private final NormalizationType norm;
    private final int resolution;
    private final GenomeWideList<SubcompartmentInterval> intraSubcompartments;
    private final double threshold;
    private final double[][] gwCleanMatrix;
    private final Map<Integer, SubcompartmentInterval> indexToIntervalMap = new HashMap<>();

    public ScaledInterchromosomalMatrix(Chromosome mainVSChromosome, ChromosomeHandler chromosomeHandler, Dataset ds, NormalizationType norm, int resolution,
                                        GenomeWideList<SubcompartmentInterval> intraSubcompartments, double threshold) {
        this.mainVSChromosome = mainVSChromosome;
        this.chromosomeHandler = chromosomeHandler;
        this.norm = norm;
        this.resolution = resolution;
        this.intraSubcompartments = intraSubcompartments;
        this.threshold = threshold;

        gwCleanMatrix = makeCleanScaledInterMatrix(ds);
        //System.out.println("Final Size "+gwCleanMatrix.length+" by "+gwCleanMatrix[0].length);
    }

    private double[][] makeCleanScaledInterMatrix(Dataset ds) {

        Chromosome[] chromosomes = chromosomeHandler.getAutosomalChromosomesArray();
        int w = calculateWidthInterMatrix(chromosomes);
        int h = calculateHeightInterMatrix();
        int[] indices = calculateOffsetIndex(chromosomes);
        //System.out.println("Size "+h+" by "+w);

        double[][] interMatrix = new double[h][w];

        Chromosome chr1 = mainVSChromosome;
        for (int j = 0; j < chromosomes.length; j++) {
            Chromosome chr2 = chromosomes[j];

            if (chr1.getIndex() == chr2.getIndex()) continue;

            Matrix matrix = ds.getMatrix(chr1, chr2);

            if (matrix == null) continue;

            HiCZoom zoom = ds.getZoomForBPResolution(resolution);
            final MatrixZoomData zd = matrix.getZoomData(zoom);

            if (zd == null) continue;

            // will need to flip across diagonal
            boolean needToFlip = chr2.getIndex() < chr1.getIndex();
            fillInInterChromosomeRegion(interMatrix, zd, chr1, chr2, indices[j], needToFlip);
        }

        if (w > 0 && h > 0)
            return MatrixTools.transpose(interMatrix);

        return interMatrix;
    }

    private int calculateHeightInterMatrix() {
        return intraSubcompartments.getFeatures("" + mainVSChromosome.getIndex()).size();
    }

    private int calculateWidthInterMatrix(Chromosome[] chromosomes) {
        int total = 0;
        for (Chromosome chromosome : chromosomes) {
            if (chromosome.getIndex() != mainVSChromosome.getIndex()) {
                total += intraSubcompartments.getFeatures("" + chromosome.getIndex()).size();
            }
        }
        return total;
    }

    private int[] calculateOffsetIndex(Chromosome[] chromosomes) {
        int[] indices = new int[chromosomes.length];
        for (int i = 0; i < chromosomes.length - 1; i++) {
            int chrIndex = chromosomes[i].getIndex();

            if (chrIndex == mainVSChromosome.getIndex()) {
                if (i > 0) {
                    indices[i + 1] = indices[i];
                }
            } else {
                indices[i + 1] = indices[i] + intraSubcompartments.getFeatures("" + chrIndex).size();
            }
        }
        return indices;
    }

    private void fillInInterChromosomeRegion(double[][] matrix, MatrixZoomData zd, Chromosome chr1,
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

                for (int j = 0; j < intervals2.size(); j++) {
                    SubcompartmentInterval interv2 = intervals2.get(j);
                    indexToIntervalMap.put(offsetIndex2 + j, interv2);
                    int binYStart = interv2.getX1() / resolution;
                    int binYEnd = Math.min(interv2.getX2() / resolution, lengthChr2);
                    double averagedValue = ExtractingOEDataUtils.extractAveragedOEFromRegion(allDataForRegion,
                            binXStart, binXEnd, binYStart, binYEnd, threshold, false);

                    try {
                        matrix[i][offsetIndex2 + j] = averagedValue;
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

    public void processGWKmeansResult(Cluster[] clusters, GenomeWideList<SubcompartmentInterval> subcompartments) {

        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        System.out.println("GW data vs " + mainVSChromosome.getName() + " clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = UniqueSubcompartmentClusterID.tempInitialClusterID.getAndIncrement();
            //System.out.println("Cluster " + currentClusterID);
            //System.out.println(Arrays.toString(cluster.getMemberIndexes()));
            for (int i : cluster.getMemberIndexes()) {

                SubcompartmentInterval interv1 = indexToIntervalMap.get(i);
                //System.out.println(i + " - " + interv1);

                int chrIndex = interv1.getChrIndex();
                String chrName = interv1.getChrName();
                int x1 = interv1.getX1();
                int x2 = interv1.getX2();

                subcompartmentIntervals.add(
                        new SubcompartmentInterval(chrIndex, chrName, x1, x2, currentClusterID));
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

}
