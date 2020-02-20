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

import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;
import org.broad.igv.feature.Chromosome;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class DataCleaner {

    final private float[][] cleanData;
    private final Map<Integer, Integer> cleanIndexRowToOriginalIndexRow = new HashMap<>();
    private final Map<Integer, Integer> cleanIndexColToOriginalIndexCol = new HashMap<>();
    final private int resolution;
    private final double maxPercentAllowedToBeZeroThreshold;
    protected final static AtomicInteger initialClusterID = new AtomicInteger(0);

    public DataCleaner(double[][] data, double maxPercentAllowedToBeZeroThreshold, int resolution, double[] convolution1d) {
        this.resolution = resolution;
        this.maxPercentAllowedToBeZeroThreshold = maxPercentAllowedToBeZeroThreshold;
        cleanData = cleanUpData(MatrixTools.smoothAndAppendDerivativeDownColumn(data, convolution1d));
        System.gc();
    }

    private float[][] cleanUpData(double[][] originalData) {

        int numRows = originalData.length;
        int numCols = originalData[0].length;
        int[] numZerosRowIndx = new int[numRows];
        double[] rowSums = new double[numRows];
        int[] numZerosColIndx = new int[numCols];
        double[] columnSums = new double[numCols];

        for (int i = 0; i < originalData.length; i++) {
            for (int j = 0; j < originalData[0].length; j++) {
                if (Double.isNaN(originalData[i][j]) || Double.isInfinite(originalData[i][j]) || isCloseToZero(originalData[i][j])) {
                    originalData[i][j] = 0;
                    numZerosRowIndx[i]++;
                    numZerosColIndx[j]++;
                }
                double absValue = Math.abs(originalData[i][j]);
                rowSums[i] += absValue;
                columnSums[j] += absValue;
            }
        }

        calculateWhichIndicesToKeep(numZerosRowIndx, rowSums, cleanIndexRowToOriginalIndexRow);
        calculateWhichIndicesToKeep(numZerosColIndx, columnSums, cleanIndexColToOriginalIndexCol);
        return MatrixTools.convertToFloatMatrix(makeCleanMatrix(originalData));
    }

    private double[][] makeCleanMatrix(double[][] originalData) {

        int numRows = cleanIndexRowToOriginalIndexRow.keySet().size();
        int numCols = cleanIndexColToOriginalIndexCol.keySet().size();

        double[][] cleanMatrx = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            int i0 = getOriginalIndexRow(i);
            for (int j = 0; j < numCols; j++) {
                int j0 = getOriginalIndexCol(j);
                cleanMatrx[i][j] = originalData[i0][j0];
            }
        }
        return cleanMatrx;
    }


    private void calculateWhichIndicesToKeep(int[] numZerosIndxCount, double[] sumsAlongDimension,
                                             Map<Integer, Integer> cleanIndexToOriginalIndex) {

        int numEntireColAllZeros = 0;
        for (int i0 = 0; i0 < numZerosIndxCount.length; i0++) {
            if (sumsAlongDimension[i0] <= 1) {
                numEntireColAllZeros++;
            }
        }

        int maxNumAllowedToBeZeroCutOff = (int) ((numZerosIndxCount.length - numEntireColAllZeros) * maxPercentAllowedToBeZeroThreshold);
        int counter = 0;

        for (int i0 = 0; i0 < numZerosIndxCount.length; i0++) {
            if (sumsAlongDimension[i0] > 1) {
                if (numZerosIndxCount[i0] - numEntireColAllZeros < maxNumAllowedToBeZeroCutOff) {
                    cleanIndexToOriginalIndex.put(counter, i0);
                    counter++;
                }
            }
        }
    }

    public void postprocessKmeansResult(Chromosome chromosome, GenomeWideList<SubcompartmentInterval> subcompartments, Cluster[] clusters) {
        List<SubcompartmentInterval> subcompartmentIntervals = new ArrayList<>();
        System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = initialClusterID.getAndIncrement();
            for (int i : cluster.getMemberIndexes()) {
                int x1 = getOriginalIndexRow(i) * resolution;
                int x2 = x1 + resolution;

                subcompartmentIntervals.add(
                        new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x1, x2, currentClusterID));
            }
        }

        DrinkUtils.reSort(subcompartments);
        subcompartments.addAll(subcompartmentIntervals);
    }

    private boolean isCloseToZero(double v) {
        return Math.abs(v) < 1E-10;
    }


    public float[][] getCleanedData() {
        return cleanData;
    }

    int getOriginalIndexRow(int i) {
        return cleanIndexRowToOriginalIndexRow.get(i);
    }

    private int getOriginalIndexCol(int i) {
        return cleanIndexColToOriginalIndexCol.get(i);
    }

    public int getLength() {
        return cleanData.length;
    }

    int getResolution() {
        return resolution;
    }
}
