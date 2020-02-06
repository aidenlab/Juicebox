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

import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.kmeansfloat.Cluster;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DataCleanerV2 extends DataCleaner {

    private final List<Integer> dataSetSeparatingIndices = new ArrayList<>();
    private final int numDatasets;

    public DataCleanerV2(List<double[][]> data, double maxPercentAllowedToBeZeroThreshold, int resolution, double[] convolution1d) {
        super(MatrixTools.stitchMultipleMatricesTogetherByRowDim(data), maxPercentAllowedToBeZeroThreshold, resolution, convolution1d);
        numDatasets = data.size();
        determineSeparatingIndices(data);
    }

    private int determineWhichDatasetThisBelongsTo(int originalRow) {
        for (int i = 0; i < numDatasets - 1; i++) {
            if (originalRow < dataSetSeparatingIndices.get(i + 1)) return i;
        }
        return numDatasets - 1;
    }

    private void determineSeparatingIndices(List<double[][]> data) {
        int rowOffSet = 0;
        for (double[][] region : data) {
            dataSetSeparatingIndices.add(rowOffSet);
            rowOffSet += region.length;
        }
    }

    public synchronized List<Map<Integer, List<Integer>>> postProcessKmeansResultV2(Cluster[] clusters,
                                                                                    Map<Integer, float[]> idToCentroidMap) {

        List<Map<Integer, List<Integer>>> mapPosIndexToCluster = new ArrayList<>();
        for (int i = 0; i < numDatasets; i++) {
            mapPosIndexToCluster.add(new HashMap<>());
        }

        for (Cluster cluster : clusters) {
            int currentClusterID = initialClusterID.getAndIncrement();
            synchronized (idToCentroidMap) {
                idToCentroidMap.put(currentClusterID, cluster.getCenter());
            }

            for (int i : cluster.getMemberIndexes()) {
                int originalRow = getOriginalIndexRow(i);
                int datasetIndx = determineWhichDatasetThisBelongsTo(originalRow);
                int xIndex = originalRow - dataSetSeparatingIndices.get(datasetIndx);
                List<Integer> newList = new ArrayList<>();
                newList.add(currentClusterID);
                mapPosIndexToCluster.get(datasetIndx).put(xIndex, newList);
            }
        }

        return mapPosIndexToCluster;
    }
}
