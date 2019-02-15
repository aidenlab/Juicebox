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

import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.drink.kmeans.Cluster;
import org.broad.igv.feature.Chromosome;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

class DataCleanerV2 extends DataCleaner {

    private final List<Integer> dataSetSeparatingIndices;
    private final int numDatasets;

    public DataCleanerV2(List<double[][]> data, int indx, double maxPercentAllowedToBeZeroThreshold, int resolution, File outDirectory,
                         List<Integer> emptyDataSetSeparatingIndices) {
        super(aggregateStitch(data, outDirectory, indx, emptyDataSetSeparatingIndices), maxPercentAllowedToBeZeroThreshold, resolution);
        numDatasets = data.size();
        dataSetSeparatingIndices = emptyDataSetSeparatingIndices;

        /*
        File outputFile2 = new File(outDirectory, "PostAggregateOrigChr"+indx+".txt");
        MatrixTools.exportData(getOriginalData(), outputFile2);

        File outputFile3 = new File(outDirectory, "PostAggregateCleanChr"+indx+".txt");
        MatrixTools.exportData(getOriginalData(), outputFile3);
        */

    }

    private static double[][] aggregateStitch(List<double[][]> data, File outDirectory, int indx, List<Integer> dataSetSeparatingIndices) {
        // todo currently assuming each one identical...
        int rowNums = 0;
        int colNums = data.get(0)[0].length;

        for (double[][] mtrx : data) {
            rowNums += mtrx.length;
        }

        //System.out.println("rows,cols "+rowNums+","+colNums);

        double[][] aggregate = new double[rowNums][colNums];

        int rowOffSet = 0;
        for (double[][] region : data) {
            dataSetSeparatingIndices.add(rowOffSet);

            MatrixTools.copyFromAToBRegion(region, aggregate, rowOffSet, 0);
            rowOffSet += region.length;
        }
        //System.out.print("\n");

        //System.out.println(aggregate.length+" - - - "+aggregate[0].length);
        //File outputFile2 = new File(outDirectory, "AggregateChr"+indx+".txt");
        //MatrixTools.exportData(aggregate, outputFile2);

        return aggregate;
    }


    public void processKmeansResultV2(Chromosome chromosome,
                                      List<GenomeWideList<SubcompartmentInterval>> subcompartmentsLists, Cluster[] clusters,
                                      Map<Integer, double[]> idToCentroidMap) {

        List<List<SubcompartmentInterval>> subcompartmentIntervals = new ArrayList<>();
        for (int i = 0; i < numDatasets; i++) {
            subcompartmentIntervals.add(new ArrayList<SubcompartmentInterval>());
        }

        System.out.println("Chromosome " + chromosome.getName() + " clustered into " + clusters.length + " clusters");

        for (Cluster cluster : clusters) {
            int currentClusterID = UniqueSubcompartmentClusterID.tempInitialClusterID.getAndIncrement();
            synchronized (idToCentroidMap) {
                idToCentroidMap.put(currentClusterID, cluster.getCenter());
            }

            for (int i : cluster.getMemberIndexes()) {
                int originalRow = getOriginalIndexRow(i);

                int datasetIndx = determineWhichDatasetThisBelongsTo(originalRow);

                int x1 = (originalRow - dataSetSeparatingIndices.get(datasetIndx)) * getResolution();
                int x2 = x1 + getResolution();

                subcompartmentIntervals.get(datasetIndx).add(
                        new SubcompartmentInterval(chromosome.getIndex(), chromosome.getName(), x1, x2, currentClusterID));
            }
        }

        for (int i = 0; i < numDatasets; i++) {
            SubcompartmentInterval.reSort(subcompartmentsLists.get(i));
            subcompartmentsLists.get(i).addAll(subcompartmentIntervals.get(i));
        }
    }

    private int determineWhichDatasetThisBelongsTo(int originalRow) {
        for (int i = 0; i < numDatasets - 1; i++) {
            if (originalRow < dataSetSeparatingIndices.get(i + 1)) return i;
        }
        return numDatasets - 1;
    }
}
