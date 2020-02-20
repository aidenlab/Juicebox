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

import juicebox.data.feature.FeatureFunction;
import juicebox.data.feature.GenomeWideList;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.dev.drink.SimpleInterval;
import juicebox.tools.utils.dev.drink.SubcompartmentInterval;

import java.io.File;
import java.util.*;

public class SecondGWApproach {


    private static GenomeWideList<SubcompartmentInterval>
    mergeIntraAndInterAnnotations(File outputDirectory, GenomeWideList<SubcompartmentInterval> origIntraSubcompartments,
                                  Map<Integer, GenomeWideList<SubcompartmentInterval>> interSubcompartmentMap, int connectedComponentThreshold) {

        final Map<SimpleInterval, Set<Integer>> intervalToClusterIDs = new HashMap<>();

        System.out.println("Start Intra List Processing");
        // set the initial set with cluster val
        origIntraSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    Set<Integer> clusterIDs = new HashSet<>();
                    clusterIDs.add(interval.getClusterID());
                    intervalToClusterIDs.put(interval.getSimpleIntervalKey(), clusterIDs);
                }
            }
        });
        System.out.println("End Intra List Processing");

        System.out.println("Start Inter List Processing");
        for (GenomeWideList<SubcompartmentInterval> intervalList : interSubcompartmentMap.values()) {
            intervalList.processLists(new FeatureFunction<SubcompartmentInterval>() {
                @Override
                public void process(String chr, List<SubcompartmentInterval> featureList) {
                    for (SubcompartmentInterval interval : featureList) {
                        intervalToClusterIDs.get(interval.getSimpleIntervalKey()).add(interval.getClusterID());
                    }
                }
            });
        }
        System.out.println("End Inter List Processing");

        int[][] adjacencyMatrix = ConnectedComponents.generateAdjacencyMatrix(intervalToClusterIDs);

        File outputFile = new File(outputDirectory, "subcompartment_adj_matrix_data.txt");
        //MatrixTools.exportData(MatrixTools.convertToDoubleMatrix(adjacencyMatrix), outputFile);
        MatrixTools.exportData(MatrixTools.convertToDoubleMatrix(adjacencyMatrix), outputFile);

        Set<Set<Integer>> connectedComponents = ConnectedComponents.calculateConnectedComponents(adjacencyMatrix, connectedComponentThreshold);

        return ConnectedComponents.stitchSubcompartments(connectedComponents, origIntraSubcompartments);
    }
}

