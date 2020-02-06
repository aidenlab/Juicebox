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
import juicebox.tools.utils.dev.drink.SimpleInterval;
import juicebox.tools.utils.dev.drink.SubcompartmentInterval;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.SingularValueDecomposition;
import org.apache.commons.math.linear.SingularValueDecompositionImpl;

import java.util.*;

public class FourthApproach {

    // https://stackoverflow.com/questions/19957076/best-way-to-compute-a-truncated-singular-value-decomposition-in-java

    private GenomeWideList<SubcompartmentInterval>
    mergeIntraAndInterAnnotations(GenomeWideList<SubcompartmentInterval> origIntraSubcompartments,
                                  GenomeWideList<SubcompartmentInterval> interOddSubcompartments,
                                  GenomeWideList<SubcompartmentInterval> interEvenSubcompartments, int connectedComponentThreshold) {

        final Map<SimpleInterval, Set<Integer>> intervalToClusterIDs = new HashMap<>();

        System.out.println("Start Intra List Processing 1");
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
        System.out.println("End Intra List Processing 1");

        System.out.println("Start Inter List Processing 2");

        //odds first
        interOddSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    intervalToClusterIDs.get(interval.getSimpleIntervalKey()).add(interval.getClusterID());
                }
            }
        });

        //even next
        interEvenSubcompartments.processLists(new FeatureFunction<SubcompartmentInterval>() {
            @Override
            public void process(String chr, List<SubcompartmentInterval> featureList) {
                for (SubcompartmentInterval interval : featureList) {
                    intervalToClusterIDs.get(interval.getSimpleIntervalKey()).add(interval.getClusterID());
                }
            }
        });


        System.out.println("End Inter List Processing 2");

        int[][] adjacencyMatrix = ConnectedComponents.generateAdjacencyMatrix(intervalToClusterIDs);

        Set<Set<Integer>> connectedComponents = ConnectedComponents.calculateConnectedComponents(adjacencyMatrix, connectedComponentThreshold);

        return ConnectedComponents.stitchSubcompartments(connectedComponents, origIntraSubcompartments);
    }

    public static double[][] getTruncatedSVD(double[][] matrix, final int k) {
        SingularValueDecomposition svd = new SingularValueDecompositionImpl(new Array2DRowRealMatrix(matrix));

        double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
        svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);

        double[][] truncatedS = new double[k][k];
        svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);

        double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
        svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

        RealMatrix approximatedSvdMatrix = (new Array2DRowRealMatrix(truncatedU)).multiply(new Array2DRowRealMatrix(truncatedS)).multiply(new Array2DRowRealMatrix(truncatedVT));

        return approximatedSvdMatrix.getData();
    }
}
