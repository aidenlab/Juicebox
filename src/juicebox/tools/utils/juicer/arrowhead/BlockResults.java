/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.Juicer.Arrowhead;

import juicebox.tools.utils.Common.ArrayTools;
import juicebox.tools.utils.Common.MatrixTools;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 6/5/15.
 */
public class BlockResults {

    private List<HighScore> results = new ArrayList<HighScore>();

    public BlockResults(RealMatrix observed, float varThreshold, float signThreshold,
                        ArrowheadScoreList givenList1, ArrowheadScoreList givenList2) {

        int n = Math.min(observed.getRowDimension(), observed.getColumnDimension());
        int gap = 7;

        RealMatrix dUpstream = calculateDirectionalityIndexUpstream(observed, n, gap);
        MatrixTriangles triangles = new MatrixTriangles(dUpstream);

        triangles.generateBlockScoreCalculations();

        triangles.updateScoresUsingList(givenList1);
        triangles.updateScoresUsingList(givenList2);

        triangles.thresholdScoreValues(varThreshold, signThreshold);

        List<Set<Point>> connectedComponents = triangles.extractConnectedComponents();

        results = triangles.calculateResults(connectedComponents);
        plotArrowheadFigures();
    }

    /**
     * TODO
     */
    private void plotArrowheadFigures() {


        // TODO
    }

    /**
     * calculate D upstream, directionality index upstream
     * @param observed
     * @param n
     * @param gap
     * @return dUpstream
     */
    private RealMatrix calculateDirectionalityIndexUpstream(RealMatrix observed, int n, int gap) {

        RealMatrix dUpstream = MatrixTools.cleanArray2DMatrix(n);

        for (int i = 0; i < n; i++) {
            int window = Math.min(n - i - gap, i - gap);
            window = Math.min(window, n);

            double[] row = observed.getRow(i);
            double[] A = ArrayTools.flipArray(ArrayTools.extractArray(row, i - window, i - gap));
            double[] B = ArrayTools.extractArray(row, i + gap, i + window);

            double[] preference = new double[A.length];
            for (int j = 0; j < A.length; j++) {
                preference[j] = (A[j] - B[j]) / (A[j] + B[j]);
            }

            int index = 0;
            for (int j = i + gap; j < i + window + 1; j++) {
                dUpstream.setEntry(i, j, preference[index]);
                index++;
            }
        }

        return dUpstream;
    }

    /**
     * @return block results
     */
    public List<HighScore> getResults() {
        return results;
    }

    public int size() {
        return results.size();
    }
}
