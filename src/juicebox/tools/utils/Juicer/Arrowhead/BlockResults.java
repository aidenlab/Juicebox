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
import juicebox.tools.utils.Juicer.Arrowhead.ConnectedComponents.BinaryConnectedComponents;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 6/5/15.
 */
public class BlockResults {

    private List<HighScore> results;
    private double[] scoreList1, scoreList2;

    public BlockResults(RealMatrix observed, double varThreshold, int signThreshold,
                        List<Integer[]> givenList1, List<Integer[]> givenList2) {

        if (givenList1 == null)
            givenList1 = new ArrayList<Integer[]>();
        if (givenList2 == null)
            givenList2 = new ArrayList<Integer[]>();

        int n = Math.min(observed.getRowDimension(), observed.getColumnDimension());
        int gap = 7;
        RealMatrix dUpstream = calculateDirectionalityIndexUpstream(observed, n, gap);
        MatrixTriangles triangles = DynamicProgrammingUtils.generateTriangles(dUpstream);

        RealMatrix up = triangles.getUp();
        RealMatrix upSign = triangles.getUpSign();
        RealMatrix upSquared = triangles.getUpSquared();
        RealMatrix upVar = upSquared.subtract(MatrixTools.elementBasedMultiplication(up, up));

        RealMatrix lo = triangles.getLo();
        RealMatrix loSign = triangles.getLoSign();
        RealMatrix loSquared = triangles.getLoSquared();
        RealMatrix loVar = loSquared.subtract(MatrixTools.elementBasedMultiplication(lo, lo));

        RealMatrix diff = MatrixTools.normalizeByMax(lo.subtract(up));
        RealMatrix diffSign = MatrixTools.normalizeByMax(loSign.subtract(upSign));
        RealMatrix diffSquared = MatrixTools.normalizeByMax((upVar).add(loVar));
        RealMatrix blockScore = (diff.add(diffSign)).subtract(diffSquared);

        scoreList1 = extractScoresUsingList(blockScore, givenList1);
        scoreList2 = extractScoresUsingList(blockScore, givenList2);

        signThresholdInternalValues(blockScore, upSign, loSign, signThreshold);

        if (varThreshold != 1000) {
            varThresholdInternalValues(blockScore, upVar.add(loVar), varThreshold);
        }

        List<Set<Point>> connectedComponents = BinaryConnectedComponents.detection(blockScore.getData(), 0);

        /*  for each connected component, get result for highest scoring point  */
        results = new ArrayList<HighScore>();
        for (Set<Point> component : connectedComponents) {
            Point score = getHighestScoringPoint(blockScore, component);
            int i = score.x, j = score.y;
            results.add(new HighScore(i, j, blockScore.getEntry(i, j), upVar.getEntry(i, j), loVar.getEntry(i, j),
                    -upSign.getEntry(i, j), loSign.getEntry(i, j)));
        }
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
     * Find the point within the connected component with the highest block score
     *
     * @param blockScore
     * @param component
     * @return scorePoint
     */
    private Point getHighestScoringPoint(RealMatrix blockScore, Set<Point> component) {
        Point scorePoint = component.iterator().next();
        double highestScore = blockScore.getEntry(scorePoint.x, scorePoint.y);

        for (Point point : component) {
            double score = blockScore.getEntry(point.x, point.y);
            if (score > highestScore) {
                highestScore = score;
                scorePoint = new Point(point);
            }
        }
        return new Point(scorePoint);
    }

    /**
     * Threshold values in block score matrix, set extremes to zero
     *
     * @param matrix
     * @param thresholdSums
     * @param threshold
     */
    private void varThresholdInternalValues(RealMatrix matrix, RealMatrix thresholdSums, double threshold) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (thresholdSums.getEntry(i, j) > threshold) {
                    matrix.setEntry(i, j, 0);
                }
            }
        }
    }

    /**
     * Threshold values in block score matrix, set extremes at either end to zero
     *
     * @param matrix
     * @param upSign
     * @param loSign
     * @param threshold
     */
    private void signThresholdInternalValues(RealMatrix matrix, RealMatrix upSign, RealMatrix loSign, int threshold) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if ((-upSign.getEntry(i, j)) < threshold || loSign.getEntry(i, j) < threshold) {
                    matrix.setEntry(i, j, 0);
                }
            }
        }
    }

    /**
     * extract block scores from regions specified in the provided list
     *
     * @param scoreMatrix
     * @param indexList
     * @return
     */
    private double[] extractScoresUsingList(RealMatrix scoreMatrix, List<Integer[]> indexList) {
        double[] scores = new double[indexList.size()];
        for (int i = 0; i < scores.length; i++) {
            Integer[] indices = indexList.get(i);
            scores[i] = MatrixTools.calculateMax(scoreMatrix.getSubMatrix(indices[0], indices[1], indices[2], indices[3]));
        }
        return scores;
    }

    /**
     * @return block results
     */
    public List<HighScore> getResults() {
        return results;
    }

    /**
     * @return list1 corresponding scores
     */
    public double[] getScoreList1() {
        return scoreList1;
    }

    /**
     * @return list2 corresponding scores
     */
    public double[] getScoreList2() {
        return scoreList2;
    }

    /**
     * Wrapper for arrowhead blockbuster results
     */
    public class HighScore {
        private int i;
        private int j;
        private double score;
        private double uVarScore;
        private double lVarScore;
        private double upSign;
        private double loSign;

        public HighScore(int i, int j, double score, double uVarScore, double lVarScore,
                         double upSign, double loSign) {
            this.i = i;
            this.j = j;
            this.score = score;
            this.uVarScore = uVarScore;
            this.lVarScore = lVarScore;
            this.upSign = upSign;
            this.loSign = loSign;
        }

        public String toString() {
            return "" + i + "\t" + j + "\t" + score + "\t" + uVarScore + "\t" + lVarScore + "\t" + upSign + "\t" + loSign;
        }
    }
}