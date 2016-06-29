/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2016 Broad Institute, Aiden Lab
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

package juicebox.tools.utils.juicer.arrowhead;

import juicebox.tools.utils.common.MatrixTools;
import juicebox.tools.utils.juicer.arrowhead.connectedcomponents.BinaryConnectedComponents;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Created by muhammadsaadshamim on 6/5/15.
 */
class MatrixTriangles {

    private boolean initialMatricesNotGenerated = true;
    private boolean blockScoresNotCalculated = true;
    private boolean blockScoresNotThresholded = true;


    private RealMatrix up;
    private RealMatrix upSign;
    private RealMatrix upSquared;
    private RealMatrix lo;
    private RealMatrix loSign;
    private RealMatrix loSquared;

    private RealMatrix upVar;
    private RealMatrix loVar;
    private RealMatrix blockScore;

    /**
     * calculate Bnew, the block score matrix. it's a combination of 3 matrices
     *
     * @param matrix
     */
    public MatrixTriangles(RealMatrix matrix) {
        int n = Math.min(matrix.getRowDimension(), matrix.getColumnDimension());
        up = MatrixTools.cleanArray2DMatrix(n);
        upSign = MatrixTools.cleanArray2DMatrix(n);
        upSquared = MatrixTools.cleanArray2DMatrix(n);
        lo = MatrixTools.cleanArray2DMatrix(n);
        loSign = MatrixTools.cleanArray2DMatrix(n);
        loSquared = MatrixTools.cleanArray2DMatrix(n);

        MatrixTools.setNaNs(matrix, 0);

        //int window= matrix.getRowDimension(); // TODO What? -> "not using this because it messed things up"
        RealMatrix matrixElementwiseSquared = MatrixTools.elementBasedMultiplication(matrix, matrix);
        RealMatrix signMatrix = MatrixTools.sign(matrix);
        RealMatrix onesMatrix = MatrixTools.ones(n);

        //System.out.println("msign "+ matrix.getNorm());
        //System.out.println("sign " + signMatrix.getNorm());

        // Matrices used as dynamic programming lookups.
        // "R" matrices are sums of the columns up to that point: R(1,5) is sum of
        // column 5 from diagonal (row 5) up to row 1
        // "U" matrices are sums of the rows up to the point: U(1,5) is sum of row 5
        // from diagonal (col 1) up to col 5
        // We want mean, mean of sign, and variance, so we are doing the sum then
        // dividing by counts
        RealMatrix rSum = DynamicProgrammingUtils.right(matrix, n);
        RealMatrix rSign = DynamicProgrammingUtils.right(signMatrix, n);
        RealMatrix rSquared = DynamicProgrammingUtils.right(matrixElementwiseSquared, n);
        RealMatrix rCount = DynamicProgrammingUtils.right(onesMatrix, n);

        RealMatrix uSum = DynamicProgrammingUtils.upper(matrix, n);
        RealMatrix uSign = DynamicProgrammingUtils.upper(signMatrix, n);
        RealMatrix uSquared = DynamicProgrammingUtils.upper(matrixElementwiseSquared, n);
        RealMatrix uCount = DynamicProgrammingUtils.upper(onesMatrix, n);

        RealMatrix upCount = MatrixTools.cleanArray2DMatrix(n);
        RealMatrix loCount = MatrixTools.cleanArray2DMatrix(n);

        // Upper triangle
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int bottom = (int) Math.floor((j - i + 1) / 2);
                // add half of column
                up.setEntry(i, j, up.getEntry(i, j - 1) + rSum.getEntry(i, j) - rSum.getEntry(i + bottom, j));
                upSign.setEntry(i, j, upSign.getEntry(i, j - 1) + rSign.getEntry(i, j) - rSign.getEntry(i + bottom, j));
                upSquared.setEntry(i, j, upSquared.getEntry(i, j - 1) + rSquared.getEntry(i, j) - rSquared.getEntry(i + bottom, j));
                upCount.setEntry(i, j, upCount.getEntry(i, j - 1) + rCount.getEntry(i, j) - rCount.getEntry(i + bottom, j));
            }
        }

        // Normalize
        MatrixTools.replaceValue(upCount, 0, 1);
        up = MatrixTools.elementBasedDivision(up, upCount);
        upSign = MatrixTools.elementBasedDivision(upSign, upCount);
        upSquared = MatrixTools.elementBasedDivision(upSquared, upCount);

        // Lower triangle
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                int val = (int) Math.floor((b - a + 1) / 2);
                int endpt = Math.min(2 * b - a, n - 1);
                loCount.setEntry(a, b, loCount.getEntry(a, b - 1) + uCount.getEntry(b, endpt) - rCount.getEntry(a + val, b));
                lo.setEntry(a, b, lo.getEntry(a, b - 1) + uSum.getEntry(b, endpt) - rSum.getEntry(a + val, b));
                loSign.setEntry(a, b, loSign.getEntry(a, b - 1) + uSign.getEntry(b, endpt) - rSign.getEntry(a + val, b));
                loSquared.setEntry(a, b, loSquared.getEntry(a, b - 1) + uSquared.getEntry(b, endpt) - rSquared.getEntry(a + val, b));
            }
        }

        // Normalize
        MatrixTools.replaceValue(loCount, 0, 1);
        lo = MatrixTools.elementBasedDivision(lo, loCount);
        loSign = MatrixTools.elementBasedDivision(loSign, loCount);
        loSquared = MatrixTools.elementBasedDivision(loSquared, loCount);

        initialMatricesNotGenerated = false;
    }

    /**
     * Calculate block scores
     */
    public void generateBlockScoreCalculations() {
        if (initialMatricesNotGenerated) {
            System.out.println("Initial matrices have not been generated");
            System.exit(45);
        }

        upVar = upSquared.subtract(MatrixTools.elementBasedMultiplication(up, up));
        loVar = loSquared.subtract(MatrixTools.elementBasedMultiplication(lo, lo));
        RealMatrix diff = MatrixTools.normalizeByMax(lo.subtract(up));
        RealMatrix diffSign = MatrixTools.normalizeByMax(loSign.subtract(upSign));
        RealMatrix diffSquared = MatrixTools.normalizeByMax((upVar).add(loVar));
        blockScore = (diff.add(diffSign)).subtract(diffSquared);

        blockScoresNotCalculated = false;
    }

    /**
     * Use give thresholds to eliminate extremes
     *
     * @param varThreshold
     * @param signThreshold
     */
    public void thresholdScoreValues(double varThreshold, double signThreshold) {
        if (blockScoresNotCalculated) {
            System.out.println("Block scores not calculated");
            System.exit(46);
        }

        signThresholdInternalValues(blockScore, upSign, loSign, signThreshold);

        if (!Double.isNaN(varThreshold)) {
            varThresholdInternalValues(blockScore, upVar.add(loVar), varThreshold);
        }
        blockScoresNotThresholded = false;
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
     * Threshold sign values in block score matrix, set extremes at either end to zero
     *
     * @param matrix
     * @param upSign
     * @param loSign
     * @param threshold
     */
    private void signThresholdInternalValues(RealMatrix matrix, RealMatrix upSign, RealMatrix loSign, double threshold) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                //System.out.println(upSign.getEntry(i, j)+" "+loSign.getEntry(i, j)+" "+threshold);
                if ((-upSign.getEntry(i, j)) < threshold || loSign.getEntry(i, j) < threshold) {
                    matrix.setEntry(i, j, 0);
                }
            }
        }
    }

    /**
     * extract block scores from regions specified in the provided list
     *
     * @return
     */
    public ArrowheadScoreList updateScoresUsingList(ArrowheadScoreList scoreList, int limStart, int limEnd) {
        if (blockScoresNotCalculated) {
            System.out.println("Block scores not calculated");
            System.exit(47);
        }

        return scoreList.updateActiveIndexScores(blockScore, limStart, limEnd);
    }

    public List<Set<Point>> extractConnectedComponents() {
        if (blockScoresNotThresholded) {
            System.out.println("Scores not fixed for threshold");
            System.exit(48);
        }

        //System.out.println("Norm "+blockScore.getNorm());

        return BinaryConnectedComponents.detection(blockScore.getData(), 0);
    }

    public List<HighScore> calculateResults(List<Set<Point>> connectedComponents) {
        /*  for each connected component, get result for highest scoring point  */
        ArrayList<HighScore> results = new ArrayList<HighScore>();
        for (Set<Point> connectedComponent : connectedComponents) {
            Point score = getHighestScoringPoint(blockScore, connectedComponent);
            int i = score.x, j = score.y;
            results.add(new HighScore(i, j, blockScore.getEntry(i, j), upVar.getEntry(i, j), loVar.getEntry(i, j),
                    -upSign.getEntry(i, j), loSign.getEntry(i, j)));
        }
        return results;
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
}



