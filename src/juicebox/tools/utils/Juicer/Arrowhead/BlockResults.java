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

    public BlockResults(RealMatrix observed, double varThreshold, int signThreshold) {

        // BLOCK_FINDER  Find blocks in input observed matrix.
        
        /**
         * if (nargin < 4)
         * list = [];
         * list1=[];
         */
        List<Integer[]> list = new ArrayList<Integer[]>();
        List<Integer[]> list1 = new ArrayList<Integer[]>();

        //calculate D upstream, directionality index upstream
        int n = Math.min(observed.getRowDimension(), observed.getColumnDimension());
        int gap=7;



        RealMatrix dUpstream = MatrixTools.cleanArray2DMatrix(n);

        for(int i = 0; i < n; i++){
            int window = Math.min(n-i-gap, i-gap);
            window = Math.min(window, n);

            double[] row = observed.getRow(i);
            double[] A = ArrayTools.flipArray(ArrayTools.extractArray(row, i-window,i-gap));
            double[] B = ArrayTools.extractArray(row, i+gap, i+window);

            double[] preference = new double[A.length];
            for(int j = 0; j < A.length; j++){
                preference[j] = (A[j]-B[j])/(A[j]+B[j]);
            }

            int index = 0;
            for(int j = i+gap; j < i+window+1; j++){
                dUpstream.setEntry(i,j,preference[index]);
                index++;
            }
        }

        MatrixTriangles triangles = DynamicProgrammingUtils.generateTriangles(dUpstream);

        RealMatrix up = triangles.getUp();
        RealMatrix lo = triangles.getLo();
        RealMatrix upSign = triangles.getUpSign();
        RealMatrix loSign = triangles.getLoSign();
        RealMatrix upSquared = triangles.getUpSquared();
        RealMatrix loSquared = triangles.getLoSquared();

        RealMatrix upVar = upSquared.subtract(MatrixTools.elementBasedMultiplication(up, up));
        RealMatrix loVar = loSquared.subtract(MatrixTools.elementBasedMultiplication(lo, lo));

        RealMatrix diff = MatrixTools.normalizeByMax(lo.subtract(up));
        RealMatrix diffSign = MatrixTools.normalizeByMax(loSign.subtract(upSign));
        RealMatrix diffSquared = MatrixTools.normalizeByMax((upVar).add(loVar));

        RealMatrix blockScore = (diff.add(diffSign)).subtract(diffSquared);


        double[] scores = extractScoresUsingList(blockScore, list);
        double[] scores1 = extractScoresUsingList(blockScore, list1);

        signThresholdInternalValues(blockScore, upSign, loSign, signThreshold);


        if (varThreshold != 1000){
            varThresholdInternalValues(blockScore, upVar.add(loVar), varThreshold);
        }


                // find connected components of local MatrixTools.max and average them

        List<Set<Point>> connectedComponents = BinaryConnectedComponents.detection(blockScore); // >0
/*
                CC1 = bwconncomp(B1>0);
        result = zeros(CC1.NumObjects, 7);

        for i=1:CC1.NumObjects
                [I, J] = ind2sub(CC1.ImageSize, CC1.PixelIdxList{i});
        [score, ind] = MatrixTools.max(B1(CC1.PixelIdxList{i}));
        result(i, 1:2) = [I(ind),J(ind)];
        result(i,3) = score;
        result(i,4) = Uvar(I(ind),J(ind));
        result(i,5) = Lvar(I(ind),J(ind));
        result(i,6) = -UpSign(I(ind), J(ind));
        result(i,7) = LoSign(I(ind), J(ind));
*/

    }

    private void varThresholdInternalValues(RealMatrix matrix, RealMatrix thresholdSums, double threshold) {
        for(int i = 0; i < matrix.getRowDimension(); i++){
            for(int j = 0; j < matrix.getColumnDimension(); j++){
                if(thresholdSums.getEntry(i,j) > threshold){
                    matrix.setEntry(i,j,0);
                }
            }
        }
    }

    /**
     * Threshold values in block score matrix
     * @param matrix
     * @param upSign
     * @param loSign
     * @param threshold
     */
    private void signThresholdInternalValues(RealMatrix matrix, RealMatrix upSign, RealMatrix loSign, int threshold) {
        for(int i = 0; i < matrix.getRowDimension(); i++){
            for(int j = 0; j < matrix.getColumnDimension(); j++){
                if((-upSign.getEntry(i,j)) < threshold || loSign.getEntry(i,j) < threshold){
                    matrix.setEntry(i,j,0);
                }
            }
        }
    }

    /**
     *
     * @param scoreMatrix
     * @param indexList
     * @return
     */
    private double[] extractScoresUsingList(RealMatrix scoreMatrix, List<Integer[]> indexList) {
        double[] scores = new double[indexList.size()];
        for(int i = 0; i < scores.length; i++){
            Integer[] indices = indexList.get(i);
            scores[i] = MatrixTools.calculateMax(scoreMatrix.getSubMatrix(indices[0], indices[1], indices[2], indices[3]));
        }
        return scores;
    }
}