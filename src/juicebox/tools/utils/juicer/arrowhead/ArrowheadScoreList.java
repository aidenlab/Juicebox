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

package juicebox.tools.utils.juicer.arrowhead;

import juicebox.tools.clt.Arrowhead;
import juicebox.tools.utils.common.MatrixTools;
import juicebox.track.feature.Feature2D;
import juicebox.track.feature.Feature2DList;
import org.apache.commons.math.linear.RealMatrix;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * Created by muhammadsaadshamim on 7/20/15.
 */
public class ArrowheadScoreList {

    private List<ArrowheadScore> arrowheadScores = new ArrayList<ArrowheadScore>();

    public ArrowheadScoreList() {}

    public ArrowheadScoreList(Set<int[]> indicesSet) {
        for(int[] indices : indicesSet){
            arrowheadScores.add(new ArrowheadScore(indices));
        }
    }

    private ArrowheadScoreList(List<ArrowheadScore> dataList) {
        for(ArrowheadScore data : dataList){
            arrowheadScores.add(new ArrowheadScore(data));
        }
    }

    public ArrowheadScoreList deepCopy() {
        return new ArrowheadScoreList(arrowheadScores);
    }

    public void updateActiveIndexScores(RealMatrix blockScore) {

        for (ArrowheadScore score : arrowheadScores) {
            if (score.isActive) {
                score.updateScore(MatrixTools.calculateMax(MatrixTools.getSubMatrix(blockScore, score.indices)));
            }
        }
    }

    public void setActiveListElements(int limStart, int limEnd) {
        for(ArrowheadScore score : arrowheadScores) {
            score.isActive = false;
        }

        for(ArrowheadScore score : arrowheadScores) {
            if (score.isWithin(limStart, limEnd)) {
                score.isActive = true;
            }
        }
    }

    public void addAll(ArrowheadScoreList arrowheadScoreList) {
        arrowheadScores.addAll(arrowheadScoreList.arrowheadScores);
    }

    public void mergeScores() {
        List<ArrowheadScore> mergedScores = new ArrayList<ArrowheadScore>();

        for(ArrowheadScore aScore : arrowheadScores){
            boolean valueNotFound = true;
            for(ArrowheadScore mScore : mergedScores){
                if(aScore.equivalentTo(mScore)){
                    mScore.updateScore(aScore.score);
                    valueNotFound = false;
                    break;
                }
            }

            if(valueNotFound){
                mergedScores.add(aScore);
            }
        }
        arrowheadScores = mergedScores;
    }

    public Feature2DList toFeature2DList(int chrIndex, String chrName) {
        Feature2DList feature2DList = new Feature2DList();
        for(ArrowheadScore score : arrowheadScores) {
            feature2DList.add(chrIndex, chrIndex, score.toFeature2D(chrName));
        }
        return feature2DList;
    }


    private class ArrowheadScore{
        private int[] indices = new int[4];
        private double score = Double.NaN;
        private boolean isActive = false;

        public ArrowheadScore(int[] indices){
            System.arraycopy(indices, 0, this.indices, 0, 4);
        }

        // use for deep copying
        public ArrowheadScore(ArrowheadScore arrowheadScore){
            System.arraycopy(arrowheadScore.indices, 0, this.indices, 0, 4);
            this.score = arrowheadScore.score;
            this.isActive = arrowheadScore.isActive;
        }

        public void updateScore(double score) {
            if(Double.isNaN(this.score))
                this.score = score;
            else if(!Double.isNaN(score))
                this.score = Math.max(score, this.score);
        }

        // fully contained within bounds
        public boolean isWithin(int limStart, int limEnd) {
            boolean containedInBounds = true;
            for(int index : indices){
                containedInBounds = containedInBounds && index >= limStart && index <= limEnd;
            }
            return containedInBounds;
        }

        public boolean equivalentTo(ArrowheadScore mScore) {
            return Arrays.equals(indices,mScore.indices);
        }

        public Feature2D toFeature2D(String chrName) {
            Map<String,String> attributes = new HashMap<String, String>();
            attributes.put("score",Double.toString(score));
            return new Feature2D(Feature2D.generic, chrName, indices[0], indices[1],
                    chrName, indices[2], indices[3], Color.yellow, attributes);
        }
    }
}


